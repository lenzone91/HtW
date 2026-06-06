import os
import time
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.vis_utils import (
    analyze_distances,
    create_comparison_gif,
    plot_losses,
    save_gif,
    show_images,
)

logger = get_logger(__name__)


def _get_gc_agent_class():
    planning_module = import_module("eb_jepa.planning")
    return getattr(planning_module, "GCAgent")


def main_unroll_eval(
    model,
    env_creator,
    eval_folder,
    num_samples=4,
    loader=None,
    prober=None,
    cfg=None,
):
    """
    Evaluate the model's unrolling capabilities by comparing unrolled predictions to ground truth.
    """
    env = env_creator()
    env.reset()
    device = next(model.parameters()).device
    normalizer = (
        loader.dataset.normalizer if hasattr(loader.dataset, "normalizer") else None
    )
    agent = _get_gc_agent_class()(
        model=model, plan_cfg=None, normalizer=normalizer, env=env, loc_prober=prober
    )
    mse_values = []
    position_mse_values = []
    unroll_times = []
    loader_iter = iter(loader)

    for idx in tqdm(
        range(num_samples), desc="Evaluating unroll", disable=cfg.logging.tqdm_silent
    ):
        try:
            x, a, loc, wall_x, door_y = next(loader_iter)
        except StopIteration:
            logger.warning(
                f"Loader exhausted after {idx} samples (requested {num_samples})"
            )
            break

        x = x.to(device)
        a = a.to(device)
        with torch.no_grad():
            obs_init = x[:, :, 0:1]  # B C T H W
            start_time = time.time()
            predicted_states = agent.unroll(obs_init, a, repeat_batch=False)[
                :, :, :-1
            ]  # discard last predicted state
            end_time = time.time()
            unroll_times.append(end_time - start_time)
            rand_predicted_states = agent.unroll(
                obs_init, torch.randn_like(a), repeat_batch=False
            )[
                :, :, :-1
            ]  # B D T H W
            # To ensure independence across timesteps when encoding the sequence, batchify it
            # There is no independence between timesteps when using GroupNorm, even in eval mode
            B, C, T, H, W = x.shape
            gt_encoded = (
                model.encode(x.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2))
                .squeeze(2)
                .unflatten(dim=0, sizes=(B, -1))
                .permute(0, 2, 1, 3, 4)
            )
            latent_mse = (
                ((gt_encoded - predicted_states) ** 2).mean(dim=(1, 3, 4)).cpu().numpy()
            )  # B T
            mse_values.append(latent_mse)

            if prober:
                gt_decoded = agent.decode_loc_to_pixel(gt_encoded, wall_x, door_y)
                pred_decoded = agent.decode_loc_to_pixel(
                    predicted_states, wall_x, door_y
                )
                rand_pred_decoded = agent.decode_loc_to_pixel(
                    rand_predicted_states, wall_x, door_y
                )  # B T H W C
                gt_frames = agent.normalizer.unnormalize_state(
                    x.permute(0, 2, 1, 3, 4)
                ).permute(0, 1, 3, 4, 2)
                gt_frames = (
                    (gt_frames * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                )  # B T H W C uint8

                pred_positions = (
                    prober.apply_head(predicted_states).permute(0, 2, 1).cpu()
                )  # B T 2
                gt_positions = loc.permute(0, 2, 1)  # B T 2
                position_mse = (
                    ((pred_positions - gt_positions.cpu()) ** 2)
                    .mean(dim=-1)
                    .cpu()
                    .numpy()
                )  # B T
                position_mse_values.append(position_mse)

                create_comparison_gif(
                    gt_frames,
                    pred_decoded,
                    rand_pred_decoded,
                    gt_dec=gt_decoded,
                    save_path=f"{eval_folder}/b{idx}.gif",
                )
    all_mse_values = np.vstack(mse_values)  # Shape: [num_batches, T]
    mean_mse_per_timestep = np.mean(all_mse_values, axis=0)  # Shape: [T]
    std_mse_per_timestep = np.std(all_mse_values, axis=0)  # Shape: [T]
    avg_unroll_time = np.mean(unroll_times)
    results = {}
    for t in range(mean_mse_per_timestep.shape[0]):
        results[f"val_rollout/mean_mse/{t}"] = mean_mse_per_timestep[t]
        results[f"val_rollout/std_mse/{t}"] = std_mse_per_timestep[t]

    if len(position_mse_values) > 0:
        all_position_mse_values = np.vstack(
            position_mse_values
        )  # Shape: [num_batches, T]
        mean_position_mse_per_timestep = np.mean(
            all_position_mse_values, axis=0
        )  # Shape: [T]
        std_position_mse_per_timestep = np.std(
            all_position_mse_values, axis=0
        )  # Shape: [T]
        for t in range(mean_position_mse_per_timestep.shape[0]):
            results[f"val_rollout/mean_pos_mse/{t}"] = mean_position_mse_per_timestep[t]
            results[f"val_rollout/std_pos_mse/{t}"] = std_position_mse_per_timestep[t]

    results["avg_unroll_time"] = avg_unroll_time

    pd.DataFrame([results]).to_csv(f"{eval_folder}/eval.csv", index=None)
    return results


def main_eval(
    plan_cfg,
    model,
    env_creator,
    eval_folder,
    num_episodes=10,
    loader=None,
    prober=None,
):
    plan_cfg = OmegaConf.create(plan_cfg)
    env = env_creator()
    env.reset()

    action_dim = env.action_space.shape[0] if hasattr(env, "action_space") else 2
    agent = _get_gc_agent_class()(
        model,
        action_dim=action_dim,
        plan_cfg=plan_cfg,
        normalizer=env.normalizer,
        loc_prober=prober,
        env=env,
    )
    logger.info(f"Agent created with planner {agent.planner.__class__.__name__}")
    logger.info(f"Planning with {plan_cfg=}")

    successes = []
    distances = []
    episode_times = []
    episode_observations = []
    episode_infos = []

    for ep in range(num_episodes):
        episode_start_time = time.time()
        ep_folder = eval_folder / f"ep_{ep}"
        os.makedirs(ep_folder, exist_ok=True)
        if agent.decode_each_iteration:
            ep_plan_vis_dir = ep_folder / "plan_vis"
            os.makedirs(ep_plan_vis_dir, exist_ok=True)

        if plan_cfg.task_specification.goal_source == "dset":
            obs_slice, a, loc, _, _ = next(iter(loader))
            # obs, init_loc = obs_slice[0], loc[0]
            # goal_img, goal_loc = obs_slice[-1], loc[-1]  # [C, H, W] uint8 tensor
            # env.set_goal(goal_img) # Set goal in the environment
        elif plan_cfg.task_specification.goal_source == "random_state":
            obs, info = env.reset()  # [C, H, W] uint8 tensor
            obs, reward, done, truncated, info = env.step(
                np.zeros(env.action_space.shape[0])
            )  # step with zero action to get the first observation
            goal_img = info["target_obs"]  # [C, H, W] uint8 tensor

        combined = torch.stack([obs, goal_img], dim=0)
        show_images(
            combined,
            nrow=2,  # Both images in one row
            titles=["Init", "Goal"],
            save_path=f"{ep_folder}/state.pdf",
            close_fig=True,
            first_channel_only=False,
            clamp=False,
        )
        agent.set_goal(
            goal_img.detach().clone().to(dtype=torch.float32),
            info["target_position"],
        )

        done = False
        steps_left = env.n_allowed_steps
        pbar = tqdm(
            desc="executing agent",
            total=steps_left,
            leave=True,
            disable=plan_cfg.logging.tqdm_silent,
        )
        t0 = True

        observations = [obs]
        infos = [info]

        prev_losses = []
        prev_elite_losses_mean = []
        prev_elite_losses_std = []

        while steps_left > 0:
            plan_vis_path = (
                f"{ep_plan_vis_dir}/step{env.n_allowed_steps - steps_left}"
                if agent.decode_each_iteration
                else None
            )
            obs_tensor = (
                env.normalizer.normalize_state(
                    obs.detach().clone().to(dtype=torch.float32, device=agent.device)
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )  # Unsqueeze the batch and time dimensions : C H W -> 1 C 1 H W
            with torch.no_grad():
                action = (
                    agent.act(
                        obs_tensor,
                        steps_left=steps_left,
                        t0=t0,
                        plan_vis_path=plan_vis_path,
                    )
                    .cpu()
                    .numpy()
                )  # T, A
            if agent._prev_losses is not None:
                prev_losses.append(agent._prev_losses)
                prev_elite_losses_mean.append(agent._prev_elite_losses_mean)
                prev_elite_losses_std.append(agent._prev_elite_losses_std)
            for a in action:
                obs, reward, done, truncated, info = env.step(a)
                t0 = False
                observations.append(obs)
                infos.append(info)
                steps_left -= 1
                pbar.update(1)
                eval_results = env.eval_state(
                    info["target_position"], info["dot_position"]
                )
                success = eval_results["success"]
                state_dist = eval_results["state_dist"]
            pbar.set_postfix({"success": success, "state_dist": state_dist})
        pbar.close()

        episode_observations.append(torch.stack(observations))
        episode_infos.append(infos)
        successes.append(success)
        distances.append(state_dist)

        if plan_cfg.logging.get("optional_plots", True):
            analyze_distances(
                episode_observations[-1],
                episode_infos[-1],
                str(ep_folder / "agent"),
                goal_position=agent.goal_position,
                goal_state=agent.goal_state,
                normalizer=agent.normalizer,
                model=agent.model,
                objective=agent.objective,
                device=agent.device,
            )
            plot_losses(
                prev_losses,
                prev_elite_losses_mean,
                prev_elite_losses_std,
                work_dir=ep_folder,
                num_act_stepped=agent.num_act_stepped,
            )
            save_path = f"{ep_folder}/agent_steps_{'succ' if success else 'fail'}.gif"
        save_gif(
            episode_observations[-1],
            save_path=save_path,
            show_frame_numbers=True,
            fps=20,
            init_frame=observations[0],
            goal_frame=goal_img,
        )
        logger.info(f"GIF saved to {save_path}")
        episode_end_time = time.time()
        episode_times.append(episode_end_time - episode_start_time)
    avg_episode_time = np.mean(episode_times)
    task_data = {
        "success_rate": np.mean(successes),
        "mean_state_dist": np.mean(distances),
        "avg_episode_time": avg_episode_time,
    }
    pd.DataFrame([task_data]).to_csv(f"{eval_folder}/eval.csv", mode="a", index=None)
    return task_data
