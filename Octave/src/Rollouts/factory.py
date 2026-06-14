from copy import deepcopy

from .configs import DEFAULT_LATENT_ROLLOUT_CONFIG
from .latent_rollout import LatentRollout


class LatentRolloutBuilder:
    """
    Build LatentRollout from a plain dictionary config.
    """

    def __init__(
        self,
        default_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        self.default_config = deepcopy(
            default_config
            if default_config is not None
            else DEFAULT_LATENT_ROLLOUT_CONFIG
        )
        self.strict = strict

    def __call__(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> LatentRollout:
        prepared_config = self.prepare_config(config)
        prepared_config.pop("rollout_type")
        return LatentRollout(**prepared_config)

    def prepare_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(
                "LatentRollout config must be a dictionary, "
                f"got {type(config).__name__}."
            )

        prepared_config = deepcopy(self.default_config)
        user_config = deepcopy(config)
        self.check_known_keys(user_config, prepared_config)
        prepared_config.update(user_config)
        self.check_rollout_type(prepared_config)
        return prepared_config

    def check_rollout_type(self, config: dict) -> None:
        if config["rollout_type"] != "latent":
            raise KeyError("Only 'latent' rollout_type is supported.")

    def check_known_keys(self, config: dict, default_config: dict) -> None:
        unknown_keys = set(config) - set(default_config)

        if not unknown_keys:
            return

        message = (
            f"Unknown LatentRollout config keys: {sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(default_config)}."
        )

        if self.strict:
            raise KeyError(message)


def build_latent_rollout(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> LatentRollout:
    builder = LatentRolloutBuilder(strict=strict)
    return builder(
        config=config or DEFAULT_LATENT_ROLLOUT_CONFIG,
        runtime_context=runtime_context,
    )
