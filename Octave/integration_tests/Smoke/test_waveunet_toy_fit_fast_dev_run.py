"""
Smoke test for WaveUNet toy training with Lightning fast_dev_run.

This file validates that the real DataModule, LightningModule, and Trainer
can execute one minimal fit run on a synthetic on-disk source-separation dataset.
"""

import copy

import lightning.pytorch as pl
import numpy as np
from scipy.io import wavfile

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataModules.factory import (
    build_datamodule,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.configs import (
    DEFAULT_WAVEUNET_LIGHTNING_MODULE_CONFIG,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.factory import (
    build_lightning_module,
)


def create_dummy_source_separation_dataset(
    root,
    n_samples: int = 4,
    sample_rate: int = 8000,
    n_timesteps: int = 16000,
    snr: float = 10.0,
):
    dataset_root = root / "dummy_source_separation"
    dataset_root.mkdir(parents=True, exist_ok=True)

    time = np.arange(n_timesteps, dtype=np.float32) / sample_rate

    for sample_index in range(n_samples):
        sample_dir = dataset_root / f"{sample_index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        voice = 0.5 * np.sin(2.0 * np.pi * 220.0 * time)
        noise = 0.05 * np.sin(2.0 * np.pi * 440.0 * time)
        mixture = voice + noise

        wavfile.write(sample_dir / "voice.wav", sample_rate, voice.astype(np.float32))
        wavfile.write(sample_dir / "noise.wav", sample_rate, noise.astype(np.float32))
        wavfile.write(
            sample_dir / f"mix_snr_{snr}.wav",
            sample_rate,
            mixture.astype(np.float32),
        )

    return dataset_root


def build_datamodule_config(dataset_root):
    dataset_config = {
        "source_separation": {
            "path": str(dataset_root),
            "path_key": None,
            "canonical_dtype": "float32",
            "load_on_demand": False,
            "normalize_by_mixture_peak": True,
        }
    }

    collator_config = {
        "tse_waveform": {
            "strict": True,
            "collator_config": {},
            "transforms": {},
        }
    }

    dataloader_config = {
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
    }

    return {
        "strict": True,
        "default": {
            "strict": True,
            "datasets": {
                "train": dataset_config,
                "val": dataset_config,
                "test": None,
            },
            "collators": {
                "train": collator_config,
                "val": collator_config,
                "test": None,
            },
            "dataloader_configs": {
                "train": dataloader_config,
                "val": dataloader_config,
            },
        },
    }


def build_waveunet_module_config():
    return {
        "waveunet": copy.deepcopy(DEFAULT_WAVEUNET_LIGHTNING_MODULE_CONFIG),
    }


def test_waveunet_toy_fit_fast_dev_run(tmp_path) -> None:
    dataset_root = create_dummy_source_separation_dataset(
        root=tmp_path,
        n_samples=4,
        n_timesteps=16000,
    )

    datamodule = build_datamodule(
        datamodule_configs=build_datamodule_config(dataset_root),
        runtime_context=None,
    )

    module = build_lightning_module(
        lightning_module_configs=build_waveunet_module_config(),
        runtime_context=None,
        loading_config=None,
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(
        model=module,
        datamodule=datamodule,
    )