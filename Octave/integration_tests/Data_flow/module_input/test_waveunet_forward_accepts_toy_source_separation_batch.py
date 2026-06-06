"""
Integration test for WaveUNet forward compatibility with toy TSE batches.

This file validates that a batch produced by the real toy source-separation
DataModule can be consumed by the WaveUNet LightningModule forward path.
"""

import copy

import numpy as np
import torch
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


#############################################
# Dummy dataset generation
#############################################


def create_dummy_source_separation_dataset(
    root,
    n_samples: int = 4,
    sample_rate: int = 8000,
    n_timesteps: int = 128,
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


#############################################
# Configs
#############################################


def build_datamodule_config(dataset_root):
    return {
        "strict": True,
        "default": {
            "strict": True,
            "datasets": {
                "train": {
                    "source_separation": {
                        "path": str(dataset_root),
                        "path_key": None,
                        "canonical_dtype": "float32",
                        "load_on_demand": False,
                        "normalize_by_mixture_peak": True,
                    }
                },
                "val": None,
                "test": None,
            },
            "collators": {
                "train": {
                    "tse_waveform": {
                        "strict": True,
                        "collator_config": {},
                        "transforms": {},
                    }
                },
                "val": None,
                "test": None,
            },
            "dataloader_configs": {
                "train": {
                    "batch_size": 2,
                    "shuffle": False,
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last": False,
                },
            },
        },
    }


def build_waveunet_module_config():
    return {
        "waveunet": copy.deepcopy(DEFAULT_WAVEUNET_LIGHTNING_MODULE_CONFIG),
    }


#############################################
# Forward compatibility
#############################################


def test_waveunet_forward_accepts_toy_source_separation_batch(tmp_path) -> None:
    dataset_root = create_dummy_source_separation_dataset(
        root=tmp_path,
        n_samples=4,
        n_timesteps=128,
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

    batch = next(iter(datamodule.train_dataloader()))
    mixture = batch["mixture"]

    with torch.no_grad():
        preds = module(mixture)

    assert isinstance(preds, torch.Tensor)
    assert preds.shape == mixture.shape
    assert preds.dtype == mixture.dtype