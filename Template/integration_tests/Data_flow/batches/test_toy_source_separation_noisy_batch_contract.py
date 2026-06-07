"""
Integration test for the noisy toy source-separation batch contract.

This file validates that the real DataModule factory can build a toy
source-separation DataModule whose collator includes a batch transform, and
that the transformed batch still follows the expected TSE batch convention.
"""

import numpy as np
import torch
from scipy.io import wavfile

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataModules.factory import (
    build_datamodule,
)


#############################################
# Dummy dataset generation
#############################################


def create_dummy_source_separation_dataset(
    root,
    n_samples: int = 4,
    sample_rate: int = 8000,
    n_timesteps: int = 64,
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
# Config
#############################################


def build_noisy_datamodule_config(dataset_root):
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
                        "transforms": {
                            "add_noise_at_snr": {
                                "input_key": "mixture",
                                "output_key": "mixture",
                                "snr_db": 20.0,
                                "eps": 1e-8,
                            }
                        },
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


#############################################
# Noisy batch contract
#############################################


def test_toy_source_separation_noisy_batch_contract(tmp_path) -> None:
    torch.manual_seed(0)

    dataset_root = create_dummy_source_separation_dataset(
        root=tmp_path,
        n_samples=4,
        n_timesteps=64,
    )

    datamodule_config = build_noisy_datamodule_config(dataset_root)

    datamodule = build_datamodule(
        datamodule_configs=datamodule_config,
        runtime_context=None,
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    assert isinstance(batch, dict)
    assert set(batch.keys()) == {"mixture", "target", "clue", "metadata"}

    assert isinstance(batch["mixture"], torch.Tensor)
    assert isinstance(batch["target"], torch.Tensor)

    assert batch["mixture"].shape == (2, 1, 64)
    assert batch["target"].shape == (2, 1, 64)

    assert batch["mixture"].dtype == torch.float32
    assert batch["target"].dtype == torch.float32

    assert batch["clue"] is None

    assert isinstance(batch["metadata"], list)
    assert len(batch["metadata"]) == 2
    assert all(isinstance(metadata, dict) for metadata in batch["metadata"])

    assert not torch.allclose(batch["mixture"], batch["target"])