import torch
from scipy.io import wavfile
from tqdm import tqdm
import os
from typing import Any

from .tse_base import BaseTSEDataset
from .waveform_base import WaveformBaseDataset


class SourceSeparationDataset(WaveformBaseDataset, BaseTSEDataset):
    """
    Toy source-separation dataset used to sanity-check the ML pipeline.

    Expected folder structure:
        path_to_folder/
            data_id_1/
                *noise*.wav
                *voice*.wav
                *mix*.wav
            data_id_2/
                ...

    Returned TSE sample convention:
        {
            "mixture": mixture,
            "target": voice,
            "clue": None,
            "metadata": {
                "data_id": ...,
                "snr": ...,
                "noise": ...,
            },
        }

    Notes:
        - target is the clean voice;
        - noise is kept in metadata for debugging/evaluation;
        - waveform tensors follow shape (1, T);
        - optional peak normalization uses the mixture peak as reference.
    """

    def __init__(
        self,
        path_to_folder: str,
        canonical_dtype: torch.dtype = torch.float32,
        load_on_demand: bool = False,
        normalize_by_mixture_peak: bool = True,
    ) -> None:
        super().__init__(canonical_dtype=canonical_dtype)

        if not os.path.isdir(path_to_folder):
            raise ValueError(f"<{path_to_folder}> is not an existing directory.")

        self.path_to_folder = path_to_folder
        self.load_on_demand = load_on_demand
        self.normalize_by_mixture_peak = normalize_by_mixture_peak

        self.data_ids = sorted(os.listdir(self.path_to_folder))

        if self.load_on_demand:
            self.data = None
        else:
            self.data = self.load_all_samples()

    def __len__(self) -> int:
        return len(self.data_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.load_on_demand:
            data_id = self.data_ids[index]
            sample = self.load_sample(data_id)
            return sample

        sample = self.data[index]
        return sample

    def load_all_samples(self) -> list[dict[str, Any]]:
        samples = []

        for data_id in tqdm(self.data_ids, total=len(self.data_ids)):
            sample = self.load_sample(data_id)
            samples.append(sample)

        return samples

    def load_sample(
        self,
        data_id: str,
    ) -> dict[str, Any]:
        loaded_wavs = self.load_wav_files(data_id)

        self.check_loaded_wavs(
            loaded_wavs=loaded_wavs,
            data_id=data_id,
        )

        noise = self.to_waveform_tensor(loaded_wavs["noise_audio"])
        voice = self.to_waveform_tensor(loaded_wavs["voice_audio"])
        mixture = self.to_waveform_tensor(loaded_wavs["mixture_audio"])

        if self.normalize_by_mixture_peak:
            waveforms = {
                "mixture": mixture,
                "voice": voice,
                "noise": noise,
            }

            normalized_waveforms = self.normalize_group_by_reference_peak(
                waveforms=waveforms,
                reference_key="mixture",
            )

            mixture = normalized_waveforms["mixture"]
            voice = normalized_waveforms["voice"]
            noise = normalized_waveforms["noise"]

        metadata = {
            "data_id": data_id,
            "snr": loaded_wavs["snr"],
        }

        sample = self.build_tse_sample(
            mixture=mixture,
            target=voice,
            clue=None,
            metadata=metadata,
        )

        return sample

    def load_wav_files(
        self,
        data_id: str,
    ) -> dict[str, Any]:
        path_to_data = os.path.join(self.path_to_folder, data_id)

        noise_audio = None
        voice_audio = None
        mixture_audio = None
        snr = None

        for filename in os.listdir(path_to_data):
            path_to_file = os.path.join(path_to_data, filename)

            if "noise" in filename:
                _, noise_audio = wavfile.read(path_to_file)

            if "voice" in filename:
                _, voice_audio = wavfile.read(path_to_file)

            if "mix" in filename:
                _, mixture_audio = wavfile.read(path_to_file)
                snr = self.extract_snr_from_mix_filename(filename)

        return {
            "noise_audio": noise_audio,
            "voice_audio": voice_audio,
            "mixture_audio": mixture_audio,
            "snr": snr,
        }

    @staticmethod
    def extract_snr_from_mix_filename(
        filename: str,
    ) -> float:
        filename_without_extension = filename.removesuffix(".wav")
        snr_as_string = filename_without_extension.split("_")[-1]

        snr = float(snr_as_string)

        return snr
    

    def check_loaded_wavs(
        self,
        loaded_wavs: dict[str, Any],
        data_id: str,
    ) -> None:
        """
        Check that all required wav entries were loaded for one data id.
        """
        required_keys = (
            "noise_audio",
            "voice_audio",
            "mixture_audio",
            "snr",
        )

        missing_keys = [
            key
            for key in required_keys
            if loaded_wavs.get(key) is None
        ]

        if missing_keys:
            raise ValueError(
                f"Missing required wav entries for data_id='{data_id}': "
                f"{missing_keys}."
            )