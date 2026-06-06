import torch

from .base import BaseDataset


class WaveformBaseDataset(BaseDataset):
    """
    Base class for datasets manipulating waveform tensors.

    Expected waveform convention:
        - single waveform: (1, T)
        - batch handling is left to DataLoaders/collate functions.

    This class provides waveform-specific helpers only.
    It does not impose any task-specific sample format.
    """

    def to_waveform_tensor(
        self,
        audio,
    ) -> torch.Tensor:
        """
        Convert audio data to a single-channel waveform tensor.

        Output shape:
            (1, T)
        """
        waveform = self.to_tensor(audio)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        return waveform
    

    ####################################################
    # Normalization helpers
    ####################################################

    @staticmethod
    def normalize_by_reference_peak(
        waveform: torch.Tensor,
        reference: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize a waveform using the peak amplitude of a reference waveform.
        """
        peak_amplitude = torch.max(torch.abs(reference)) + epsilon
        normalized_waveform = waveform / peak_amplitude

        return normalized_waveform

    @staticmethod
    def normalize_group_by_reference_peak(
        waveforms: dict[str, torch.Tensor],
        reference_key: str,
        epsilon: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        """
        Normalize several waveforms using the peak amplitude of one reference waveform.
        """
        if reference_key not in waveforms:
            raise KeyError(f"Missing reference waveform key: {reference_key}")

        reference = waveforms[reference_key]
        peak_amplitude = torch.max(torch.abs(reference)) + epsilon

        normalized_waveforms = {}

        for waveform_name, waveform in waveforms.items():
            normalized_waveforms[waveform_name] = waveform / peak_amplitude

        return normalized_waveforms