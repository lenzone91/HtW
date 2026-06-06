from .base import BaseBatchTransform

import torch

    


##################################################
# Add noise with controlled snr transform
##################################################

class AddNoiseAtSNR(BaseBatchTransform):
    """
    Add Gaussian noise to a waveform batch with controlled SNR.

    Expected input:
        batch[input_key]: torch.Tensor

    Typical shapes:
        (B, T)
        (B, C, T)

    The transformed signal is written to batch[output_key].
    If output_key == input_key, the original value is replaced.
    """

    def __init__(
        self,
        input_key: str = "mixture",
        output_key: str = "mixture",
        snr_db: float = 20.0,
        eps: float = 1e-8,
    ) -> None:
        self.input_key = input_key
        self.output_key = output_key
        self.snr_db = snr_db
        self.eps = eps

    def transform(self, batch: dict) -> dict:
        self.check_required_key(batch, self.input_key)

        signal = batch[self.input_key]
        self.check_is_batched_tensor(signal, name=self.input_key)

        noise = torch.randn_like(signal)

        signal_power = self.compute_power(signal)
        noise_power = self.compute_power(noise)

        target_noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        noise_scale = torch.sqrt(target_noise_power / (noise_power + self.eps))

        batch[self.output_key] = signal + noise_scale * noise

        return batch

    ############################################################################
    # Helpers
    ############################################################################

    def compute_power(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample power while preserving broadcastable dimensions.
        """
        reduce_dims = tuple(range(1, signal.ndim))
        return signal.pow(2).mean(dim=reduce_dims, keepdim=True)