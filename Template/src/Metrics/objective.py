import torch
from torch import nn
from .base import BaseMetric




###############################
# SNR
###############################

# Already exists we wrap it to fit our metric framework

class SNR(BaseMetric):
    """
    Signal-to-Noise Ratio.

    Intrusive metric:
        preds  = estimated/extracted speech
        target = clean reference target speech

    Differentiable TorchMetrics wrapper.
    """

    def __init__(
        self,
        zero_mean: bool = False,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.zero_mean = zero_mean

        # Optional dependency kept out of module-level imports.
        from torchmetrics.functional.audio import signal_noise_ratio

        self._signal_noise_ratio = signal_noise_ratio

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_inputs(preds, target)

        preds = self._to_single_channel_shape(preds)
        target = self._to_single_channel_shape(target)

        values = self._signal_noise_ratio(
            preds=preds,
            target=target,
            zero_mean=self.zero_mean,
        )

        return self.reduce(values)

    def check_inputs(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.check_same_shape(preds, target)
        self.check_single_channel(preds)
        self.check_single_channel(target)


###############################
# SI-... Abstract class
###############################

# The SI-... framework relies on computing residuals based on orthogonal projection. To avoid redundancies between the 
# custom SI-... metrics we use an abstract parent class


class ScaleInvariantMetric(BaseMetric):
    """
    Shared geometric decomposition for scale-invariant TSE metrics.

    Common convention:
        preds   = estimated/extracted speech, \\hat{s}
        target  = clean reference target speech, s
        mixture = input mixture, x = s + n

    This parent handles only the common SI geometry.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.eps = eps

    def _proj(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(x * y, dim=-1, keepdim=True)
            / (torch.sum(y ** 2, dim=-1, keepdim=True) + self.eps)
        ) * y

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x ** 2, dim=-1)

    def _prepare_inputs(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = self._to_single_channel_shape(preds)
        target = self._to_single_channel_shape(target)
        mixture = self._to_single_channel_shape(mixture)

        return preds, target, mixture

    def _si_decomposition(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        preds, target, mixture = self._prepare_inputs(preds, target, mixture)

        noise = mixture - target

        e_target = self._proj(preds, target)
        e_res = e_target - preds

        # Since e_res is orthogonal to target, the projection onto span(target, noise)
        # reduces to the projection onto the component of noise orthogonal to target.
        noise_orth = noise - self._proj(noise, target)

        e_interf = self._proj(e_res, noise_orth)
        e_artif = e_res - e_interf

        return e_target, e_res, e_interf, e_artif

    def check_inputs(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> None:
        self.check_same_shape(preds, target)
        self.check_same_shape(preds, mixture)

        self.check_single_channel(preds)
        self.check_single_channel(target)
        self.check_single_channel(mixture)

###############################
# SI-SDR
###############################

# Already exists we just wrap it into our framework

class SISDR(BaseMetric):
    """
    Scale-Invariant Signal-to-Distortion Ratio.

    Intrusive metric:
        preds  = estimated/extracted speech
        target = clean reference target speech

    Differentiable TorchMetrics wrapper.
    """

    def __init__(
        self,
        zero_mean: bool = False,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.zero_mean = zero_mean

        # Optional dependency kept out of module-level imports.
        from torchmetrics.functional.audio import (
            scale_invariant_signal_distortion_ratio,
        )

        self._si_sdr = scale_invariant_signal_distortion_ratio

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_inputs(preds, target)

        preds = self._to_single_channel_shape(preds)
        target = self._to_single_channel_shape(target)

        values = self._si_sdr(
            preds=preds,
            target=target,
            zero_mean=self.zero_mean,
        )

        return self.reduce(values)

    def check_inputs(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.check_same_shape(preds, target)
        self.check_single_channel(preds)
        self.check_single_channel(target)

################################
# SI-SIR
################################

# Does not seem to exist so we use a custom object that inherits from our SI-... framework

class SISIR(ScaleInvariantMetric):
    """
    Scale-Invariant Signal-to-Interference Ratio.

    Intrusive TSE metric:
        preds   = estimated/extracted speech, \\hat{s}
        target  = clean reference target speech, s
        mixture = input mixture, x = s + n

    Differentiable pure PyTorch metric.
    """

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        self.check_inputs(preds, target, mixture)

        e_target, _, e_interf, _ = self._si_decomposition(
            preds=preds,
            target=target,
            mixture=mixture,
        )

        values = 10.0 * torch.log10(
            (self._energy(e_target) + self.eps)
            / (self._energy(e_interf) + self.eps)
        )

        return self.reduce(values)
    


################################
# SI-SAR
################################

# Does not seem to exist

class SISAR(ScaleInvariantMetric):
    """
    Scale-Invariant Signal-to-Artifact Ratio.

    Intrusive TSE metric:
        preds   = estimated/extracted speech, \\hat{s}
        target  = clean reference target speech, s
        mixture = input mixture, x = s + n

    Differentiable pure PyTorch metric.
    """

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        self.check_inputs(preds, target, mixture)

        e_target, _, _, e_artif = self._si_decomposition(
            preds=preds,
            target=target,
            mixture=mixture,
        )

        values = 10.0 * torch.log10(
            (self._energy(e_target) + self.eps)
            / (self._energy(e_artif) + self.eps)
        )

        return self.reduce(values)
    
##################################
# SD-SDR
##################################

# Does not seem to exist

class SDSDR(BaseMetric):
    """
    Scale-Dependent Signal-to-Distortion Ratio.

    Intrusive metric:
        preds  = estimated/extracted speech, \\hat{s}
        target = clean reference target speech, s

    Unlike SI-SDR, this penalizes wrong amplitude scaling.
    Differentiable pure PyTorch metric.
    """

    def __init__(
        self,
        mu: float = 1.0,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.mu = mu
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_inputs(preds, target)

        preds = self._to_single_channel_shape(preds)
        target = self._to_single_channel_shape(target)

        target_energy = torch.sum((self.mu * target) ** 2, dim=-1)
        error_energy = torch.sum((target - preds) ** 2, dim=-1)

        values = 10.0 * torch.log10(
            (target_energy + self.eps)
            / (error_energy + self.eps)
        )

        return self.reduce(values)

    def check_inputs(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.check_same_shape(preds, target)
        self.check_single_channel(preds)
        self.check_single_channel(target)
    

#########################################
# DTW
#########################################

# Does not exist in a torch friendly implementation package
# But not necessary to be torch-friendly. DTW is computationnally heavy and should not be used during traning. 
# Hence there is no need for gradient tracking.
# We can use the numpy-friendly implementation instead and just wrap it

# from dtaidistance import dtw

# distance = dtw.distance(x, y)

# Wrapper so that we can "mindlessly pass tensors" to dtaidistance's dtw DURING EVAL ONLY
class DTW(BaseMetric):
    """
    Dynamic Time Warping distance.

    Intrusive metric:
        preds  = estimated/extracted speech
        target = clean reference target speech

    Evaluation-only, non-differentiable.
    Uses dtaidistance internally.
    """

    def __init__(
        self,
        reduction: str = "mean",
        use_pruning: bool = True,
    ):
        super().__init__(reduction=reduction)
        self.use_pruning = use_pruning

        # Optional dependency kept out of module-level imports.
        from dtaidistance import dtw

        self._dtw_distance_fast = dtw.distance_fast

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_inputs(preds, target)

        preds = self._to_single_channel_shape(preds)
        target = self._to_single_channel_shape(target)

        device = preds.device
        dtype = preds.dtype

        preds_flat, batch_shape = self._to_flat_batch(preds)
        target_flat, _ = self._to_flat_batch(target)

        preds_np = self._to_numpy(preds_flat)
        target_np = self._to_numpy(target_flat)

        values = [
            self._dtw_distance_fast(
                pred,
                tgt,
                use_pruning=self.use_pruning,
            )
            for pred, tgt in zip(preds_np, target_np)
        ]

        values = torch.tensor(values, device=device, dtype=dtype)
        values = values.reshape(batch_shape)

        return self.reduce(values)

    def check_inputs(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.check_same_shape(preds, target)
        self.check_single_channel(preds)
        self.check_single_channel(target)
        self.check_not_autograd_tracked(preds, target)


###################################
# L_p Norms
###################################

class LPNorm(BaseMetric):
    """
    L_p norm wrapper.

    Generic metric:
        x = tensor to measure, usually preds - target for an error norm.

    Differentiable pure PyTorch metric.
    """

    def __init__(
        self,
        p: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check_inputs(x)

        x = self._to_single_channel_shape(x)

        values = torch.linalg.vector_norm(
            x,
            ord=self.p,
            dim=-1,
        )

        return self.reduce(values)

    def check_inputs(self, x: torch.Tensor) -> None:
        self.check_single_channel(x)

#####################################
# L_p error
#####################################

class LPError(LPNorm):
    """
    L_p error norm.

    Computes ||target - preds||_p along the time dimension.
    """

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._check_inputs(preds, target)
        return super().forward(target - preds)

    # We add a _ before the method name to avoid conflict with LPNorm's check_inputs
    def _check_inputs(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.check_same_shape(preds, target)
        # The single_channel test is performed by the LPNorm's forward



####################################
# Spectral  metrics abstract class
####################################

# Most spectral metrics rely on computing fourier related stuff. 
# We centralize this in an abstract parent class

class SpectralMetric(BaseMetric):
    """
    Shared STFT/spectral utilities for spectral metrics.

    Expected waveform shapes:
        (T,), (B, T), or (B, 1, T)

    Pure PyTorch, so differentiable unless a child metric says otherwise.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.eps = eps

        self.register_buffer("_window", torch.empty(0), persistent=False)

    def _get_window(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.window is None:
            return None

        win_length = self.win_length or self.n_fft

        if (
            self._window.numel() != win_length
            or self._window.device != x.device
            or self._window.dtype != x.dtype
        ):
            if self.window == "hann":
                self._window = torch.hann_window(
                    win_length,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                raise ValueError(f"Unsupported window: {self.window}")

        return self._window

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_single_channel_shape(x)

        x_flat, batch_shape = self._to_flat_batch(x)

        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._get_window(x),
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )

        return spec.reshape(*batch_shape, *spec.shape[-2:])

    def _magnitude_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        return self._stft(x).abs()

    def _power_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        return self._magnitude_spectrogram(x).square()

    def _log_magnitude_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self._magnitude_spectrogram(x) + self.eps)

    def _log_power_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self._power_spectrogram(x) + self.eps)

    def _power_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Time-averaged power spectrum.

        This is not a physically normalized PSD. It is sufficient for
        normalized spectral divergences where global constants cancel.
        """
        return self._power_spectrogram(x).mean(dim=-1)

    def check_inputs(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.check_same_shape(preds, target)
        self.check_single_channel(preds)
        self.check_single_channel(target)
    

############################
# LSD Log-Spectral Distance
############################

class LSD(SpectralMetric):
    """
    Log-Spectral Distance.

    Intrusive spectral metric:
        preds  = estimated/extracted speech
        target = clean reference target speech

    Differentiable pure PyTorch metric.
    """

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_inputs(preds, target)

        log_pred = self._log_magnitude_spectrogram(preds)
        log_target = self._log_magnitude_spectrogram(target)

        values = torch.sqrt(
            (log_pred - log_target).square().mean(dim=(-2, -1)) + self.eps
        )

        return self.reduce(values)


#############################
# Spectral KL-divergence
#############################

class SpectralKLDivergence(SpectralMetric):
    """
    Spectral Kullback-Leibler divergence.

    Intrusive spectral metric:
        preds  = estimated/extracted speech
        target = clean reference target speech

    Uses time-averaged power spectra, not full spectrograms.

    If normalize=True, computes a true KL divergence between normalized
    spectral distributions. Otherwise computes the unnormalized spectral
    divergence.
    """

    def __init__(
        self,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normalize = normalize

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_inputs(preds, target)

        pred_spectrum = self._power_spectrum(preds) + self.eps
        target_spectrum = self._power_spectrum(target) + self.eps

        if self.normalize:
            pred_spectrum = pred_spectrum / pred_spectrum.sum(dim=-1, keepdim=True)
            target_spectrum = target_spectrum / target_spectrum.sum(dim=-1, keepdim=True)

        values = (
            target_spectrum
            * (torch.log(target_spectrum) - torch.log(pred_spectrum))
        ).sum(dim=-1)

        return self.reduce(values)
    

##############################
# Itakura-Saito divergence
##############################

class ItakuraSaitoDivergence(SpectralMetric):
    """
    Itakura-Saito divergence.

    Intrusive spectral metric:
        preds  = estimated/extracted speech
        target = clean reference target speech

    Uses time-averaged power spectra, not full spectrograms.
    Differentiable pure PyTorch metric.
    """

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_inputs(preds, target)

        pred_spectrum = self._power_spectrum(preds) + self.eps
        target_spectrum = self._power_spectrum(target) + self.eps

        ratio = target_spectrum / pred_spectrum

        values = (ratio - torch.log(ratio) - 1.0).sum(dim=-1)

        return self.reduce(values)