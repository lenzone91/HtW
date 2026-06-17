import numpy as np
import torch
from torch import nn


#############################################
# Base metric
#############################################


class BaseMetric(nn.Module):
    """
    Abstract parent for metrics.

    Owns the generic, modality-agnostic metric machinery:
        - reduction handling (mean / sum / none);
        - an overridable input-validation hook;
        - shared shape and autograd checks;
        - a NumPy conversion helper for evaluation-only metrics.

    It defines no default config and is not registered. Concrete metrics
    subclass it and register themselves.

    Audio-specific helpers from Project_2 (single-channel waveform checks and
    waveform reshaping) are intentionally NOT here; they belong to an audio
    metric base in the Audio pillar (Phase 3).
    """

    valid_reductions = {"mean", "sum", "none"}

    def __init__(self, reduction: str = "mean"):
        super().__init__()

        if reduction not in self.valid_reductions:
            raise ValueError(
                f"Invalid reduction: {reduction}. "
                f"Expected one of {self.valid_reductions}."
            )

        self.reduction = reduction

    def reduce(self, values: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return values.mean()

        if self.reduction == "sum":
            return values.sum()

        return values

    #############################################
    # Validation hooks / helpers
    #############################################

    def check_inputs(self, *args, **kwargs) -> None:
        """
        Default: no checks. Subclasses override to validate before computation.
        """
        return None

    def check_same_shape(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Intrusive metrics compare one estimate to one reference; a shape
        mismatch usually means the comparison is ill-defined.
        """
        if preds.shape != target.shape:
            raise ValueError(
                "preds and target must have the same shape, got "
                f"{preds.shape} and {target.shape}."
            )

    def check_not_autograd_tracked(self, *tensors: torch.Tensor) -> None:
        """
        Evaluation-only metrics (NumPy / pretrained inference / non-differentiable
        ops) must not run while autograd tracks their inputs. Failing explicitly
        is safer than silently detaching.
        """
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise RuntimeError(
                f"{self.__class__.__name__} is evaluation-only and cannot be used "
                "with tensors that require gradients. Use it under torch.no_grad()."
            )

    #############################################
    # Helpers
    #############################################

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """
        Convert a tensor to a CPU NumPy array (for NumPy/C-backed metrics).
        Autograd checks should happen before calling this.
        """
        return x.detach().cpu().double().contiguous().numpy()
