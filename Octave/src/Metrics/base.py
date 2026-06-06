import torch 
from torch import nn
import numpy as np

########################
# Abstract Custom Metric
########################

# Most metric have the same code pattern so we use an abstract parent class


class BaseMetric(nn.Module):
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
    
    
    ####################################
    # Tests
    ###################################

    def check_inputs(self, *args, **kwargs) -> None:
        # Default: no checks.
        # Subclasses override this method to define the validations
        # required before metric computation.
        return None


    # Most metrics will require a specific context to be used
    # We use a test approach so that our pipeline breaks if we are out of context
    # This will help avoiding "hidden" bugs
    # To make the reading easier and centralize, we put all the context tests here


    def check_same_shape(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        # Intrusive metrics compare one estimate with one reference.
        # Shape mismatch usually means the metric comparison is ill-defined.
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape, got "
                f"{preds.shape} and {target.shape}."
            )    

    def check_single_channel(
        self,
        x: torch.Tensor,
    ) -> None:
        
        # Single-channel waveform metrics accept:
        # - one waveform: (T,)
        # - a batch of waveforms: (B, T)
        # - a batch with explicit channel dimension: (B, 1, T)
        # Multi-channel handling should happen before metric computation.

        is_single_waveform = x.ndim == 1
        is_batched_waveform = x.ndim == 2
        is_batched_single_channel = x.ndim == 3 and x.shape[1] == 1

        if not (
            is_single_waveform
            or is_batched_waveform
            or is_batched_single_channel
        ):
            raise ValueError(
                "Expected single-channel audio with shape "
                "(T,), (B, T), or (B, 1, T)."
            )
        
    
    def check_not_autograd_tracked(
        self,
        *tensors: torch.Tensor,
    ) -> None:
        # Evaluation-only metrics may rely on external libraries, NumPy code,
        # pretrained inference APIs, or non-differentiable operations.
        # If such a metric is called while autograd tracks its inputs, failing
        # explicitly is safer than silently detaching tensors and hiding misuse.
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
            raise RuntimeError(
                f"{self.__class__.__name__} is evaluation-only and cannot be used "
                "with tensors that require gradients. Use it under torch.no_grad()."
            )
        


    #########################################################
    # Helpers
    #########################################################

    def _to_single_channel_shape(self, x: torch.Tensor) -> torch.Tensor:
        # Convert explicit single-channel batches from (B, 1, T) to (B, T).
        # Already flattened single-channel inputs are left unchanged.
        if x.ndim == 3 and x.shape[1] == 1:
            return x[:, 0, :]

        return x
    

    def _to_flat_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        """
        Flatten all leading dimensions of a waveform tensor (..., T)
        into a batch dimension.

        Returns the flattened tensor and the original leading shape so
        metric outputs can later be reshaped consistently.
        """
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        return x, batch_shape
    
    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """
        Convert a tensor to a CPU NumPy array.

        Useful for evaluation-only metrics backed by NumPy/C libraries.
        Autograd checks should be done before calling this helper.
        """
        return x.detach().cpu().double().contiguous().numpy()
