import torch
from .base import BaseMetric



############################
# STOI
############################ 

# STOI already exists but does not behave like an usual metric
# This wrapper does the adaptation

# Additionnally, STOI is not compatible with gradient tracking so we add an explicit test to check whether the 
# tensors that are processed have their gradient tracked or not

# STOI expect audio with one single channel so we add an explicit test to check this.

class STOI(BaseMetric):
    """
    Short-Time Objective Intelligibility.

    Intrusive metric:
        preds  = estimated/extracted speech
        target = clean reference target speech

    Evaluation-only, non-differentiable.
    """

    def __init__(
        self,
        sample_rate: int,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.sample_rate = sample_rate
        self.extended = False
        
        # Put heavy imports in the __init__
        from torchmetrics.functional.audio.stoi import (
            short_time_objective_intelligibility
        )
        self.stoi = short_time_objective_intelligibility
        

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        

        self.check_inputs(preds, target)
        preds = self._to_single_channel_shape(preds)
        target = self._to_single_channel_shape(target)

        values = self.stoi(
            preds=preds.detach(),
            target=target.detach(),
            fs=self.sample_rate,
            extended=self.extended,
            keep_same_device=True,
        )

        return self.reduce(values)
    
    def check_inputs(self, preds: torch.Tensor, target: torch.Tensor,) -> None:
        self.check_same_shape(preds, target)
        self.check_single_channel(preds)
        self.check_single_channel(target)
        self.check_not_autograd_tracked(preds, target)
        

###########################################
# ESTOI
###########################################

# Same exact package and function as STOI, the extended flag just needs to be set to True

class ESTOI(STOI):
    """
    Extended Short-Time Objective Intelligibility.

    Same backend as STOI, with extended=True.
    Evaluation-only.
    """

    def __init__(
        self,
        sample_rate: int,
        reduction: str = "mean",
    ):
        super().__init__(
            sample_rate=sample_rate,
            reduction=reduction,
        )
        self.extended = True



#########################################
# PESQ
#########################################

# Already exists in a clean form but requires additional tests for context

class PESQ(BaseMetric):
    def __init__(
        self,
        sample_rate: int,
        mode: str = "wb",
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)

        if sample_rate not in {8000, 16000}:
            raise ValueError(
                f"PESQ expects sample_rate to be 8000 or 16000, got {sample_rate}."
            )

        if mode not in {"nb", "wb"}:
            raise ValueError(
                f"PESQ expects mode to be 'nb' or 'wb', got {mode}."
            )

        from torchmetrics.functional.audio.pesq import (
            perceptual_evaluation_speech_quality,
        )
        

        self.pesq = perceptual_evaluation_speech_quality
        self.sample_rate = sample_rate
        self.mode = mode

    def check_inputs(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        self.check_same_shape(preds, target)
        self.check_single_channel(preds)
        self.check_single_channel(target)
        self.check_not_autograd_tracked(preds, target)

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        self.check_inputs(preds, target)

        preds = self._to_single_channel_shape(preds)   # (B, 1, T) -> (B,T) or (B,T) -> (B,T)
        target = self._to_single_channel_shape(target)

        values = self.pesq(
            preds=preds.detach(),
            target=target.detach(),
            fs=self.sample_rate,
            mode=self.mode,
            keep_same_device=True,
        )

        return self.reduce(values)



###########################################
# DNSMOS
###########################################

# Already exists in a clean form we wrap it to perform tests and etc
# This implementation does DNSMOS P808 and DNSMOS P835
# index 0 = DNSMOS P808 
# indexes 1 to 4 = DNSMOS P835 , aka SIG, BAK, OVRL

class BaseDNSMOS(BaseMetric):
    def __init__(
        self,
        sample_rate: int,
        personalized: bool = False,
        device: str | None = None,
        num_threads: int | None = None,
        cache_session: bool = True,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)

        from torchmetrics.functional.audio.dnsmos import (
            deep_noise_suppression_mean_opinion_score,
        )

        self.dnsmos = deep_noise_suppression_mean_opinion_score
        self.sample_rate = sample_rate
        self.personalized = personalized
        self.device = device
        self.num_threads = num_threads
        self.cache_session = cache_session

    def check_inputs(self, preds: torch.Tensor) -> None:
        self.check_single_channel(preds)
        self.check_not_autograd_tracked(preds)

    def compute_scores(self, preds: torch.Tensor) -> torch.Tensor:
        self.check_inputs(preds)

        preds = self._to_single_channel_shape(preds)

        return self.dnsmos(
            preds=preds.detach(),
            fs=self.sample_rate,
            personalized=self.personalized,
            device=self.device,
            num_threads=self.num_threads,
            cache_session=self.cache_session,
        )

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        scores = self.compute_scores(preds)
        return self.reduce(scores)
    

##########################
# DNSMOS P808
#########################


class DNSMOSP808(BaseDNSMOS):
    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        scores = self.compute_scores(preds)
        return self.reduce(scores[..., 0])
    
################################
# DNSMOS P835
################################

class DNSMOSP835(BaseDNSMOS):
    def forward(self, preds: torch.Tensor) -> tuple:
        scores = self.compute_scores(preds)

        return (
            self.reduce(scores[..., 1]),
            self.reduce(scores[..., 2]),
            self.reduce(scores[..., 3]),
        )
    

########################################
# SIG
########################################


class DNSMOSSIG(BaseDNSMOS):
    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        scores = self.compute_scores(preds)
        return self.reduce(scores[..., 1])
    

##########################################
# BAK
##########################################

class DNSMOSBAK(BaseDNSMOS):
    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        scores = self.compute_scores(preds)
        return self.reduce(scores[..., 2])
    

############################################
# OVRL
############################################

class DNSMOSOVRL(BaseDNSMOS):
    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        scores = self.compute_scores(preds)
        return self.reduce(scores[..., 3])