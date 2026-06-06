import torch


class ReprTargetDistMPCObjective:
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        target_enc: torch.Tensor,
        sum_all_diffs: bool = False,
        **kwargs,
    ):
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        """
        Args:
            encodings: [B, D, T, H, W]
            keepdims: if True, return [B, T], else return [B]

        Returns:
            diff: [B, T] else [B] if sum_all_diffs or not keepdims
        """
        if self.sum_all_diffs:
            keepdims = True
        target = self.target_enc
        if target.shape != encodings.shape:
            target = target.expand(encodings.shape[0], -1, encodings.shape[2], -1, -1)

        metric = torch.nn.MSELoss(reduction="none")
        diff = metric(target, encodings).mean(dim=(1, 3, 4))  # B T
        if not keepdims:
            diff = diff[:, -1]
        if self.sum_all_diffs:
            diff = diff.sum(dim=1)
        return diff
