"""
IMPALA-style convolutional encoder.

Re-implemented from EB-JEPA (the project does not depend on `eb_jepa`,
Decision 30). Encodes a video clip frame-by-frame into a sequence of latent
vectors.

Shape contract:
    input  [B, C, T, H, W]
    output [B, D, T, 1, 1]   (D = mlp_output_dim; spatial dims collapsed to 1x1)
"""

import torch
import torch.nn.functional as F
from torch import nn

from ....AIML.Models.Models.registry import MODEL_REGISTRY


class ResnetBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    """Initial conv + optional max-pool + a stack of residual blocks."""

    def __init__(
        self,
        input_channels: int,
        num_features: int,
        num_blocks: int,
        max_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )
        self.max_pool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if max_pooling
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


DEFAULT_IMPALA_ENCODER_CONFIG = {
    "width": 1,
    "stack_sizes": [16, 32, 32],
    "num_blocks": 2,
    "dropout_rate": None,
    "layer_norm": False,
    "input_channels": 2,
    "final_ln": True,
    "mlp_output_dim": 512,
    "input_shape": [2, 65, 65],
}


@MODEL_REGISTRY.register_class(
    name="impala_encoder",
    default_config=DEFAULT_IMPALA_ENCODER_CONFIG,
)
class ImpalaEncoder(nn.Module):
    """
    IMPALA encoder applied independently per time step.

    The MLP input dimension is inferred from `input_shape` by a dummy forward
    pass at construction. `final_ln` is exposed as an attribute so a predictor
    can optionally reuse the encoder's final LayerNorm.
    """

    def __init__(
        self,
        width: int = 1,
        stack_sizes=(16, 32, 32),
        num_blocks: int = 2,
        dropout_rate: float | None = None,
        layer_norm: bool = False,
        input_channels: int = 2,
        final_ln: bool = True,
        mlp_output_dim: int = 512,
        input_shape=(2, 65, 65),
    ) -> None:
        super().__init__()
        self.width = width
        self.stack_sizes = tuple(stack_sizes)
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.mlp_output_dim = mlp_output_dim
        self.input_shape = tuple(input_shape)

        # Each stack's input channels are the previous stack's output channels
        # (stack i outputs stack_sizes[i] * width). Identical to EB-JEPA at the
        # default width=1; correct for width != 1 (EB-JEPA mis-wired that case).
        stack_input_channels = [input_channels] + [s * width for s in self.stack_sizes]
        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=stack_input_channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(self.stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        flattened_dim = self._infer_flattened_dim()
        self.mlp = nn.Linear(flattened_dim, mlp_output_dim)
        self.final_ln = nn.LayerNorm(mlp_output_dim) if final_ln else nn.Identity()

    def _infer_flattened_dim(self) -> int:
        with torch.no_grad():
            conv_out = torch.zeros(1, *self.input_shape)
            for stack_block in self.stack_blocks:
                conv_out = stack_block(conv_out)
            return conv_out.view(conv_out.size(0), -1).shape[1]

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single frame `[B, C, H, W]` into `[B, D]`."""
        conv_out = frame
        for stack_block in self.stack_blocks:
            conv_out = stack_block(conv_out)
            conv_out = self.dropout(conv_out)
        conv_out = F.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm(conv_out.size()[1:], device=conv_out.device)(
                conv_out
            )
        flat = conv_out.view(conv_out.size(0), -1)
        return self.final_ln(self.mlp(flat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T, H, W] -> iterate over T, encode each frame.
        time_steps = x.shape[2]
        frames = x.permute(2, 0, 1, 3, 4)  # [T, B, C, H, W]
        features = [self.encode_frame(frames[t]) for t in range(time_steps)]
        features = torch.stack(features, dim=1)  # [B, T, D]
        # -> [B, D, T, 1, 1]
        return features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
