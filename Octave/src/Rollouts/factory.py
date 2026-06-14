from . import latent_rollout as _latent_rollout  # noqa: F401
from .configs import DEFAULT_LATENT_ROLLOUT_CONFIG
from .latent_rollout import LatentRollout
from .registry import ROLLOUT_REGISTRY
from ..Workflow.Factory.builder import RegistryBuilder


def make_rollout_builder(strict: bool = True) -> RegistryBuilder:
    return RegistryBuilder(
        registry=ROLLOUT_REGISTRY,
        strict=strict,
        type_field="rollout_type",
    )


def build_rollout(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    builder = make_rollout_builder(strict=strict)
    return builder.build_one(
        config=config or DEFAULT_LATENT_ROLLOUT_CONFIG,
        runtime_context=runtime_context,
    )


def build_latent_rollout(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> LatentRollout:
    return build_rollout(
        config=config or DEFAULT_LATENT_ROLLOUT_CONFIG,
        runtime_context=runtime_context,
        strict=strict,
    )
