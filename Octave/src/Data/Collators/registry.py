from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import FieldResolution, Registry


def resolve_collator_transforms(
    config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
    **kwargs,
) -> list | None:
    transforms = config.get("transforms", None)

    if transforms is None:
        return None

    if not isinstance(transforms, list):
        raise TypeError(
            "Collator config 'transforms' must be a list, "
            f"got {type(transforms).__name__}."
        )

    for transform_index, transform in enumerate(transforms):
        if not callable(transform):
            raise TypeError(
                "Collator transforms must be callable, "
                f"but transform {transform_index} has type "
                f"{type(transform).__name__}."
            )

    return transforms


COLLATOR_REGISTRY = Registry(object_family="collator")

COLLATOR_BUILDER = RegistryBuilder(
    registry=COLLATOR_REGISTRY,
    type_field="collator_type",
)