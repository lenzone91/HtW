from copy import deepcopy

from .ac_video_jepa_collator import AcVideoJepaCollator
from .configs import DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG


COLLATOR_BUILDERS_REGISTRY = {
    "ac_video_jepa": AcVideoJepaCollator,
}


class CollatorBuilder:
    """
    Build one collator from a plain dictionary config.
    """

    def __init__(
        self,
        default_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        self.default_config = deepcopy(
            default_config
            if default_config is not None
            else DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG
        )
        self.strict = strict

    def __call__(self, config: dict) -> AcVideoJepaCollator:
        prepared_config = self.prepare_config(config)
        collator_type = prepared_config.pop("collator_type")
        transforms = prepared_config.pop("transforms")

        collator_class = COLLATOR_BUILDERS_REGISTRY[collator_type]
        return collator_class(transforms=transforms)

    def prepare_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(
                "Collator config must be a dictionary, "
                f"got {type(config).__name__}."
            )

        prepared_config = deepcopy(self.default_config)
        user_config = deepcopy(config)

        self.check_known_keys(user_config)
        prepared_config.update(user_config)
        self.check_collator_type(prepared_config)
        self.check_transforms(prepared_config)

        return prepared_config

    def check_known_keys(self, config: dict) -> None:
        unknown_keys = set(config) - set(self.default_config)

        if not unknown_keys:
            return

        message = (
            "Unknown collator config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(self.default_config)}."
        )

        if self.strict:
            raise KeyError(message)

    def check_collator_type(self, config: dict) -> None:
        collator_type = config["collator_type"]

        if collator_type not in COLLATOR_BUILDERS_REGISTRY:
            raise KeyError(
                f"Unknown collator_type '{collator_type}'. "
                f"Available collators are: {sorted(COLLATOR_BUILDERS_REGISTRY)}."
            )

    def check_transforms(self, config: dict) -> None:
        transforms = config["transforms"]

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


def build_ac_video_jepa_collator(
    config: dict | None = None,
) -> AcVideoJepaCollator:
    builder = CollatorBuilder()
    return builder(config=config or DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG)


def build_collator(config: dict) -> AcVideoJepaCollator:
    builder = CollatorBuilder()
    return builder(config=config)
