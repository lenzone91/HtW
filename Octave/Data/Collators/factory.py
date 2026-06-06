from .ac_video_jepa_collator import AcVideoJepa_collator


def build_collator(config: dict = {}) -> AcVideoJepa_collator:
    """
    Build a collator from a config dictionary.

    Expected config:
        {
            "transforms": list | None
        }
    """
    return AcVideoJepa_collator(
        transforms=config.get("transforms", None),
    )