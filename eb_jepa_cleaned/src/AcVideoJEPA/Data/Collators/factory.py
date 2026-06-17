"""
AcVideoJEPA collators factory. Importing this registers the ac_video_jepa
collator onto the AIML collator registry.
"""

from . import ac_video_jepa_collator  # noqa: F401  (registers `ac_video_jepa`)

REGISTERED_COLLATORS = ("ac_video_jepa",)
