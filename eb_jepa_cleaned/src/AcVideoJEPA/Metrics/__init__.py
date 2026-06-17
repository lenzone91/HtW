"""
AcVideoJEPA JEPA objective: the latent prediction loss and the anti-collapse /
auxiliary regularizers, as metrics over a `LatentRolloutOutput`, built on AIML's
metric machinery. Importing this subpackage registers the metrics onto the AIML
metric registry.
"""

from . import factory  # noqa: F401  (registration side effect)
