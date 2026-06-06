from .base import BaseCollator
from ..DataTransforms.base import BaseBatchTransform

"""
No specific behavior yet
"""

class AcVideoJepa_collator(BaseCollator):
    def __init__(self, transforms: list[BaseBatchTransform] | None = None) -> None:
        super().__init__(transforms)
