"""Registry-based factory primitives."""

from .builder import RegistryBuilder
from .registry import FieldResolution, Registry, RegistryEntry, SubBuildDeclaration

__all__ = [
    "FieldResolution",
    "Registry",
    "RegistryBuilder",
    "RegistryEntry",
    "SubBuildDeclaration",
]
