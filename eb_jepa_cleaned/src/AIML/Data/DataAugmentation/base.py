from ..BatchTransform.base import BaseBatchTransform


#############################################
# Base augmentation
#############################################


class BaseAugmentation(BaseBatchTransform):
    """
    Base class for data augmentations: stochastic perturbations of a batch
    (e.g. adding noise at a controlled SNR).

    Concrete augmentations subclass this and implement `transform`. It is a
    semantic family marker over `BaseBatchTransform`; shared augmentation logic
    (e.g. seeding / RNG handling) can be added here when needed. It defines no
    default config and is not registered.
    """
