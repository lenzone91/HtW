from .base import BaseDataModule

class DefaultDataModule(BaseDataModule):
    """
    Default project DataModule.

    It adds no behavior over BaseDataModule.
    It exists as the standard concrete DataModule exposed to the factory.
    """

    pass