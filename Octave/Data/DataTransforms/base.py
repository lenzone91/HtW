from abc import ABC, abstractmethod


##################################################################
##################################################################
##################################################################
# Base transsform parent class
##################################################################
##################################################################
##################################################################


class BaseBatchTransform(ABC):
    """
    Base class for batch-level transforms.

    A transform maps:

        batch: dict -> batch: dict
    """

    ###################################################################################
    # Usage logic
    ###################################################################################

    def __call__(self, batch: dict) -> dict:
        self.check_input_batch(batch)

        transformed_batch = self.transform(batch)

        self.check_output_batch(transformed_batch)

        return transformed_batch
    
    
    @abstractmethod
    def transform(self, batch: dict) -> dict:
        raise NotImplementedError
    
    #################################################################################
    # Tests logic
    #################################################################################

    def check_input_batch(self, batch: dict) -> None:
        self.check_is_dict(batch, name="input batch")

    def check_output_batch(self, batch: dict) -> None:
        self.check_is_dict(batch, name="output batch")

    def check_is_dict(self, value: object, name: str) -> None:
        if not isinstance(value, dict):
            raise TypeError(
                f"{self.__class__.__name__} expected {name} to be a dict, "
                f"got {type(value).__name__}."
            )

    def check_required_key(self, batch: dict, key: str) -> None:
        if key not in batch:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{key}' in batch."
            )

    def check_is_tensor(self, value: object, name: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{self.__class__.__name__} expected {name} to be a torch.Tensor, "
                f"got {type(value).__name__}."
            )

    def check_is_batched_tensor(
        self,
        value: object,
        name: str,
        min_ndim: int = 2,
    ) -> None:
        self.check_is_tensor(value, name=name)

        if value.ndim < min_ndim:
            raise ValueError(
                f"{self.__class__.__name__} expected {name} to have at least "
                f"{min_ndim} dimensions, got shape {tuple(value.shape)}."
            )
        

    
