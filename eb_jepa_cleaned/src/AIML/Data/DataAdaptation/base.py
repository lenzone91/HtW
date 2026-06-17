from ..BatchTransform.base import BaseBatchTransform


#############################################
# Base adaptation
#############################################


class BaseAdaptation(BaseBatchTransform):
    """
    Base class for data adaptations: mapping a dataset's output representation to
    a model's input representation (e.g. waveform -> spectrogram). The
    dataset->model "interface".

    Concrete adaptations subclass this and implement `transform`. It is a
    semantic family marker over `BaseBatchTransform`; the representation-contract
    helpers (declaring/propagating source and target representations) can be
    added here when the concrete audio adaptations land. It defines no default
    config and is not registered.
    """
