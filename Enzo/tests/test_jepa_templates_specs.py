import pytest

from Enzo.jepa_templates.specs import JEPABatch, validate_jepa_batch


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


def test_validate_jepa_batch_accepts_world_model_shapes():
    batch = JEPABatch(
        obs=FakeTensor((2, 3, 4, 16, 16)),
        actions=FakeTensor((2, 2, 3)),
    )

    validate_jepa_batch(batch)


def test_validate_jepa_batch_rejects_non_sequence_obs():
    batch = JEPABatch(obs=FakeTensor((2, 3, 16, 16)))

    with pytest.raises(ValueError):
        validate_jepa_batch(batch)


def test_validate_jepa_batch_rejects_action_batch_mismatch():
    batch = JEPABatch(
        obs=FakeTensor((2, 3, 4, 16, 16)),
        actions=FakeTensor((3, 2, 4)),
    )

    with pytest.raises(ValueError):
        validate_jepa_batch(batch)

