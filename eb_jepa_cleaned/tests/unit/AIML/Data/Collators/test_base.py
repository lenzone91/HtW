"""
Unit tests for AIML.Data.Collators.base.BaseCollator.
"""

import pytest

from src.AIML.Data.BatchTransform.base import (
    BaseBatchTransform,
)
from src.AIML.Data.Collators.base import (
    BaseCollator,
)


class ListCollator(BaseCollator):
    def collate_samples(self, samples):
        return {"input": [s["input"] for s in samples]}


class AddFlag(BaseBatchTransform):
    def transform(self, batch):
        return {**batch, "flag": True}


def test_collate_without_transforms():
    collator = ListCollator()

    batch = collator([{"input": 1}, {"input": 2}])

    assert batch == {"input": [1, 2]}


def test_batch_transforms_applied_in_order():
    collator = ListCollator(batch_transforms=[AddFlag()])

    batch = collator([{"input": 1}])

    assert batch == {"input": [1], "flag": True}


def test_dict_transforms_normalized_to_list():
    collator = ListCollator(batch_transforms={"add": AddFlag()})

    assert isinstance(collator.batch_transforms, list)
    assert collator([{"input": 1}])["flag"] is True


def test_rejects_non_list_samples():
    with pytest.raises(TypeError, match="samples to be a list"):
        ListCollator()({"input": 1})


def test_rejects_empty_samples():
    with pytest.raises(ValueError, match="empty sample list"):
        ListCollator()([])


def test_rejects_non_dict_sample():
    with pytest.raises(TypeError, match="each sample to be a dict"):
        ListCollator()([{"input": 1}, "bad"])
