"""Tests for bounded retry behavior in dataset __getitem__ methods."""

from unittest.mock import MagicMock, patch

import pytest

from boltz.data.module.training import _MAX_GETITEM_RETRIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_dataset():
    """Build a TrainingDataset with a mocked single dataset."""
    from boltz.data.module.training import TrainingDataset

    record = MagicMock()
    record.id = "test_record"

    sample = MagicMock()
    sample.record = record
    sample.chain_id = None
    sample.interface_id = None

    dataset = MagicMock()
    dataset.prob = 1.0
    dataset.manifest.records = [record]
    dataset.sampler.sample.return_value = iter(
        [sample] * (_MAX_GETITEM_RETRIES + 5)
    )

    ds = TrainingDataset(
        datasets=[dataset],
        samples_per_epoch=10,
        symmetries={},
        max_atoms=1000,
        max_tokens=None,  # disable cropping
        max_seqs=10,
    )
    return ds


def _make_validation_dataset():
    """Build a ValidationDataset with a mocked single dataset."""
    from boltz.data.module.training import ValidationDataset

    record = MagicMock()
    record.id = "test_record"

    dataset = MagicMock()
    dataset.manifest.records = [record]

    ds = ValidationDataset(
        datasets=[dataset],
        seed=42,
        symmetries={},
        max_atoms=1000,
        max_tokens=None,  # disable cropping
        max_seqs=10,
    )
    return ds


def _make_training_dataset_v2():
    """Build a trainingv2.TrainingDataset with a mocked single dataset."""
    from boltz.data.module.trainingv2 import TrainingDataset

    record = MagicMock()
    record.id = "test_record_v2"

    sample = MagicMock()
    sample.record = record
    sample.chain_id = None
    sample.interface_id = None

    dataset = MagicMock()
    dataset.prob = 1.0
    dataset.manifest.records = [record]
    dataset.sampler.sample.return_value = iter(
        [sample] * (_MAX_GETITEM_RETRIES + 5)
    )

    ds = TrainingDataset(
        datasets=[dataset],
        samples_per_epoch=10,
        symmetries={},
        max_atoms=1000,
        max_tokens=None,
        max_seqs=10,
    )
    return ds


def _make_validation_dataset_v2():
    """Build a trainingv2.ValidationDataset with a mocked single dataset."""
    from boltz.data.module.trainingv2 import ValidationDataset

    record = MagicMock()
    record.id = "test_record_v2"

    dataset = MagicMock()
    dataset.manifest.records = [record]

    ds = ValidationDataset(
        datasets=[dataset],
        symmetries={},
        max_atoms=1000,
        max_tokens=None,
        max_seqs=10,
    )
    return ds


# ---------------------------------------------------------------------------
# training.py — TrainingDataset
# ---------------------------------------------------------------------------

class TestTrainingDatasetRetry:
    """Training __getitem__ uses a retry loop capped at _MAX_GETITEM_RETRIES."""

    @patch("boltz.data.module.training.load_input")
    def test_all_retries_exhausted_raises(self, mock_load):
        """If load_input always fails, RuntimeError after max retries."""
        mock_load.side_effect = RuntimeError("corrupt file")
        ds = _make_training_dataset()

        with pytest.raises(RuntimeError, match="Failed to load any sample"):
            ds[0]

        assert mock_load.call_count == _MAX_GETITEM_RETRIES

    @patch("boltz.data.module.training.load_input")
    def test_succeeds_after_transient_failures(self, mock_load):
        """Retry loop recovers when load_input succeeds on a later attempt."""
        fake_features = {"coords": "ok"}
        mock_load.side_effect = [
            RuntimeError("err"),
            RuntimeError("err"),
            MagicMock(),
        ]

        ds = _make_training_dataset()
        dataset = ds.datasets[0]

        mock_tokenized = MagicMock()
        mock_tokenized.tokens = [1, 2, 3]  # non-empty
        dataset.tokenizer.tokenize.return_value = mock_tokenized
        dataset.featurizer.process.return_value = fake_features

        result = ds[0]
        assert result == fake_features
        assert mock_load.call_count == 3

    @patch("boltz.data.module.training.load_input")
    def test_empty_tokens_retries(self, mock_load):
        """Empty tokens after tokenize triggers continue, not crash."""
        mock_load.return_value = MagicMock()
        ds = _make_training_dataset()
        dataset = ds.datasets[0]

        mock_tokenized = MagicMock()
        mock_tokenized.tokens = []
        dataset.tokenizer.tokenize.return_value = mock_tokenized

        with pytest.raises(RuntimeError, match="Failed to load any sample"):
            ds[0]

        assert mock_load.call_count == _MAX_GETITEM_RETRIES


# ---------------------------------------------------------------------------
# training.py — ValidationDataset
# ---------------------------------------------------------------------------

class TestValidationDatasetRetry:
    """Validation __getitem__ passes _retries through recursive calls."""

    @patch("boltz.data.module.training.load_input")
    def test_all_retries_exhausted_raises(self, mock_load):
        """If load_input always fails, RuntimeError after max retries."""
        mock_load.side_effect = RuntimeError("corrupt file")
        ds = _make_validation_dataset()

        with pytest.raises(RuntimeError, match="Failed to load any sample"):
            ds[0]

        assert mock_load.call_count == _MAX_GETITEM_RETRIES

    @patch("boltz.data.module.training.load_input")
    def test_empty_tokens_retries(self, mock_load):
        """Empty tokens triggers retry, not ValueError."""
        mock_load.return_value = MagicMock()
        ds = _make_validation_dataset()
        dataset = ds.datasets[0]

        mock_tokenized = MagicMock()
        mock_tokenized.tokens = []
        dataset.tokenizer.tokenize.return_value = mock_tokenized

        with pytest.raises(RuntimeError, match="Failed to load any sample"):
            ds[0]


# ---------------------------------------------------------------------------
# trainingv2.py — TrainingDataset
# ---------------------------------------------------------------------------

class TestTrainingDatasetV2Retry:
    """trainingv2 TrainingDataset retry loop mirrors training.py."""

    @patch("boltz.data.module.trainingv2.load_input")
    def test_all_retries_exhausted_raises(self, mock_load):
        mock_load.side_effect = RuntimeError("corrupt file")
        ds = _make_training_dataset_v2()

        with pytest.raises(RuntimeError, match="Failed to load any sample"):
            ds[0]

        assert mock_load.call_count == _MAX_GETITEM_RETRIES

    @patch("boltz.data.module.trainingv2.load_input")
    def test_succeeds_after_transient_failures(self, mock_load):
        """Retry loop recovers when load_input succeeds on a later attempt."""
        fake_features = {"coords": "ok"}
        mock_load.side_effect = [
            RuntimeError("err"),
            RuntimeError("err"),
            MagicMock(),
        ]

        ds = _make_training_dataset_v2()
        dataset = ds.datasets[0]

        mock_tokenized = MagicMock()
        mock_tokenized.tokens = [1, 2, 3]
        dataset.tokenizer.tokenize.return_value = mock_tokenized
        dataset.featurizer.process.return_value = fake_features

        result = ds[0]
        assert result == fake_features
        assert mock_load.call_count == 3

    @patch("boltz.data.module.trainingv2.load_input")
    def test_empty_tokens_retries(self, mock_load):
        mock_load.return_value = MagicMock()
        ds = _make_training_dataset_v2()
        dataset = ds.datasets[0]

        mock_tokenized = MagicMock()
        mock_tokenized.tokens = []
        dataset.tokenizer.tokenize.return_value = mock_tokenized

        with pytest.raises(RuntimeError, match="Failed to load any sample"):
            ds[0]

        assert mock_load.call_count == _MAX_GETITEM_RETRIES


# ---------------------------------------------------------------------------
# trainingv2.py — ValidationDataset
# ---------------------------------------------------------------------------

class TestValidationDatasetV2Retry:
    """trainingv2 ValidationDataset retry mirrors training.py."""

    @patch("boltz.data.module.trainingv2.load_input")
    def test_all_retries_exhausted_raises(self, mock_load):
        mock_load.side_effect = RuntimeError("corrupt file")
        ds = _make_validation_dataset_v2()

        with pytest.raises(RuntimeError, match="Failed to load any sample"):
            ds[0]

        assert mock_load.call_count == _MAX_GETITEM_RETRIES

    @patch("boltz.data.module.trainingv2.load_input")
    def test_empty_tokens_retries(self, mock_load):
        mock_load.return_value = MagicMock()
        ds = _make_validation_dataset_v2()
        dataset = ds.datasets[0]

        mock_tokenized = MagicMock()
        mock_tokenized.tokens = []
        dataset.tokenizer.tokenize.return_value = mock_tokenized

        with pytest.raises(RuntimeError, match="Failed to load any sample"):
            ds[0]
