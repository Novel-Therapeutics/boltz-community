"""Tests for the boltz CLI: option defaults, CPU affinity, and download logic."""

import os
from unittest.mock import patch

import pytest


class TestSubsampleMsaDefault:
    """The --subsample_msa CLI option must default to True."""

    @pytest.fixture(autouse=True)
    def _import_predict(self):
        """Import predict command, skipping if heavy deps are missing."""
        try:
            from boltz.main import predict
            self.predict = predict
        except ImportError as e:
            pytest.skip(f"Cannot import boltz.main: {e}")

    def test_default_is_true(self):
        """predict command's subsample_msa param defaults to True."""
        for param in self.predict.params:
            if param.name == "subsample_msa":
                assert param.default is True, (
                    f"subsample_msa default should be True, got {param.default}"
                )
                return
        pytest.fail("subsample_msa parameter not found on predict command")

    def test_is_boolean_flag_pair(self):
        """subsample_msa should support --subsample_msa / --no_subsample_msa."""
        for param in self.predict.params:
            if param.name == "subsample_msa":
                assert param.is_flag, "subsample_msa should be a flag"
                # Click 8.x uses secondary_opts (list); Click 7.x uses secondary (bool)
                secondary = getattr(param, "secondary_opts", None) or getattr(param, "secondary", None)
                assert secondary, (
                    "subsample_msa should be a boolean flag pair "
                    "(--subsample_msa/--no_subsample_msa)"
                )
                return
        pytest.fail("subsample_msa parameter not found on predict command")


class TestAvailableCpuCount:
    """_available_cpu_count must respect cgroup/taskset limits."""

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        """Import the helper, skipping if heavy deps are missing."""
        try:
            from boltz.main import _available_cpu_count
            self._available_cpu_count = _available_cpu_count
        except ImportError as e:
            pytest.skip(f"Cannot import boltz.main: {e}")

    def test_returns_positive_int(self):
        """_available_cpu_count returns a positive integer."""
        result = self._available_cpu_count()
        assert isinstance(result, int)
        assert result >= 1

    def test_fallback_on_oserror(self):
        """Falls back to os.cpu_count when sched_getaffinity raises OSError."""
        if hasattr(os, "sched_getaffinity"):
            with patch("os.sched_getaffinity", side_effect=OSError("mocked")):
                result = self._available_cpu_count()
        else:
            # On macOS, the fallback path is the default path
            result = self._available_cpu_count()
        assert isinstance(result, int)
        assert result >= 1

    def test_fallback_on_missing_attr(self):
        """Falls back when sched_getaffinity is absent (macOS)."""
        if hasattr(os, "sched_getaffinity"):
            with patch("os.sched_getaffinity", side_effect=AttributeError):
                result = self._available_cpu_count()
        else:
            # Already exercising the fallback path on macOS
            result = self._available_cpu_count()
        assert isinstance(result, int)
        assert result >= 1


class TestDownloadBoltz2:
    """download_boltz2 must skip already-cached files."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if boltz.main can't be imported."""
        try:
            from boltz.main import download_boltz2  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import boltz.main: {e}")

    @staticmethod
    def _prefill_all_downloads(tmp_path):
        """Pre-create all downloadable files so no network calls are needed."""
        (tmp_path / "mols").mkdir(exist_ok=True)
        (tmp_path / "boltz2_conf.ckpt").write_text("fake")
        (tmp_path / "boltz2_aff.ckpt").write_text("fake")

    def test_no_tar_download_when_mols_exists(self, tmp_path):
        """When mols/ exists, download_boltz2 must not download mols.tar."""
        from boltz.main import download_boltz2

        self._prefill_all_downloads(tmp_path)

        with patch("urllib.request.urlretrieve") as mock_retrieve:
            download_boltz2(tmp_path)
            mock_retrieve.assert_not_called()

    def test_tar_downloaded_when_mols_missing(self, tmp_path):
        """When mols/ is missing, download_boltz2 must download and extract."""
        from boltz.main import download_boltz2

        (tmp_path / "boltz2_conf.ckpt").write_text("fake")
        (tmp_path / "boltz2_aff.ckpt").write_text("fake")

        with patch("urllib.request.urlretrieve") as mock_retrieve, \
             patch("tarfile.open") as mock_tarfile:
            mock_tar = mock_tarfile.return_value.__enter__.return_value

            def fake_extractall(path):
                (path / "mols").mkdir(exist_ok=True)
            mock_tar.extractall.side_effect = fake_extractall

            download_boltz2(tmp_path)
            mock_retrieve.assert_called_once()
