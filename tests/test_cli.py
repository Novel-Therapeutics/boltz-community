"""Tests for the boltz CLI: option defaults, CPU affinity, and download logic."""

import os
from pathlib import Path
from unittest.mock import patch

import click
import pytest

from boltz.main import _available_cpu_count, _validate_checkpoint_for_load, predict


class TestSubsampleMsaDefault:
    """The --subsample_msa CLI option must default to True."""

    def test_default_is_true(self):
        """predict command's subsample_msa param defaults to True."""
        for param in predict.params:
            if param.name == "subsample_msa":
                assert param.default is True, (
                    f"subsample_msa default should be True, got {param.default}"
                )
                return
        pytest.fail("subsample_msa parameter not found on predict command")

    def test_is_boolean_flag_pair(self):
        """subsample_msa should support --subsample_msa / --no_subsample_msa."""
        for param in predict.params:
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

    def test_returns_positive_int(self):
        """_available_cpu_count returns a positive integer."""
        result = _available_cpu_count()
        assert isinstance(result, int)
        assert result >= 1

    def test_fallback_on_oserror(self):
        """Falls back to os.cpu_count when sched_getaffinity raises OSError."""
        if hasattr(os, "sched_getaffinity"):
            with patch("os.sched_getaffinity", side_effect=OSError("mocked")):
                result = _available_cpu_count()
        else:
            # On macOS, the fallback path is the default path
            result = _available_cpu_count()
        assert isinstance(result, int)
        assert result >= 1

    def test_fallback_on_missing_attr(self):
        """Falls back when sched_getaffinity is absent (macOS)."""
        if hasattr(os, "sched_getaffinity"):
            with patch("os.sched_getaffinity", side_effect=AttributeError):
                result = _available_cpu_count()
        else:
            # Already exercising the fallback path on macOS
            result = _available_cpu_count()
        assert isinstance(result, int)
        assert result >= 1


class TestDownloadBoltz2:
    """download_boltz2 must skip already-cached files."""

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

        with patch("boltz.main._download") as mock_download:
            download_boltz2(tmp_path)
            mock_download.assert_not_called()

    def test_tar_downloaded_when_mols_missing(self, tmp_path):
        """When mols/ is missing, download_boltz2 must download and extract."""
        from boltz.main import download_boltz2

        (tmp_path / "boltz2_conf.ckpt").write_text("fake")
        (tmp_path / "boltz2_aff.ckpt").write_text("fake")

        with patch("boltz.main._download") as mock_download, \
             patch("tarfile.open") as mock_tarfile:
            mock_tar = mock_tarfile.return_value.__enter__.return_value

            def fake_extractall(path):
                (path / "mols").mkdir(exist_ok=True)
            mock_tar.extractall.side_effect = fake_extractall

            download_boltz2(tmp_path)
            mock_download.assert_called_once()


class TestCheckpointValidation:
    """Checkpoint validation should fail early with clear errors."""

    def test_rejects_empty_checkpoint_before_load(self, tmp_path):
        checkpoint = tmp_path / "boltz2_conf.ckpt"
        checkpoint.write_bytes(b"")

        with pytest.raises(click.ClickException, match="checkpoint is empty"):
            _validate_checkpoint_for_load(checkpoint, "Boltz-2 weights")

    def test_accepts_non_empty_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "boltz2_conf.ckpt"
        checkpoint.write_bytes(b"not-empty")

        _validate_checkpoint_for_load(checkpoint, "Boltz-2 weights")


class TestDownloadBoltz1CheckpointValidation:
    """download_boltz1 should recover from empty cached/downloaded checkpoints."""

    def test_redownloads_existing_empty_checkpoint(self, tmp_path):
        from boltz.main import download_boltz1

        (tmp_path / "ccd.pkl").write_text("fake-ccd")
        checkpoint = tmp_path / "boltz1_conf.ckpt"
        checkpoint.write_bytes(b"")

        def fake_download(_url, dest):
            Path(dest).write_bytes(b"fresh-weights")

        with patch("boltz.main._download", side_effect=fake_download) as mock_download:
            download_boltz1(tmp_path)

        assert mock_download.call_count == 1
        assert checkpoint.read_bytes() == b"fresh-weights"

    def test_retries_when_first_download_is_empty(self, tmp_path):
        from boltz.main import download_boltz1

        (tmp_path / "ccd.pkl").write_text("fake-ccd")
        checkpoint = tmp_path / "boltz1_conf.ckpt"
        call_count = 0

        def fake_download(_url, dest):
            nonlocal call_count
            call_count += 1
            Path(dest).write_bytes(b"" if call_count == 1 else b"fresh-weights")

        with patch("boltz.main._download", side_effect=fake_download):
            download_boltz1(tmp_path)

        assert call_count == 2
        assert checkpoint.read_bytes() == b"fresh-weights"

    def test_raises_clear_error_when_all_downloads_are_empty(self, tmp_path):
        from boltz.main import download_boltz1

        (tmp_path / "ccd.pkl").write_text("fake-ccd")

        def fake_download(_url, dest):
            Path(dest).write_bytes(b"")

        with patch("boltz.main._download", side_effect=fake_download):
            with pytest.raises(RuntimeError, match="checkpoint is empty"):
                download_boltz1(tmp_path)


class TestCheckInputs:
    """check_inputs behaviour with and without --skip_bad_inputs."""

    def test_single_file_returned_as_list(self, tmp_path):
        from boltz.main import check_inputs

        f = tmp_path / "input.yaml"
        f.write_text("dummy")
        assert check_inputs(f) == [f]

    def test_directory_filters_valid_extensions(self, tmp_path):
        from boltz.main import check_inputs

        (tmp_path / "a.yaml").write_text("y")
        (tmp_path / "b.fasta").write_text("f")
        names = sorted(p.name for p in check_inputs(tmp_path))
        assert names == ["a.yaml", "b.fasta"]

    def test_directory_aborts_on_bad_extension_by_default(self, tmp_path):
        from boltz.main import check_inputs

        (tmp_path / "good.yaml").write_text("y")
        (tmp_path / "bad.txt").write_text("t")
        with pytest.raises(RuntimeError, match="Unable to parse filetype"):
            check_inputs(tmp_path)

    def test_directory_aborts_on_subdirectory_by_default(self, tmp_path):
        from boltz.main import check_inputs

        (tmp_path / "good.yaml").write_text("y")
        (tmp_path / "subdir").mkdir()
        with pytest.raises(RuntimeError, match="Found directory"):
            check_inputs(tmp_path)

    def test_skip_bad_inputs_filters_bad_extension(self, tmp_path):
        from boltz.main import check_inputs

        (tmp_path / "good.yaml").write_text("y")
        (tmp_path / "bad.txt").write_text("t")
        result = check_inputs(tmp_path, skip_bad_inputs=True)
        assert [p.name for p in result] == ["good.yaml"]

    def test_skip_bad_inputs_filters_subdirectory(self, tmp_path):
        from boltz.main import check_inputs

        (tmp_path / "good.fasta").write_text("f")
        (tmp_path / "subdir").mkdir()
        result = check_inputs(tmp_path, skip_bad_inputs=True)
        assert [p.name for p in result] == ["good.fasta"]


class TestProcessInputSkipBadInputs:
    """process_input raises by default, returns error string with skip_bad_inputs."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from boltz.main import process_input
            self.process_input = process_input
        except ImportError as e:
            pytest.skip(f"Cannot import boltz.main: {e}")

    def _call(self, path, skip_bad_inputs=False):
        """Call process_input with dummy args — only the parse dispatch matters."""
        return self.process_input(
            path=path,
            ccd={},
            msa_dir=Path("/unused"),
            mol_dir=Path("/unused"),
            boltz2=False,
            use_msa_server=False,
            msa_server_url="",
            msa_pairing_strategy="greedy",
            msa_server_username=None,
            msa_server_password=None,
            api_key_header=None,
            api_key_value=None,
            max_msa_seqs=1,
            processed_msa_dir=Path("/unused"),
            processed_constraints_dir=Path("/unused"),
            processed_templates_dir=Path("/unused"),
            processed_mols_dir=Path("/unused"),
            structure_dir=Path("/unused"),
            records_dir=Path("/unused"),
            skip_bad_inputs=skip_bad_inputs,
        )

    def test_bad_input_raises_by_default(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("not valid yaml: [")
        with pytest.raises(Exception):
            self._call(bad)

    def test_bad_input_returns_error_with_skip(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("not valid yaml: [")
        result = self._call(bad, skip_bad_inputs=True)
        assert result is not None
        assert "bad.yaml" in result

    def test_unsupported_extension_raises_by_default(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("hello")
        with pytest.raises(RuntimeError, match="Unable to parse filetype"):
            self._call(bad)

    def test_unsupported_extension_returns_error_with_skip(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("hello")
        result = self._call(bad, skip_bad_inputs=True)
        assert result is not None
        assert "data.txt" in result


class TestSkipBadInputsClickOption:
    """The --skip_bad_inputs CLI option exists and defaults to False."""

    def test_option_exists_and_defaults_false(self):
        for param in predict.params:
            if param.name == "skip_bad_inputs":
                assert param.default is False
                assert param.is_flag
                return
        pytest.fail("skip_bad_inputs parameter not found on predict command")
