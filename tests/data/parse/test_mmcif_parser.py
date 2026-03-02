"""Tests for boltz.data.parse.mmcif and mmcif_with_constraints — CIF parsing."""

import pytest


class TestEmptyCifParsing:
    """Both mmCIF parsers must reject CIF files that contain no coordinate models."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if gemmi or parser deps are missing."""
        try:
            import gemmi  # noqa: F401
            from boltz.data.parse.mmcif import parse_mmcif  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import parser deps: {e}")

    def _empty_cif(self, tmp_path, name="empty.cif"):
        """Write a minimal CIF with no coordinate models."""
        cif_content = """\
data_EMPTY
_entry.id EMPTY
_entity.id 1
_entity.type polymer
_entity.pdbx_description 'Empty test'
"""
        cif_path = tmp_path / name
        cif_path.write_text(cif_content)
        return cif_path

    def test_parse_mmcif_raises_valueerror(self, tmp_path):
        """CIF with no coordinate models raises ValueError, not IndexError."""
        from boltz.data.parse.mmcif import parse_mmcif

        cif_path = self._empty_cif(tmp_path)

        with pytest.raises(ValueError, match="no models"):
            parse_mmcif(str(cif_path))

    def test_parse_mmcif_with_constraints_raises_valueerror(self, tmp_path):
        """mmcif_with_constraints parser also guards against empty CIF."""
        try:
            from boltz.data.parse.mmcif_with_constraints import parse_mmcif as parse_mmcif_wc
        except ImportError as e:
            pytest.skip(f"Cannot import mmcif_with_constraints: {e}")

        cif_path = self._empty_cif(tmp_path, name="empty_wc.cif")

        with pytest.raises(ValueError, match="no models"):
            parse_mmcif_wc(str(cif_path))
