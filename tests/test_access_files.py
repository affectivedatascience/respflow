# ============================================================================
# Imports
# ============================================================================

# --- Standard library ---
import os
from pathlib import Path
from importlib import resources as ir

# --- Third-party packages ---
import pytest

# --- Local application / package imports ---
from RESPFlow.access_files import make_paths, map_files, make_sample_data

# ============================================================================
# Tests
# ===========================================================================

# ============================================================================
# Test make_paths function
# ============================================================================
@pytest.fixture
def mock_filesystem(monkeypatch):
    """Fixture to mock filesystem interactions for all tests."""
    faked_paths = []

    # Mock os.makedirs to record calls
    def fake_makedirs(path, exist_ok=True):
        faked_paths.append(path)
    monkeypatch.setattr(os, "makedirs", fake_makedirs)

    # Mock abspath and getcwd for consistent paths
    monkeypatch.setattr(os.path, "abspath", lambda x: f"/abs/{x}")
    monkeypatch.setattr(os, "getcwd", lambda: "/abs/cwd")

    return faked_paths


def test_make_paths_defaults(mock_filesystem):
    paths = make_paths()

    expected_keys = {
        'raw', 'notch', 'bandpass', 'fwr',
        'screened', 'filled', 'smooth', 'feature'
    }
    
    # Assert all expected keys are present
    assert set(paths.keys()) == expected_keys

    # Assert all paths should start with /abs/
    for path in paths.values():
        assert path.startswith("/abs/")

    # Assert makedirs was called for every path
    assert set(mock_filesystem) == set(paths.values())


def test_make_paths_custom_root_raw(mock_filesystem):
    custom_root = "my_root"
    custom_raw = "my_raw"

    paths = make_paths(root=custom_root, raw=custom_raw)

    # Assert custom raw should be absolute and point to my_raw
    assert paths['raw'] == "/abs/my_raw"

    # Assert other folders should be based on custom root
    assert paths['notch'] == "/abs/my_root/2_notch"
    assert paths['bandpass'] == "/abs/my_root/3_bandpass"

    # Assert makedirs was called for every path
    assert set(mock_filesystem) == set(paths.values())
    
# ============================================================================
# Test map_files function
# ============================================================================

# ============================================================================
# Tests for make_sample_data function
# ============================================================================

def _fake_pkg_with_data(tmp_path: Path) -> Path:
    """Create a fake RESPFlow/data tree with a few files."""
    root = tmp_path / "RESPFlow_fake"
    (root / "data" / "10").mkdir(parents=True, exist_ok=True)
    (root / "data" / "10" / "file1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "data" / "10" / "nested").mkdir(parents=True, exist_ok=True)
    (root / "data" / "10" / "nested" / "info.txt").write_text("hello\n", encoding="utf-8")
    (root / "data" / "23").mkdir(parents=True, exist_ok=True)
    (root / "data" / "23" / "file2.csv").write_text("x,y\n3,4\n", encoding="utf-8")
    return root


def test_copies_all_and_preserves_structure(tmp_path: Path, monkeypatch):
    fake_pkg = _fake_pkg_with_data(tmp_path)
    # Make ir.files("RESPFlow") point to our fake package root
    monkeypatch.setattr(ir, "files", lambda _pkg: fake_pkg)

    dest = tmp_path / "raw"
    make_sample_data({"raw": str(dest)})

    assert (dest / "10" / "file1.csv").is_file()
    assert (dest / "10" / "nested" / "info.txt").is_file()
    assert (dest / "23" / "file2.csv").is_file()
    # spot-check contents
    assert (dest / "10" / "file1.csv").read_text(encoding="utf-8").startswith("a,b\n")


def test_missing_raw_raises(monkeypatch, tmp_path: Path):
    fake_pkg = _fake_pkg_with_data(tmp_path)
    monkeypatch.setattr(ir, "files", lambda _pkg: fake_pkg)

    with pytest.raises(ValueError, match="Raw path not detected"):
        make_sample_data({})

# ============================================================================
# End of tests
# ============================================================================