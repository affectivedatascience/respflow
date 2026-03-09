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
from RespFlow.access_files import make_paths, map_files, make_sample_data

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
        'raw', 'hard_fault', 'detrend', 'micro_interp', 'bandpass', 'fwr',
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
    assert paths['detrend'] == "/abs/my_root/3_detrend"
    assert paths['micro_interp'] == "/abs/my_root/4_micro_interp"
    assert paths['bandpass'] == "/abs/my_root/5_bandpass"

    # Assert makedirs was called for every path
    assert set(mock_filesystem) == set(paths.values())
    
# ============================================================================
# Test map_files function
# ============================================================================

def _touch(p: str, text: str = "") -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def test_map_files_basic_recursive(tmp_path):
    # data/
    #   10/file1.csv
    #   10/nested/info.txt
    #   23/file2.csv
    root = tmp_path / "data"
    f1 = root / "10" / "file1.csv"
    f_txt = root / "10" / "nested" / "info.txt"
    f2 = root / "23" / "file2.csv"
    _touch(str(f1), "a,b\n1,2\n")
    _touch(str(f_txt), "hello\n")
    _touch(str(f2), "x,y\n3,4\n")

    result = map_files(str(root))  # default file_ext='csv'
    
    # Only CSVs should appear
    keys = set(result.keys())
    assert os.path.join("10", "file1.csv") in keys
    assert os.path.join("23", "file2.csv") in keys
    
    # Non-csv shouldn't be included
    assert not any("info.txt" in k for k in keys)

    # Values should be absolute paths pointing to the files
    assert os.path.isabs(result[os.path.join("10", "file1.csv")])
    assert os.path.exists(result[os.path.join("23", "file2.csv")])


def test_map_files_extension_filter(tmp_path):
    root = tmp_path / "data"
    _touch(str(root / "a.csv"), "csv\n")
    _touch(str(root / "b.txt"), "txt\n")

    only_txt = map_files(str(root), file_ext="txt")

    assert set(only_txt.keys()) == {"b.txt"}

    only_csv = map_files(str(root), file_ext="csv")
    
    assert set(only_csv.keys()) == {"a.csv"}


def test_map_files_regex_filter_on_filename(tmp_path):
    root = tmp_path / "data"
    _touch(str(root / "10" / "keep_this.csv"), "ok\n")
    _touch(str(root / "10" / "skip_this.csv"), "no\n")
    _touch(str(root / "23" / "also_skip.csv"), "no\n")

    # Match only files whose name ends with 'keep_this.csv'
    expr = r".*keep_this\.csv$"
    
    result = map_files(str(root), expression=expr)
    
    keys = set(result.keys())
    assert any(k.endswith(os.path.join("10", "keep_this.csv")) for k in keys)
    assert not any(k.endswith("skip_this.csv") for k in keys)

def test_map_files_invalid_regex_raises(tmp_path):
    root = tmp_path / "data"
    _touch(str(root / "a.csv"), "a\n")

    with pytest.raises(Exception, match="Invalid regex expression"):
        map_files(str(root), expression="(")  # invalid regex

# ============================================================================
# Tests for make_sample_data function
# ============================================================================

def _fake_pkg_with_data(tmp_path: Path) -> Path:
    """Create a fake RespFlow/data tree with a few files."""
    root = tmp_path / "RespFlow_fake"
    (root / "data" / "10").mkdir(parents=True, exist_ok=True)
    (root / "data" / "10" / "file1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "data" / "10" / "nested").mkdir(parents=True, exist_ok=True)
    (root / "data" / "10" / "nested" / "info.txt").write_text("hello\n", encoding="utf-8")
    (root / "data" / "23").mkdir(parents=True, exist_ok=True)
    (root / "data" / "23" / "file2.csv").write_text("x,y\n3,4\n", encoding="utf-8")
    return root


def test_copies_all_and_preserves_structure(tmp_path: Path, monkeypatch):
    fake_pkg = _fake_pkg_with_data(tmp_path)
    # Make ir.files("RespFlow") point to our fake package root
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