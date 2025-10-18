import os
import pytest
import RESPFlow

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
    paths = RESPFlow.make_paths()

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

    paths = RESPFlow.make_paths(root=custom_root, raw=custom_raw)

    # Assert custom raw should be absolute and point to my_raw
    assert paths['raw'] == "/abs/my_raw"

    # Assert other folders should be based on custom root
    assert paths['notch'] == "/abs/my_root/2_notch"
    assert paths['bandpass'] == "/abs/my_root/3_bandpass"

    # Assert makedirs was called for every path
    assert set(mock_filesystem) == set(paths.values())
    
# ============================================================================
# End of tests
# ============================================================================