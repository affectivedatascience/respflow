import pytest
import pandas as pd
from pathlib import Path

# ============================================================================
# Test integrity of data in ../data function
# ============================================================================

DATA_DIR = Path("./src/RespFlow/data")
EXPECTED_COLUMNS = {"Time", "Respiration"}

@pytest.fixture(scope="module")
def csv_files():
    """Fixture that collects all CSV files in the data directory."""
    files = list(DATA_DIR.rglob("*.csv"))
    assert files, "No CSV files found in data/ directory."
    return files

def test_csv_columns(csv_files):
    """Ensure all CSVs have at least Time and Respiration columns."""
    bad_files = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, nrows=1)
        cols = set(df.columns)

        # Check that both expected columns exist
        if not EXPECTED_COLUMNS.issubset(cols):
            bad_files.append((csv_file, f"Missing columns: {EXPECTED_COLUMNS - cols}"))

        # # Check that no unexpected columns exist
        # extra_cols = cols - EXPECTED_COLUMNS
        # if extra_cols:
        #     bad_files.append((csv_file, f"Unexpected columns: {extra_cols}"))

    assert not bad_files, (
        "❌ Some files failed column validation:\n"
        + "\n".join(f"{path}: {msg}" for path, msg in bad_files)
    )

# ============================================================================
# End of tests
# ============================================================================