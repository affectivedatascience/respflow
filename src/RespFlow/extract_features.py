from RespFlow.access_files import map_files
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import os

#
# =============================================================================
#

"""
A collection of functions for extracting features from respiratory signals.
"""

#
# BREATH CYCLE DETECTION
# =============================================================================
#

def detect_breath_cycles(
    signal: np.ndarray,
    sampling_rate: float,
    min_bpm: float = 4.0,
    max_bpm: float = 60.0,
) -> dict:
    """
    Detect peaks (end-inspiration) and troughs (end-expiration) in a
    respiratory signal and return structured cycle information.

    Parameters
    ----------
    signal : np.ndarray
        1-D respiratory signal values (NaN-safe).
    sampling_rate : float
        Sampling rate in Hz.
    min_bpm : float, optional
        Minimum expected breathing rate in breaths per minute. Used to set
        the maximum allowed distance between peaks. The default is 4.0.
    max_bpm : float, optional
        Maximum expected breathing rate in breaths per minute. Used to set
        the minimum required distance between peaks. The default is 60.0.

    Returns
    -------
    cycles : dict
        Dictionary with keys:
        - 'peaks'        : np.ndarray of peak indices
        - 'troughs'      : np.ndarray of trough indices
        - 'peak_values'  : np.ndarray of signal values at peaks
        - 'trough_values': np.ndarray of signal values at troughs
        - 'n_cycles'     : int, number of complete breath cycles
    """

    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(signal)

    empty = {
        'peaks': np.array([], dtype=int),
        'troughs': np.array([], dtype=int),
        'peak_values': np.array([], dtype=float),
        'trough_values': np.array([], dtype=float),
        'n_cycles': 0,
    }

    # Need at least 10 finite samples to attempt any peak detection
    if valid.sum() < 10:
        return empty

    # Interpolate internal NaNs so find_peaks works on a continuous signal
    y = signal.copy()
    if np.isnan(y).any():
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

    # Minimum distance between peaks from max_bpm
    min_dist = max(1, int(round(sampling_rate * 60.0 / max_bpm)))

    # Robust prominence threshold (MAD-based, matching preprocess_signals)
    med = np.nanmedian(y)
    # 1.4826 is the standard MAD-to-sigma conversion factor for normal distributions
    mad = np.nanmedian(np.abs(y - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(y)
    if not np.isfinite(mad) or mad == 0:
        return empty

    # 0.5 * MAD: tuned for clean post-processed signals; higher than the 0.25
    # used in preprocess_signals._detect_troughs to reduce false positives
    prominence = 0.5 * mad

    # Detect peaks and troughs
    peaks, _ = find_peaks(y, distance=min_dist, prominence=prominence)
    troughs, _ = find_peaks(-y, distance=min_dist, prominence=prominence)

    if len(peaks) < 1 or len(troughs) < 1:
        return empty

    # Interleave into alternating trough-peak-trough-peak sequence
    peaks, troughs = _interleave_peaks_troughs(peaks, troughs, y)

    if len(peaks) < 1 or len(troughs) < 2:
        return empty

    return {
        'peaks': peaks,
        'troughs': troughs,
        'peak_values': signal[peaks],
        'trough_values': signal[troughs],
        'n_cycles': len(troughs) - 1,
    }


def _interleave_peaks_troughs(
    peaks: np.ndarray,
    troughs: np.ndarray,
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean peak and trough arrays so they strictly alternate starting with a
    trough: T, P, T, P, ... , T. When consecutive duplicates occur, keep the
    most extreme value.
    """

    # Merge into one sorted list with labels
    events = (
        [(int(idx), 'P') for idx in peaks]
        + [(int(idx), 'T') for idx in troughs]
    )
    events.sort(key=lambda x: x[0])

    clean_peaks = []
    clean_troughs = []
    expect = 'T'  # start with a trough

    i = 0
    while i < len(events):
        idx, kind = events[i]

        if kind == expect:
            # Collect consecutive same-type events and keep the best one
            group = [idx]
            while i + 1 < len(events) and events[i + 1][1] == kind:
                i += 1
                group.append(events[i][0])

            if kind == 'T':
                best = min(group, key=lambda j: signal[j])
                clean_troughs.append(best)
            else:
                best = max(group, key=lambda j: signal[j])
                clean_peaks.append(best)

            expect = 'P' if expect == 'T' else 'T'
        else:
            # Skip events that break alternation
            pass
        i += 1

    # Ensure sequence ends with a trough
    if len(clean_peaks) > 0 and len(clean_troughs) > 0:
        if clean_peaks[-1] > clean_troughs[-1]:
            clean_peaks.pop()

    return np.array(clean_peaks, dtype=int), np.array(clean_troughs, dtype=int)


#
# BASIC STATISTICS
# =============================================================================
#

def calc_min(Signal: pd.DataFrame, column_name: str) -> float:
    """
    Calculate the minimum value of a signal column.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the column to analyze.

    Returns
    -------
    float
        The minimum value.
    """
    if column_name not in Signal.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    return float(np.nanmin(Signal[column_name].values))


def calc_max(Signal: pd.DataFrame, column_name: str) -> float:
    """
    Calculate the maximum value of a signal column.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the column to analyze.

    Returns
    -------
    float
        The maximum value.
    """
    if column_name not in Signal.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    return float(np.nanmax(Signal[column_name].values))


def calc_mean(Signal: pd.DataFrame, column_name: str) -> float:
    """
    Calculate the mean value of a signal column.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the column to analyze.

    Returns
    -------
    float
        The mean value.
    """
    if column_name not in Signal.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    return float(np.nanmean(Signal[column_name].values))


def calc_sd(Signal: pd.DataFrame, column_name: str) -> float:
    """
    Calculate the standard deviation of a signal column.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the column to analyze.

    Returns
    -------
    float
        The standard deviation.
    """
    if column_name not in Signal.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    return float(np.nanstd(Signal[column_name].values))


#
# BREATHING RATE
# =============================================================================
#

def calc_breathing_rate(
    Signal: pd.DataFrame,
    column_name: str,
    sampling_rate: float,
    cycles: dict,
) -> float:
    """
    Calculate the breathing rate in breaths per minute.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    sampling_rate : float
        Sampling rate in Hz.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        Breathing rate in breaths per minute, or np.nan if insufficient data.
    """
    if cycles['n_cycles'] < 1:
        return np.nan

    # Use the span from first trough to last trough for accurate rate
    troughs = cycles['troughs']
    span_minutes = (troughs[-1] - troughs[0]) / sampling_rate / 60.0

    if span_minutes <= 0:
        return np.nan

    return cycles['n_cycles'] / span_minutes


#
# AMPLITUDE FEATURES
# =============================================================================
#

def calc_mean_insp_amplitude(
    Signal: pd.DataFrame,
    column_name: str,
    cycles: dict,
) -> float:
    """
    Calculate the mean inspiratory amplitude (trough-to-peak rise).

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        Mean inspiratory amplitude, or np.nan if insufficient cycles.
    """
    if cycles['n_cycles'] < 1:
        return np.nan

    values = Signal[column_name].values
    peaks = cycles['peaks']
    troughs = cycles['troughs']

    amplitudes = []
    for i, pk in enumerate(peaks):
        # Each peak is between troughs[i] and troughs[i+1]
        trough_val = values[troughs[i]]
        peak_val = values[pk]
        if np.isfinite(trough_val) and np.isfinite(peak_val):
            amplitudes.append(peak_val - trough_val)

    return float(np.mean(amplitudes)) if amplitudes else np.nan


def calc_mean_exp_amplitude(
    Signal: pd.DataFrame,
    column_name: str,
    cycles: dict,
) -> float:
    """
    Calculate the mean expiratory amplitude (peak-to-trough fall).

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        Mean expiratory amplitude, or np.nan if insufficient cycles.
    """
    if cycles['n_cycles'] < 1:
        return np.nan

    values = Signal[column_name].values
    peaks = cycles['peaks']
    troughs = cycles['troughs']

    amplitudes = []
    for i, pk in enumerate(peaks):
        # Expiratory: from peak down to the following trough
        peak_val = values[pk]
        trough_val = values[troughs[i + 1]]
        if np.isfinite(peak_val) and np.isfinite(trough_val):
            amplitudes.append(peak_val - trough_val)

    return float(np.mean(amplitudes)) if amplitudes else np.nan


def calc_mean_amplitude(
    Signal: pd.DataFrame,
    column_name: str,
    cycles: dict,
) -> float:
    """
    Calculate the mean breath amplitude (peak-to-trough distance per cycle),
    a proxy for tidal volume.

    Computed as the average of (peak_value - mean of surrounding troughs) for
    each complete cycle.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        Mean breath amplitude, or np.nan if insufficient cycles.
    """
    if cycles['n_cycles'] < 1:
        return np.nan

    values = Signal[column_name].values
    peaks = cycles['peaks']
    troughs = cycles['troughs']

    amplitudes = []
    for i, pk in enumerate(peaks):
        t_before = values[troughs[i]]
        t_after = values[troughs[i + 1]]
        peak_val = values[pk]
        if np.isfinite(t_before) and np.isfinite(t_after) and np.isfinite(peak_val):
            baseline = (t_before + t_after) / 2.0
            amplitudes.append(peak_val - baseline)

    return float(np.mean(amplitudes)) if amplitudes else np.nan


#
# TIMING FEATURES
# =============================================================================
#

def calc_mean_insp_time(
    Signal: pd.DataFrame,
    column_name: str,
    sampling_rate: float,
    cycles: dict,
) -> float:
    """
    Calculate the mean inspiratory time (Ti) in seconds.

    Inspiratory time is measured from each trough to the following peak.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    sampling_rate : float
        Sampling rate in Hz.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        Mean inspiratory time in seconds, or np.nan if insufficient cycles.
    """
    if cycles['n_cycles'] < 1:
        return np.nan

    peaks = cycles['peaks']
    troughs = cycles['troughs']

    ti_values = []
    for i, pk in enumerate(peaks):
        ti_values.append((pk - troughs[i]) / sampling_rate)

    return float(np.mean(ti_values)) if ti_values else np.nan


def calc_mean_exp_time(
    Signal: pd.DataFrame,
    column_name: str,
    sampling_rate: float,
    cycles: dict,
) -> float:
    """
    Calculate the mean expiratory time (Te) in seconds.

    Expiratory time is measured from each peak to the following trough.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    sampling_rate : float
        Sampling rate in Hz.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        Mean expiratory time in seconds, or np.nan if insufficient cycles.
    """
    if cycles['n_cycles'] < 1:
        return np.nan

    peaks = cycles['peaks']
    troughs = cycles['troughs']

    te_values = []
    for i, pk in enumerate(peaks):
        te_values.append((troughs[i + 1] - pk) / sampling_rate)

    return float(np.mean(te_values)) if te_values else np.nan


def calc_duty_cycle(
    Signal: pd.DataFrame,
    column_name: str,
    sampling_rate: float,
    cycles: dict,
) -> float:
    """
    Calculate the inspiratory duty cycle (Ti / Ttot).

    A value of ~0.4 is typical for relaxed breathing.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    sampling_rate : float
        Sampling rate in Hz.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        Duty cycle ratio (Ti / Ttot), or np.nan if insufficient cycles.
    """
    ti = calc_mean_insp_time(Signal, column_name, sampling_rate, cycles)
    te = calc_mean_exp_time(Signal, column_name, sampling_rate, cycles)

    if np.isnan(ti) or np.isnan(te) or (ti + te) == 0:
        return np.nan

    return ti / (ti + te)


#
# VARIABILITY FEATURES
# =============================================================================
#

def calc_cycle_variability(
    Signal: pd.DataFrame,
    column_name: str,
    sampling_rate: float,
    cycles: dict,
) -> float:
    """
    Calculate the coefficient of variation (CV) of breath cycle durations.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    sampling_rate : float
        Sampling rate in Hz.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        CV of cycle durations, or np.nan if fewer than 3 cycles.
    """
    # Need >= 3 cycles for a meaningful CV (>= 3 durations to compute variance)
    if cycles['n_cycles'] < 3:
        return np.nan

    troughs = cycles['troughs']
    durations = np.diff(troughs) / sampling_rate

    mean_dur = np.mean(durations)
    if mean_dur == 0:
        return np.nan

    return float(np.std(durations) / mean_dur)


def calc_rmssd_intervals(
    Signal: pd.DataFrame,
    column_name: str,
    sampling_rate: float,
    cycles: dict,
) -> float:
    """
    Calculate the root mean square of successive differences (RMSSD) of
    breath cycle intervals. Analogous to HRV RMSSD.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    sampling_rate : float
        Sampling rate in Hz.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        RMSSD of breath intervals in seconds, or np.nan if fewer than 3 cycles.
    """
    # Need >= 3 cycles (>= 3 durations) to get >= 2 successive differences
    if cycles['n_cycles'] < 3:
        return np.nan

    troughs = cycles['troughs']
    durations = np.diff(troughs) / sampling_rate
    successive_diffs = np.diff(durations)

    return float(np.sqrt(np.mean(successive_diffs ** 2)))


def calc_cv_amplitude(
    Signal: pd.DataFrame,
    column_name: str,
    cycles: dict,
) -> float:
    """
    Calculate the coefficient of variation (CV) of breath amplitudes.

    Parameters
    ----------
    Signal : pd.DataFrame
        DataFrame containing the signal data.
    column_name : str
        Name of the signal column.
    cycles : dict
        Pre-computed breath cycle information from detect_breath_cycles().

    Returns
    -------
    float
        CV of breath amplitudes, or np.nan if fewer than 3 cycles.
    """
    # Need >= 3 cycles for a meaningful CV
    if cycles['n_cycles'] < 3:
        return np.nan

    values = Signal[column_name].values
    peaks = cycles['peaks']
    troughs = cycles['troughs']

    amplitudes = []
    for i, pk in enumerate(peaks):
        t_before = values[troughs[i]]
        t_after = values[troughs[i + 1]]
        peak_val = values[pk]
        if np.isfinite(t_before) and np.isfinite(t_after) and np.isfinite(peak_val):
            baseline = (t_before + t_after) / 2.0
            amplitudes.append(peak_val - baseline)

    if len(amplitudes) < 3:
        return np.nan

    amplitudes = np.array(amplitudes)
    mean_amp = np.mean(amplitudes)
    if mean_amp == 0:
        return np.nan

    return float(np.std(amplitudes) / mean_amp)


#
# MAIN EXTRACTION FUNCTION
# =============================================================================
#

# Feature names in output order
_MEASURE_NAMES = [
    'Min',
    'Max',
    'Mean',
    'SD',
    'Breathing_Rate',
    'Mean_Insp_Amplitude',
    'Mean_Exp_Amplitude',
    'Mean_Amplitude',
    'Mean_Insp_Time',
    'Mean_Exp_Time',
    'Duty_Cycle',
    'Cycle_Variability',
    'RMSSD_Intervals',
    'CV_Amplitude',
]


def extract_features(
    path_names: dict,
    column_names: list[str] | None = None,
    sampling_rate: float = 2000.0,
    expression: str | None = None,
    file_ext: str = 'csv',
    short_name: bool = True,
) -> pd.DataFrame:
    """
    Extract respiratory features from all processed signal files.

    Reads cleaned signals from the 'smooth' stage directory in path_names,
    computes 14 features per signal column, and saves a Features.csv to the
    'feature' directory.

    Parameters
    ----------
    path_names : dict
        Dictionary of pipeline stage paths (from make_paths()). Must contain
        'smooth' and 'feature' keys.
    column_names : list[str], optional
        Signal columns to analyze. If None, auto-detects all non-Time columns
        from the first file.
    sampling_rate : float, optional
        Sampling rate in Hz. The default is 2000.0.
    expression : str, optional
        Regex to filter which files to process. The default is None (all files).
    file_ext : str, optional
        File extension to read. The default is 'csv'.
    short_name : bool, optional
        If True, use relative paths as file identifiers. If False, use full
        absolute paths. The default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per file and columns for File_Path plus each
        extracted feature (prefixed by signal column name).
    """

    if 'smooth' not in path_names:
        raise ValueError("'smooth' key not found in path_names.")
    if 'feature' not in path_names:
        raise ValueError("'feature' key not found in path_names.")

    mapped_files = map_files(
        path_names['smooth'], file_ext=file_ext, expression=expression
    )

    if len(mapped_files) == 0:
        raise ValueError(
            f"No .{file_ext} files found in {path_names['smooth']}."
        )

    # Auto-detect columns from first file if not specified
    if column_names is None:
        first_file = next(iter(mapped_files.values()))
        first_df = pd.read_csv(first_file)
        column_names = [
            c for c in first_df.columns if c.lower() != 'time'
        ]
        if not column_names:
            raise ValueError(
                f"No signal columns found in {first_file} "
                "(only a Time column was present)."
            )

    # Build output column names
    out_columns = ['File_Path']
    for col in column_names:
        for measure in _MEASURE_NAMES:
            out_columns.append(f"{col}_{measure}")

    rows = []

    for file_key, file_path in mapped_files.items():
        df = pd.read_csv(file_path)

        # Validate columns exist
        missing = [c for c in column_names if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} not found in {file_path}. "
                f"Available: {list(df.columns)}"
            )

        file_id = file_key if short_name else file_path

        row = [file_id]

        for col in column_names:
            signal_values = df[col].values
            cycles = detect_breath_cycles(signal_values, sampling_rate)

            row.append(calc_min(df, col))
            row.append(calc_max(df, col))
            row.append(calc_mean(df, col))
            row.append(calc_sd(df, col))
            row.append(calc_breathing_rate(df, col, sampling_rate, cycles))
            row.append(calc_mean_insp_amplitude(df, col, cycles))
            row.append(calc_mean_exp_amplitude(df, col, cycles))
            row.append(calc_mean_amplitude(df, col, cycles))
            row.append(calc_mean_insp_time(df, col, sampling_rate, cycles))
            row.append(calc_mean_exp_time(df, col, sampling_rate, cycles))
            row.append(calc_duty_cycle(df, col, sampling_rate, cycles))
            row.append(calc_cycle_variability(df, col, sampling_rate, cycles))
            row.append(calc_rmssd_intervals(df, col, sampling_rate, cycles))
            row.append(calc_cv_amplitude(df, col, cycles))

        rows.append(row)

    features_df = pd.DataFrame(rows, columns=out_columns)
    features_df.sort_values('File_Path', inplace=True)
    features_df.reset_index(drop=True, inplace=True)

    # Save to feature directory
    os.makedirs(path_names['feature'], exist_ok=True)
    out_path = os.path.join(path_names['feature'], 'Features.csv')
    features_df.to_csv(out_path, index=False)

    print(
        f"Extracted features from {len(mapped_files)} files "
        f"to {out_path}"
    )

    return features_df
