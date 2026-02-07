from RespFlow.access_files import map_files
from scipy.signal import butter, sosfiltfilt
import pandas as pd
from pathlib import Path
from scipy.ndimage import median_filter
import numpy as np
from adtk.detector import AutoregressionAD
from adtk.data import validate_series

#
# =============================================================================
#

"""
A collection of functions for preprocessing signals.
"""

#
# DETREND
# =============================================================================
#

def apply_detrend(signal: list | tuple, sampling_rate: int, window_size_seconds: int = 60) -> tuple:
    # 60-second window default based on BreathMetrics paper
    # if signal is shorter than window, use global median

    W0 = window_size_seconds  # seconds
    N = len(signal)
    signal_duration = N / sampling_rate

    use_rolling = (signal_duration >= W0)

    if use_rolling:
        k = int(round(W0 * sampling_rate))  # window length in samples
        if k % 2 == 0:
            k += 1

        # Guard: do not use a window longer than the signal
        if k <= N:
            s = pd.Series(signal)

            # NaN-safe centered rolling median baseline
            baseline = (
                s.rolling(window=k, center=True, min_periods=max(1, k // 2))
                .median()
            )

            # If baseline has NaNs (edges or long NaN runs), fill from nearest valid values
            baseline = baseline.ffill().bfill().to_numpy()
        else:
            baseline = np.full(N, np.nanmedian(signal))
    else:
        baseline = np.full(N, np.nanmedian(signal))


    # Detrend
    detrended_signal = signal - baseline

    return detrended_signal, baseline

def detrend_signals(in_path: str, out_path: str, sampling_rate: int, window_size_seconds: int = 60) -> None:
    """
    Applies detrending to all columns except 'time' in all CSV files.
    Preserves the folder structure from in_path to out_path.

    Parameters:
    -----------
    in_path : str
        Input directory path
    out_path : str
        Output directory path
    sampling_rate : float
        Sampling rate in Hz
    window_size_seconds : int, optional
        Window size for median filter in seconds (default: 60)
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Apply detrending to all columns except 'time'
        for column in df.columns:
            if column.lower() != 'time':
                detrended_signal, baseline = apply_detrend(df[column].values, sampling_rate, window_size_seconds)
                df[column] = detrended_signal
                df[f"{column}_baseline"] = baseline

        # Determine the relative path from in_path to preserve folder structure
        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(in_path_obj)

        # Create output path
        output_file_path = out_path_obj / relative_path

        # Create output directory if it doesn't exist
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the detrended data
        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# BANDPASS
# =============================================================================
#

def apply_bandpass(data: list | tuple, sampling_rate: int, lowcut: float = 0.05, highcut: float = 2.0, order: int = 2) -> list | tuple:
    """
    Applies a zero-phase Butterworth bandpass filter.
    Standard: 0.05-2.0 Hz for RIP belt data.
    """
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design filter
    sos = butter(order, [low, high], btype='band', output='sos')

    # Apply zero-phase filter (filtfilt) with padlen adjusted for short signals
    padlen = min(len(data) - 1, 15)
    y = sosfiltfilt(sos, data, padlen=padlen)

    return y


# Strictly needs path_names (raw files), and sampling rate
# optional is upper and lower frequency for bandpass filter
def bandpass_filter_signals(in_path: str, out_path: str, sampling_rate: int, passband: str | tuple = 'default', order: int = 2) -> None:
    """
    Applies a Butterworth bandpass filter to all columns except 'time' in all CSV files.
    Preserves the folder structure from in_path to out_path.

    Parameters:
    -----------
    in_path : str
        Input directory path
    out_path : str
        Output directory path
    sampling_rate : float
        Sampling rate in Hz
    passband : str or tuple, optional
        Preset name or tuple of (lowcut, highcut) in Hz.
        Presets: 'default' (0.05-2.0 Hz), 'resting_adult' (0.05-1.0 Hz),
        'narrow_band' (0.1-0.35 Hz), 'wide_band' (0.05-3.0 Hz).
        Default: 'resting_adult'
    order : int, optional
        Filter order (default: 2)
    """
    
    PASSBANDS = {
        'default': (0.05, 2.0),
        'resting_adult': (0.05, 1),
        'narrow_band': (0.1, 0.35),
        'wide_band': (0.05, 3.0)
    }
    
    # Determine lowcut and highcut from passband argument
    if isinstance(passband, str):
        if passband in PASSBANDS:
            lowcut, highcut = PASSBANDS[passband]
        else:
            raise ValueError(f"Unknown passband preset '{passband}'. Available: {list(PASSBANDS.keys())}")
    elif isinstance(passband, (tuple, list)) and len(passband) == 2:
        lowcut, highcut = passband
    else:
        raise ValueError("passband must be a string preset or a tuple of (lowcut, highcut)")

    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Apply bandpass filter to all columns except 'time'
        for column in df.columns:
            if column.lower() != 'time':
                df[column] = apply_bandpass(df[column].values, sampling_rate, lowcut, highcut, order)

        # Determine the relative path from in_path to preserve folder structure
        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(in_path_obj)

        # Create output path
        output_file_path = out_path_obj / relative_path

        # Create output directory if it doesn't exist
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the filtered data
        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# ANOMALY DETECTION
# =============================================================================
#

def _merge_anomaly_intervals(
    anomaly_times: np.ndarray,
    time_step: float = 0.0005,
    step_tol: float = 0.005,
    merge_gap: float = 10.0
) -> list[tuple[float, float]]:
    """
    Convert anomaly time values to intervals and merge nearby ones.

    Ported from AD_Format in TookKitTester.ipynb.

    Parameters:
    -----------
    anomaly_times : np.ndarray
        Time values where anomalies were detected
    time_step : float
        Expected time step between consecutive samples (default 0.0005 for 2000Hz)
    step_tol : float
        Tolerance for considering samples consecutive (default 0.005)
    merge_gap : float
        Maximum time gap (seconds) to merge adjacent intervals (default 10.0)

    Returns:
    --------
    list of (start_time, end_time) tuples representing merged anomaly intervals
    """
    if len(anomaly_times) == 0:
        return []

    # Step 1: Group consecutive anomaly times into initial ranges
    section = [anomaly_times[0]]
    ranges = []

    for i in range(1, len(anomaly_times)):
        diff = anomaly_times[i] - anomaly_times[i - 1]
        if abs(diff - time_step) < step_tol:
            # Consecutive - add to current section
            section.append(anomaly_times[i])
        else:
            # Gap detected - save current section and start new one
            ranges.append((section[0], section[-1]))
            section = [anomaly_times[i]]

    # Don't forget the last section
    if section:
        ranges.append((section[0], section[-1]))

    if len(ranges) == 0:
        return []

    # Step 2: Merge ranges that are close together (within merge_gap seconds)
    merged = []
    current_start, current_end = ranges[0]

    for next_start, next_end in ranges[1:]:
        gap = next_start - current_end
        if gap < merge_gap:
            # Merge: extend current interval
            current_end = next_end
        else:
            # Don't merge: save current and start new
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    # Append the final interval
    merged.append((current_start, current_end))

    return merged


def apply_anomaly_detection(
    signal: np.ndarray,
    time_values: np.ndarray | None = None,
    c: float = 4.0,
    side: str = "both",
    n_steps: int = 3,
    step_size: int = 50,
    merge_gap: float = 10
) -> np.ndarray:
    """
    Detect anomalies using AutoregressionAD with interval merging.
    Returns a binary mask (0 = normal, 1 = anomaly).

    Parameters:
    -----------
    signal : array-like
        The signal to process
    time_values : array-like, optional
        Time values for the signal (used for interval merging).
        If None, merging is skipped.
    c : float
        Coefficient for anomaly threshold (higher = less sensitive). Default 4.0.
    side : str
        'positive', 'negative', or 'both'
    n_steps : int
        Number of AR lags. Default 3.
    step_size : int
        Step size between lags. Default 50.
    merge_gap : float
        Maximum time gap (seconds) to merge adjacent anomaly intervals. Default 10.0.

    Returns:
    --------
    np.ndarray
        Binary array where 1 indicates anomaly, 0 indicates normal
    """
    import warnings

    # Create a datetime index for adtk compatibility
    # adtk requires a DatetimeIndex for its time series operations
    datetime_index = pd.date_range(start='2000-01-01', periods=len(signal), freq='ms')
    series = pd.Series(signal, index=datetime_index)

    # Validate and prepare series using adtk's validate_series
    series = validate_series(series)

    # Check if there are enough valid (non-NaN) values for training
    # AutoregressionAD needs at least n_steps * step_size valid consecutive values
    min_required = n_steps * step_size + 1
    valid_count = series.notna().sum()

    if valid_count < min_required:
        # Not enough data for anomaly detection, return all zeros
        return np.zeros(len(signal), dtype=int)

    # Initialize and fit the detector
    detector = AutoregressionAD(
        c=c,
        side=side,
        n_steps=n_steps,
        step_size=step_size
    )

    try:
        # Suppress pandas FutureWarnings from adtk
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # Detect anomalies - returns a boolean Series
            anomalies = detector.fit_detect(series)

            # Get raw anomaly mask
            raw_mask = anomalies.fillna(False).values

    except RuntimeError:
        # If adtk fails (e.g., not enough valid consecutive values), return all zeros
        return np.zeros(len(signal), dtype=int)

    # If time_values provided, apply interval merging (matches TookKitTester.ipynb logic)
    if time_values is not None and len(time_values) == len(signal):
        # Get indices where anomalies were detected
        anomaly_indices = np.where(raw_mask)[0]

        if len(anomaly_indices) == 0:
            return np.zeros(len(signal), dtype=int)

        # Get time values at anomaly indices
        anomaly_times = time_values[anomaly_indices]

        # Compute time step from data (for step_tol calculation)
        if len(time_values) > 1:
            time_step = np.median(np.diff(time_values))
            step_tol = time_step * 10  # Allow some tolerance
        else:
            time_step = 0.0005
            step_tol = 0.005

        # Merge intervals
        merged_intervals = _merge_anomaly_intervals(
            anomaly_times,
            time_step=time_step,
            step_tol=step_tol,
            merge_gap=merge_gap
        )

        # Create final binary mask from merged intervals
        binary_mask = np.zeros(len(signal), dtype=int)
        for start_time, end_time in merged_intervals:
            mask_interval = (time_values >= start_time) & (time_values <= end_time)
            binary_mask[mask_interval] = 1
    else:
        # No time values - use raw mask directly
        binary_mask = raw_mask.astype(int)

    return binary_mask


def detect_anomalies(
    in_path: str,
    out_path: str,
    c: float = 5.0,
    side: str = "both",
    n_steps: int = 3,
    step_size: int = 100,
    merge_gap: float = 5
) -> None:
    """
    Applies anomaly detection to all CSV files.
    Adds {column}_anomaly columns with binary 0/1 flags.
    Preserves folder structure from in_path to out_path.

    Parameters:
    -----------
    in_path : str
        Input directory path
    out_path : str
        Output directory path
    c : float
        Coefficient for anomaly threshold (higher = less sensitive). Default 4.0.
    side : str
        'positive', 'negative', or 'both'
    n_steps : int
        Number of AR lags. Default 3.
    step_size : int
        Step size between lags. Default 50.
    merge_gap : float
        Maximum time gap (seconds) to merge adjacent anomaly intervals. Default 10.0.
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Try to get time values from the dataframe
        time_col = None
        for col in df.columns:
            if col.lower() == 'time':
                time_col = col
                break
        time_values = df[time_col].values if time_col else None

        # Apply anomaly detection to all columns except 'time' and baseline columns
        for column in df.columns:
            if column.lower() != 'time' and not column.endswith('_baseline'):
                anomaly_mask = apply_anomaly_detection(
                    df[column].values,
                    time_values=time_values,
                    c=c,
                    side=side,
                    n_steps=n_steps,
                    step_size=step_size,
                    merge_gap=merge_gap
                )
                df[f"{column}_anomaly"] = anomaly_mask

        # Determine the relative path from in_path to preserve folder structure
        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(in_path_obj)

        # Create output path
        output_file_path = out_path_obj / relative_path

        # Create output directory if it doesn't exist
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the data with anomaly flags
        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# =============================================================================
#

