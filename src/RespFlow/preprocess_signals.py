from RespFlow.access_files import map_files
from scipy.signal import butter, sosfiltfilt
import pandas as pd
from pathlib import Path
from scipy.ndimage import median_filter
import numpy as np

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


def min_viable_length_sosfiltfilt(
    sampling_rate: int,
    lowcut: float = 0.05,
    highcut: float = 2.0,
    order: int = 2,
) -> dict:
    """
    Compute the SciPy sosfiltfilt *default* padlen for a Butterworth bandpass and
    return the minimum viable segment length N_min such that padlen < N-1.

    Returns:
        {
          "n_sections": int,
          "padlen_default": int,
          "min_sequence_length": int
        }
    """
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    sos = butter(order, [low, high], btype="band", output="sos")
    n_sections = sos.shape[0]

    # From SciPy docs for sosfiltfilt default padding length:
    # padlen_default = 3 * (2*n_sections + 1 - min(z0, p0))
    # where z0 is the number of zeros at the origin, p0 is the number of poles at the origin.
    z0 = int(np.sum(sos[:, 2] == 0.0))  # b2 == 0 indicates a zero at z=0
    p0 = int(np.sum(sos[:, 5] == 0.0))  # a2 == 0 indicates a pole at z=0
    padlen_default = int(3 * (2 * n_sections + 1 - min(z0, p0)))

    # sosfiltfilt requires padlen < N-1  =>  N >= padlen + 2
    min_sequence_length = padlen_default + 2

    return {
        "n_sections": int(n_sections),
        "padlen_default": int(padlen_default),
        "min_sequence_length": int(min_sequence_length),
    }
    
def nan_gap_indices(x: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Return (start, end, length) for each contiguous NaN gap in a 1D array.
    Indices are half-open: x[start:end] are all NaN.
    """
    x = np.asarray(x)
    is_nan = np.isnan(x)

    if not np.any(is_nan):
        return []

    d = np.diff(is_nan.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1

    if is_nan[0]:
        starts = np.r_[0, starts]
    if is_nan[-1]:
        ends = np.r_[ends, len(x)]

    return [(s, e, e - s) for s, e in zip(starts.tolist(), ends.tolist())]


def interpolate_nan_gaps(
    data: np.ndarray,
    method: str,
    max_gap: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate NaN gaps in data.

    Parameters:
        data: 1D array with NaN gaps
        method: "pchip" or "cubic"
        max_gap: If provided, only interpolate gaps with length <= max_gap

    Returns:
        (interpolated_data, nan_mask) where nan_mask marks original NaN positions
    """
    from scipy.interpolate import PchipInterpolator, CubicSpline

    data = np.asarray(data, dtype=float)
    nan_mask = np.isnan(data)

    if not np.any(nan_mask):
        return data.copy(), nan_mask

    result = data.copy()
    gaps = nan_gap_indices(data)

    # Determine which gaps to interpolate
    if max_gap is not None:
        gaps_to_fill = [(s, e) for s, e, length in gaps if length <= max_gap]
    else:
        gaps_to_fill = [(s, e) for s, e, _ in gaps]

    if not gaps_to_fill:
        return result, nan_mask

    # Build mask of indices to interpolate
    fill_mask = np.zeros(len(data), dtype=bool)
    for s, e in gaps_to_fill:
        fill_mask[s:e] = True

    # Get valid (non-NaN) indices and values for interpolation
    valid_idx = np.where(~nan_mask)[0]
    valid_vals = data[valid_idx]

    if len(valid_idx) < 2:
        return result, nan_mask  # Can't interpolate with < 2 points

    # Create interpolator defaulting to pchip
    if method == "pchip":
        interp = PchipInterpolator(valid_idx, valid_vals)
    elif method == "cubic_spline":
        interp = CubicSpline(valid_idx, valid_vals)
    else:
        raise ValueError("Invalid interpolation method specified")
        
    

    # Fill only the gaps we want to fill
    fill_idx = np.where(fill_mask)[0]
    result[fill_idx] = interp(fill_idx)

    return result, nan_mask


def nan_islands(x: np.ndarray) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs for contiguous non-NaN regions ("islands")
    in a 1D array x. Indices are half-open: [start, end).

    Example: if x[10:25] are non-NaN, returns (10, 25).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("nan_islands expects a 1D array")

    valid = ~np.isnan(x)

    # Find rising edges (False->True) and falling edges (True->False)
    d = np.diff(valid.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1

    # Handle island starting at index 0
    if valid[0]:
        starts = np.r_[0, starts]

    # Handle island ending at last index
    if valid[-1]:
        ends = np.r_[ends, len(x)]

    return list(zip(starts.tolist(), ends.tolist()))


def iter_nan_islands(x: np.ndarray):
    """
    Generator yielding (start, end, segment) for each non-NaN island.
    """
    x = np.asarray(x)
    for start, end in nan_islands(x):
        yield start, end, x[start:end]


def apply_bandpass_nan_safe(
    data: np.ndarray,
    sampling_rate: int,
    lowcut: float,
    highcut: float,
    order: int,
    max_gap: int | None = None,
    interp_method: str = "pchip"
) -> np.ndarray:
    """
    NaN-safe bandpass filter using interpolation.

    Parameters:
        data: Input signal (may contain NaN)
        sampling_rate, lowcut, highcut, order: Filter parameters
        max_gap: Max gap size (samples) to interpolate. If None, interpolate all gaps.
                 Gaps larger than max_gap remain as NaN in output.
        interp_method: Specified interpolation method. Defaults to pchip.
                 Alternatively user can specify "cubic_spline".
    """
    data = np.asarray(data, dtype=float)

    # Fast path: no NaNs
    if not np.any(np.isnan(data)):
        return apply_bandpass(data, sampling_rate, lowcut, highcut, order)

    # Interpolate gaps
    interpolated, original_nan_mask = interpolate_nan_gaps(data, method=interp_method, max_gap=max_gap)

    # Determine which NaNs were NOT filled (large gaps when max_gap is set)
    still_nan = np.isnan(interpolated)

    if np.any(still_nan):
        # Some gaps weren't filled - filter valid segments only
        min_len = min_viable_length_sosfiltfilt(sampling_rate, lowcut, highcut, order)["min_sequence_length"]
        result = np.full_like(data, np.nan)

        for start, end, segment in iter_nan_islands(interpolated):
            if len(segment) >= min_len:
                result[start:end] = apply_bandpass(segment, sampling_rate, lowcut, highcut, order)
    else:
        # All gaps filled - filter entire signal
        result = apply_bandpass(interpolated, sampling_rate, lowcut, highcut, order)

    # Restore original NaN positions
    # result[original_nan_mask] = np.nan

    return result


# Strictly needs path_names (raw files), and sampling rate
# optional is upper and lower frequency for bandpass filter
def bandpass_filter_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    passband: str | tuple = 'default',
    order: int = 2,
    max_gap: int | None = None,
    interp_method: str = "pchip"
) -> None:
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
    max_gap : int, optional
        Maximum gap size (in samples) to interpolate over. If None, all NaN gaps
        are interpolated before filtering. Gaps larger than max_gap remain as NaN
        in the output. (default: None)
    interp_method : Specified interpolation method. Defaults to pchip.
        Alternatively user can specify "cubic_spline".
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
                df[column] = apply_bandpass_nan_safe(df[column].values, sampling_rate, lowcut, highcut, order, max_gap, interp_method=interp_method)

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
# =============================================================================
#

#
# =============================================================================
#

