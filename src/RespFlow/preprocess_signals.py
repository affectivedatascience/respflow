from RespFlow.access_files import map_files
from scipy.signal import butter, sosfiltfilt
import pandas as pd
from pathlib import Path
from scipy.ndimage import median_filter
from scipy.ndimage import binary_closing
import numpy as np
from dataclasses import dataclass

#
# =============================================================================
#

"""
A collection of functions for preprocessing signals.
"""

#
# HARD FAULT
# =============================================================================
#

@dataclass
class HardFaultConfig:
    """
    Hard-fault detection parameters.

    Hard faults are unambiguous sensor/data failures that should be masked
    before detrend, micro-gap fill, and bandpass filtering.
    """
    # Flatline / stuck sensor
    flat_min_s: float = 1.0              # minimum duration to qualify as flatline
    flat_sensitivity: float = 0.05       # multiplier on MAD(dx) for flatline threshold

    # Clipping / saturation (data-driven rails)
    clip_percentile: float = 0.1         # lower percentile for rail detection (upper = 100 - this)
    clip_min_run_s: float = 0.25         # minimum clipping run duration (seconds)

    # Step/discontinuity spikes
    step_sensitivity: float = 12.0       # multiplier on MAD(dx) for step threshold
    step_pad_s: float = 0.05             # pad around steps (seconds)
    step_verify_window_s: float = 0.5    # window (seconds) to check sustained level shift

    # Optional: pad around any hard fault
    fault_pad_s: float = 0.0             # additional dilation of final hard-fault mask (seconds)


def _runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) half-open index pairs for contiguous True-runs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return list(zip(starts.tolist(), ends.tolist()))


def _apply_min_run_length(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True-runs of length >= min_len."""
    mask = np.asarray(mask, dtype=bool)
    if min_len <= 1:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for a, b in _runs_from_mask(mask):
        if (b - a) >= min_len:
            out[a:b] = True
    return out


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask by +/- radius samples using convolution."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or mask.size == 0:
        return mask
    kernel = np.ones(2 * radius + 1, dtype=int)
    return np.convolve(mask.astype(int), kernel, mode="same") > 0


def _robust_mad(x: np.ndarray) -> float:
    """NaN-safe median absolute deviation (MAD)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def apply_hard_fault(
    signal: np.ndarray,
    sampling_rate: int,
    config: HardFaultConfig | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Detect hard faults and set them to NaN.

    Returns:
        signal_out: signal with hard-fault samples set to NaN
        info: dict with masks and thresholds used
    """
    x = np.asarray(signal, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("apply_hard_fault expects a 1D signal.")

    N = x.size
    fs = float(sampling_rate)
    if config is None:
        config = HardFaultConfig()

    mask_nan = np.isnan(x)

    # Early return for very short or fully-missing signals
    if N < 3 or np.all(mask_nan):
        mask_hardfault = mask_nan.copy()
        x[mask_hardfault] = np.nan
        info = {
            "mask_nan": mask_nan,
            "mask_flatline": np.zeros(N, dtype=bool),
            "mask_clip": np.zeros(N, dtype=bool),
            "mask_step": np.zeros(N, dtype=bool),
            "mask_hardfault": mask_hardfault,
            "runs_hardfault": _runs_from_mask(mask_hardfault),
        }
        return x, info

    # Robust scales (computed on available data)
    mad_x = _robust_mad(x)
    dx = np.diff(x)  # NaNs propagate into dx where adjacent samples include NaN
    mad_dx = _robust_mad(dx)

    if not np.isfinite(mad_x) or mad_x <= 0:
        mad_x = np.finfo(float).eps
    if not np.isfinite(mad_dx) or mad_dx <= 0:
        mad_dx = np.finfo(float).eps

    # -------------------------------------------------------------------------
    # Flatline / stuck sensor (near-zero first differences sustained)
    # -------------------------------------------------------------------------
    flat_eps = max(
        config.flat_sensitivity * mad_dx,
        1e-4 * mad_x,       # safety floor: fraction of signal scale
        np.finfo(float).eps,
    )

    dx_abs = np.abs(dx)
    mask_dx_small = (dx_abs <= flat_eps) & ~np.isnan(dx)

    min_flat_samples = int(np.ceil(config.flat_min_s * fs))
    mask_flatline = np.zeros(N, dtype=bool)

    # A run of small dx from [a, b) implies constant samples [a, b+1)
    for run_start, run_end in _runs_from_mask(mask_dx_small):
        sample_start, sample_end = run_start, min(N, run_end + 1)
        if (sample_end - sample_start) >= min_flat_samples:
            mask_flatline[sample_start:sample_end] = True

    mask_flatline &= ~mask_nan

    # -------------------------------------------------------------------------
    # Clipping / saturation (runs near inferred rails)
    # -------------------------------------------------------------------------
    valid_signal = x[~mask_nan]
    rail_low = np.percentile(valid_signal, config.clip_percentile)
    rail_high = np.percentile(valid_signal, 100 - config.clip_percentile)
    clip_tol = 0.01 * mad_x

    mask_clip_raw = (~mask_nan) & ((x <= rail_low + clip_tol) | (x >= rail_high - clip_tol))
    min_clip_samples = int(np.ceil(config.clip_min_run_s * fs))
    mask_clip = _apply_min_run_length(mask_clip_raw, min_clip_samples)

    # -------------------------------------------------------------------------
    # Step/discontinuity spikes (robust threshold on dx)
    # -------------------------------------------------------------------------
    valid_dx = dx[~np.isnan(dx)]
    dx_med = float(np.median(valid_dx)) if valid_dx.size else 0.0

    # Floor the threshold so it never collapses for smooth signals
    step_threshold = max(config.step_sensitivity * mad_dx, 0.5 * mad_x)

    step_candidates = np.where(np.abs(dx - dx_med) > step_threshold)[0]
    mask_step = np.zeros(N, dtype=bool)
    step_pad = int(np.ceil(config.step_pad_s * fs))

    verify_win = int(np.ceil(config.step_verify_window_s * fs))
    min_shift = 0.3 * mad_x
    # Massive spikes (3x threshold) bypass verification
    spike_bypass_thr = 3.0 * step_threshold

    for i in step_candidates:
        dx_mag = abs(float(dx[i]) - dx_med)

        # Massive spike — flag unconditionally, no verification needed
        if dx_mag >= spike_bypass_thr:
            mask_start = max(0, i - step_pad)
            mask_end = min(N, i + 2 + step_pad)
            mask_step[mask_start:mask_end] = True
            continue

        # Moderate spike — verify sustained level shift
        before_start = max(0, i - verify_win)
        after_end = min(N, i + 2 + verify_win)
        seg_before = x[before_start:i]
        seg_after = x[i + 1:after_end]

        seg_before = seg_before[~np.isnan(seg_before)]
        seg_after = seg_after[~np.isnan(seg_after)]

        if seg_before.size == 0 or seg_after.size == 0:
            continue

        shift = abs(float(np.median(seg_after)) - float(np.median(seg_before)))
        if shift < min_shift:
            continue

        mask_start = max(0, i - step_pad)
        mask_end = min(N, i + 2 + step_pad)
        mask_step[mask_start:mask_end] = True

    mask_step &= ~mask_nan

    # -------------------------------------------------------------------------
    # Combine and pad
    # -------------------------------------------------------------------------
    mask_hardfault = mask_nan | mask_flatline | mask_clip | mask_step

    if config.fault_pad_s and config.fault_pad_s > 0:
        fault_pad = int(np.ceil(config.fault_pad_s * fs))
        mask_hardfault = _dilate_mask(mask_hardfault, fault_pad)

    x[mask_hardfault] = np.nan

    info: dict[str, object] = {
        "mask_nan": mask_nan,
        "mask_flatline": mask_flatline,
        "mask_clip": mask_clip,
        "mask_step": mask_step,
        "mask_hardfault": mask_hardfault,
        "runs_hardfault": _runs_from_mask(mask_hardfault),
        "mad_x": mad_x,
        "mad_dx": mad_dx,
        "flat_eps": flat_eps,
        "rail_low": rail_low,
        "rail_high": rail_high,
        "clip_tol": clip_tol,
        "step_threshold": step_threshold,
    }
    return x, info


def apply_hard_fault_to_df(
    df: pd.DataFrame,
    sampling_rate: int,
    config: HardFaultConfig | None = None,
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Apply hard-fault detection to all non-time columns of a dataframe.
    Replaces hard-fault samples with NaN in each signal column.
    """
    out = df.copy()

    for column in out.columns:
        if column.lower() == time_col.lower():
            continue

        x = out[column].to_numpy(dtype=float)
        x_hf, _info = apply_hard_fault(x, sampling_rate, config=config)
        out[column] = x_hf

    return out


def hard_fault_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    config: HardFaultConfig | None = None,
) -> None:
    """
    Applies hard-fault detection to all columns except 'time' in all CSV files.
    Preserves folder structure from in_path to out_path.

    Parameters
    ----------
    in_path : str
        Input directory path
    out_path : str
        Output directory path
    sampling_rate : int
        Sampling rate in Hz
    config : HardFaultConfig, optional
        Detection configuration. Uses defaults if None.
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        df2 = apply_hard_fault_to_df(df, sampling_rate, config=config)

        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        df2.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

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
                # Commented out for now. Don't need to keep track of this.
                # df[f"{column}_baseline"] = baseline 

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
# MICRO INTERP
# =============================================================================
#

# Physiological constant: typical resting respiratory rate
RESTING_RR_HZ = 0.25  # 0.25 Hz ≈ 15 breaths/min (upper bound for resting adults)


def default_max_gap(sampling_rate: int, percentage_fill = 0.3, resting_rate: float = RESTING_RR_HZ) -> int:
    """
    Compute a default max_gap (in samples) for NaN micro gap interpolation.

    Rule: 30% of one respiratory cycle length.
    At 2000 Hz, 0.25 Hz: 0.3 * (2000 / 0.25) = 2400 samples.
    """
    return int(round(percentage_fill * sampling_rate / resting_rate))

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

    # Anchor edge gaps to 0 so the interpolator ramps smoothly
    # instead of extrapolating wildly.
    if valid_idx[0] != 0 and fill_mask[0]:
        valid_idx = np.insert(valid_idx, 0, 0)
        valid_vals = np.insert(valid_vals, 0, 0.0)
    if valid_idx[-1] != len(data) - 1 and fill_mask[-1]:
        valid_idx = np.append(valid_idx, len(data) - 1)
        valid_vals = np.append(valid_vals, 0.0)

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


def apply_micro_interp(
    signal: np.ndarray,
    sampling_rate: int,
    interp_method: str = "pchip",
    percentage_fill: float = 0.3
) -> np.ndarray:
    """
    Interpolate small NaN gaps in a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal (may contain NaN).
    sampling_rate : int
        Sampling rate in Hz.
    interp_method : str
        Interpolation method: "pchip" (default) or "cubic_spline".
    percentage_fill : float
        Fraction of one breath cycle to fill. Default 0.3 (30%).

    Returns
    -------
    np.ndarray
        Signal with small NaN gaps filled via interpolation.
    """
    signal = np.asarray(signal, dtype=float)
    max_gap = default_max_gap(sampling_rate, percentage_fill=percentage_fill)
    filled, _nan_mask = interpolate_nan_gaps(signal, method=interp_method, max_gap=max_gap)
    return filled


def micro_interp_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    interp_method: str = "pchip",
    percentage_fill: float = 0.3
) -> None:
    """
    Interpolate small NaN gaps in all columns except 'time' in all CSV files.
    Preserves folder structure from in_path to out_path.

    Parameters
    ----------
    in_path : str
        Input directory path
    out_path : str
        Output directory path
    sampling_rate : int
        Sampling rate in Hz
    interp_method : str, optional
        Interpolation method: "pchip" (default) or "cubic_spline".
    percentage_fill : float, optional
        Fraction of one breath cycle to fill. Default 0.3 (30%).
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        for column in df.columns:
            if column.lower() != 'time':
                df[column] = apply_micro_interp(df[column].values, sampling_rate, interp_method, percentage_fill)

        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

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
) -> np.ndarray:
    """
    NaN-safe bandpass filter.

    If the signal has no NaNs, filters directly. If NaNs remain (e.g. large
    unfilled gaps), filters each contiguous non-NaN island separately.

    Parameters:
        data: Input signal (may contain NaN)
        sampling_rate, lowcut, highcut, order: Filter parameters
    """
    data = np.asarray(data, dtype=float)

    # Fast path: no NaNs
    if not np.any(np.isnan(data)):
        return apply_bandpass(data, sampling_rate, lowcut, highcut, order)

    # NaNs present — filter each non-NaN island separately
    min_len = min_viable_length_sosfiltfilt(sampling_rate, lowcut, highcut, order)["min_sequence_length"]
    result = np.full_like(data, np.nan)

    for start, end, segment in iter_nan_islands(data):
        if len(segment) >= min_len:
            result[start:end] = apply_bandpass(segment, sampling_rate, lowcut, highcut, order)

    return result


# Strictly needs path_names (raw files), and sampling rate
# optional is upper and lower frequency for bandpass filter
def bandpass_filter_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    passband: str | tuple = 'default',
    order: int = 2,
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
        Default: 'default'
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
                df[column] = apply_bandpass_nan_safe(df[column].values, sampling_rate, lowcut, highcut, order)

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

def detect_anomalies_iqr(signal_series, window_size=60_000):
    # Calculate rolling Q1 (25th percentile) and Q3 (75th percentile)
    rolling_q1 = signal_series.rolling(window=window_size, center=True, min_periods=1).quantile(0.25)
    rolling_q3 = signal_series.rolling(window=window_size, center=True, min_periods=1).quantile(0.75)
    
    # Calculate rolling IQR
    rolling_iqr = rolling_q3 - rolling_q1
    
    # Define dynamic upper and lower bounds
    lower_bound = rolling_q1 - (1.5 * rolling_iqr)
    upper_bound = rolling_q3 + (1.5 * rolling_iqr)
    
    # Flag points outside the bounds
    anomalies = (signal_series < lower_bound) | (signal_series > upper_bound)
    
    # Fill NaN values created by the rolling window at the edges
    return anomalies.fillna(False)


def detect_anomalies_zscore(signal_series, window_size=60_000, z_threshold=2.0):
    # Calculate rolling mean and standard deviation
    rolling_mean = signal_series.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = signal_series.rolling(window=window_size, center=True, min_periods=1).std()
    
    # Calculate the Z-score for each point
    z_scores = (signal_series - rolling_mean) / rolling_std
    
    # Flag points where the absolute Z-score exceeds the threshold
    anomalies = np.abs(z_scores) > z_threshold
    
    # Fill NaN values at the edges
    return anomalies.fillna(False)


def detect_anomalies_energy_ratio(signal_series, short_window=8_000, long_window=60_000,
                                   upper_ratio=3.0, lower_ratio=0.1):
    """
    Flags regions where local energy deviates from the longer-term baseline.

    Compares short-term rolling variance to long-term rolling variance.
    ratio >> 1  →  local burst  (motion, cough, artifact)
    ratio << 1  →  local quiescence  (apnea, signal dropout)

    Parameters
    ----------
    signal_series : pd.Series
    short_window : int
        Short-term variance window in samples.  Must span at least ~2 breath
        cycles so that normal sinusoidal oscillation averages out.
        Default 8000 = 4 s at 2000 Hz (~2 breaths at 0.25 Hz).
    long_window : int
        Long-term variance window in samples (default 60000 = 30 s at 2000 Hz).
    upper_ratio : float
        Flag where ratio exceeds this (energy burst). Default 3.0.
    lower_ratio : float
        Flag where ratio falls below this (energy drop). Default 0.1.
    """
    short_var = signal_series.rolling(window=short_window, center=True, min_periods=1).var()
    long_var = signal_series.rolling(window=long_window, center=True, min_periods=1).var()

    # Avoid division by zero — where long_var is ~0 the signal is essentially
    # flatlined, which is itself anomalous
    ratio = short_var / long_var.replace(0, np.nan)

    anomalies = (ratio > upper_ratio) | (ratio < lower_ratio)
    return anomalies.fillna(False)


def detect_anomalies_ensemble(signal_series, window_size=60_000, min_votes=2):
    """
    Combines IQR, Z-Score, and Energy-Ratio methods to find anomalies.
    Requires at least 'min_votes' methods to flag a point as True.
    """
    iqr_flags = detect_anomalies_iqr(signal_series, window_size=window_size)
    zscore_flags = detect_anomalies_zscore(signal_series, window_size=window_size)
    energy_flags = detect_anomalies_energy_ratio(signal_series, long_window=window_size)

    total_votes = iqr_flags.astype(int) + zscore_flags.astype(int) + energy_flags.astype(int)

    ensemble_anomalies = total_votes >= min_votes
    return ensemble_anomalies   

def merge_close_anomalies(anomaly_mask, max_gap_samples=6000):
    """
    Merges anomaly blocks that are separated by fewer than 'max_gap_samples'.
    """
    # Create a structural element of ones (the size of the allowed gap)
    structure = np.ones(max_gap_samples)

    # Run the closing operation
    # It will turn [True, False, False, True] into [True, True, True, True]
    merged_mask = binary_closing(anomaly_mask, structure=structure)

    return merged_mask


def pad_anomaly_mask(anomaly_mask, pad_factor=2.0):
    """
    Expands each contiguous anomaly region by a fraction of its own length.

    For each block of True values of length L, adds (pad_factor * L / 2)
    samples on each side, clamped to the array bounds.

    Parameters
    ----------
    anomaly_mask : array-like of bool
        Boolean mask where True indicates an anomalous sample.
    pad_factor : float
        Total padding as a multiple of the anomaly length, split equally
        between both sides. Default 2.0 means each side gets 1x the anomaly
        length (so the padded region is 3x the original).

    Returns
    -------
    padded : np.ndarray of bool
    """
    mask = np.asarray(anomaly_mask, dtype=bool)
    padded = mask.copy()
    n = len(mask)

    # Find starts and ends of contiguous True runs
    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        length = end - start
        pad = int(length * pad_factor / 2)
        padded[max(0, start - pad): min(n, end + pad)] = True

    return padded


def detect_anomalies(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    window_size_seconds: float = 30,
    min_votes: int = 2,
    merge_gap_seconds: float = 0,
    pad_factor: float = 2.0,
) -> None:
    """
    Detects anomalies using an ensemble of IQR, Z-Score, and Energy-Ratio
    methods, and replaces flagged points with NaN.

    Parameters
    ----------
    in_path : str
        Input directory path containing CSV files.
    out_path : str
        Output directory path for anomaly-screened CSV files.
    sampling_rate : int
        Sampling rate in Hz.
    window_size_seconds : float, optional
        Rolling window size in seconds for detection methods (default: 30).
    min_votes : int, optional
        Minimum number of methods that must flag a point (default: 2).
    merge_gap_seconds : float, optional
        Maximum gap in seconds between anomaly blocks to merge (default: 0, no merging).
    pad_factor : float, optional
        Padding around each anomaly as a multiple of its length, split equally
        on both sides. Default 2.0 means each side is padded by 1x the anomaly
        length. Set to 0 to disable padding.
    """
    window_samples = int(window_size_seconds * sampling_rate)
    merge_gap_samples = int(merge_gap_seconds * sampling_rate)

    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        for column in df.columns:
            if column.lower() != 'time':
                mask = detect_anomalies_ensemble(
                    df[column],
                    window_size=window_samples,
                    min_votes=min_votes,
                )

                if merge_gap_samples > 0:
                    mask = merge_close_anomalies(mask, max_gap_samples=merge_gap_samples)

                if pad_factor > 0:
                    mask = pad_anomaly_mask(mask, pad_factor=pad_factor)

                df[f'{column}_anomaly'] = mask
                df.loc[mask, column] = np.nan

        # Preserve folder structure
        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# POST ANOMALY MICRO INTERP
# =============================================================================
#

def post_anomaly_interp_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    interp_method: str = "pchip",
    percentage_fill: float = 0.5
) -> None:
    """
    Post-anomaly interpolation: fills larger NaN gaps (default 50% of one
    breath cycle) after anomaly detection has NaN-ed out bad regions.

    Parameters
    ----------
    in_path : str
        Input directory path (typically the anomaly output).
    out_path : str
        Output directory path.
    sampling_rate : int
        Sampling rate in Hz.
    interp_method : str, optional
        Interpolation method: "pchip" (default) or "cubic_spline".
    percentage_fill : float, optional
        Fraction of one breath cycle to fill. Default 0.5 (50%).
    """
    mapped_files = map_files(in_path, file_ext='csv')
    max_gap = default_max_gap(sampling_rate, percentage_fill=percentage_fill)

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        data_columns = [
            c for c in df.columns
            if c.lower() != 'time' and not c.endswith('_anomaly')
        ]

        for column in data_columns:
            anomaly_col = f'{column}_anomaly'
            if anomaly_col not in df.columns:
                continue

            signal = df[column].values.copy()
            anomaly_mask = df[anomaly_col].values.astype(bool)
            nan_mask = np.isnan(signal)
            non_anomaly_nans = nan_mask & ~anomaly_mask

            # Temporarily fill non-anomaly NaNs via linear interp so they
            # don't distort the curve but aren't seen as gaps.
            if np.any(non_anomaly_nans):
                valid = np.where(~nan_mask)[0]
                if len(valid) >= 2:
                    signal[non_anomaly_nans] = np.interp(
                        np.where(non_anomaly_nans)[0], valid, signal[valid]
                    )

            filled, _ = interpolate_nan_gaps(signal, method=interp_method, max_gap=max_gap)

            # Restore non-anomaly NaNs
            filled[non_anomaly_nans] = np.nan
            df[column] = filled

        # Drop anomaly mask columns
        anomaly_cols = [c for c in df.columns if c.endswith('_anomaly')]
        df.drop(columns=anomaly_cols, inplace=True)

        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# =============================================================================
#