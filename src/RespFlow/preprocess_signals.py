from RespFlow.access_files import map_files
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.ndimage import binary_closing, gaussian_filter1d
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

#
# =============================================================================
#

"""
A collection of functions for preprocessing signals and EMG data.
"""

#
# =============================================================================
#
#
# HARD FAULT
#
#
# =============================================================================
#

@dataclass
class HardFaultConfig:
    """
    Configuration for hard-fault detection.

    Hard faults are unambiguous sensor/data failures (flatlines, clipping,
    step discontinuities) that should be masked before downstream processing.

    Parameters
    ----------
    flat_min_s : float
        Minimum duration (seconds) for a run to qualify as a flatline. Default 0.5.
    flat_sensitivity : float
        Multiplier on MAD(dx) to set the flatline threshold. Default 0.05.
    clip_percentile : float
        Lower percentile for rail detection; upper rail is 100 minus this value.
        Default 0.1.
    clip_min_run_s : float
        Minimum clipping run duration in seconds. Default 0.25.
    step_sensitivity : float
        Multiplier on MAD(dx) for the step-discontinuity threshold. Default 12.0.
    step_pad_s : float
        Padding (seconds) applied around detected steps. Default 0.05.
    step_verify_window_s : float
        Window (seconds) used to verify a sustained level shift. Default 0.5.
    fault_pad_s : float
        Padding (seconds) added before AND after each detected fault region.
        A value of 1.0 adds 1 second on each side. Default 0.0.
    """
    flat_min_s: float = 0.5
    flat_sensitivity: float = 0.05
    clip_percentile: float = 0.1
    clip_min_run_s: float = 0.25
    step_sensitivity: float = 12.0
    step_pad_s: float = 0.05
    step_verify_window_s: float = 0.5
    fault_pad_s: float = 0.0


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
    """
    Dilate a boolean mask by radius samples on each side 
    (so a radius of 1s pads 1s before AND 1s after each flagged region) using convolution.
    """
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


def _apply_hard_fault(
    signal: np.ndarray,
    sampling_rate: int,
    config: HardFaultConfig | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Detect hard faults in a 1-D signal and replace them with NaN.

    Returns:
        signal_out: signal with hard-fault samples set to NaN
        info: dict with masks and thresholds used (useful for debugging but not passed onto user)
    """
    x = np.asarray(signal, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("_apply_hard_fault expects a 1D signal.")

    num_samples = x.size
    if config is None:
        config = HardFaultConfig()

    mask_nan = np.isnan(x)

    # Early return for very short or fully-missing signals
    if num_samples < 3 or np.all(mask_nan):
        mask_hardfault = mask_nan.copy()
        x[mask_hardfault] = np.nan
        info = {
            "mask_nan": mask_nan,
            "mask_flatline": np.zeros(num_samples, dtype=bool),
            "mask_clip": np.zeros(num_samples, dtype=bool),
            "mask_step": np.zeros(num_samples, dtype=bool),
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

    min_flat_samples = seconds_to_samples(config.flat_min_s, sampling_rate)
    mask_flatline = np.zeros(num_samples, dtype=bool)

    # A run of small dx from [a, b) implies constant samples [a, b+1)
    for run_start, run_end in _runs_from_mask(mask_dx_small):
        sample_start, sample_end = run_start, min(num_samples, run_end + 1)
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
    min_clip_samples = seconds_to_samples(config.clip_min_run_s, sampling_rate)
    mask_clip = _apply_min_run_length(mask_clip_raw, min_clip_samples)

    # -------------------------------------------------------------------------
    # Step/discontinuity spikes (robust threshold on dx)
    # -------------------------------------------------------------------------
    valid_dx = dx[~np.isnan(dx)]
    dx_med = float(np.median(valid_dx)) if valid_dx.size else 0.0

    # Floor the threshold so it never collapses for smooth signals
    step_threshold = max(config.step_sensitivity * mad_dx, 0.5 * mad_x)

    step_candidates = np.where(np.abs(dx - dx_med) > step_threshold)[0]
    mask_step = np.zeros(num_samples, dtype=bool)
    step_pad = seconds_to_samples(config.step_pad_s, sampling_rate)

    verify_win = seconds_to_samples(config.step_verify_window_s, sampling_rate)
    min_shift = 0.3 * mad_x
    # Massive spikes (3x threshold) bypass verification
    spike_bypass_thr = 3.0 * step_threshold

    for i in step_candidates:
        dx_mag = abs(float(dx[i]) - dx_med)

        # Massive spike — flag unconditionally, no verification needed
        if dx_mag >= spike_bypass_thr:
            mask_start = max(0, i - step_pad)
            mask_end = min(num_samples, i + 2 + step_pad)
            mask_step[mask_start:mask_end] = True
            continue

        # Moderate spike — verify sustained level shift
        before_start = max(0, i - verify_win)
        after_end = min(num_samples, i + 2 + verify_win)
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
        mask_end = min(num_samples, i + 2 + step_pad)
        mask_step[mask_start:mask_end] = True

    mask_step &= ~mask_nan

    # -------------------------------------------------------------------------
    # Combine and pad
    # -------------------------------------------------------------------------
    mask_hardfault = mask_nan | mask_flatline | mask_clip | mask_step

    if config.fault_pad_s and config.fault_pad_s > 0:
        fault_pad = seconds_to_samples(config.fault_pad_s, sampling_rate)
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


def _apply_hard_fault_to_df(
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
        x_hf, _info = _apply_hard_fault(x, sampling_rate, config=config)
        out[column] = x_hf

    return out


def hard_fault_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    config: HardFaultConfig | None = None,
    columns: list[str] | None = None,
) -> None:
    """
    Applies hard-fault detection to selected columns in all CSV files.
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
    columns : list[str], optional
        Column names to process. Only these columns (plus the time column)
        will be kept in the output. Defaults to all non-time columns.
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        # Find the time column (case-insensitive)
        time_col = next((c for c in df.columns if c.lower() == 'time'), None)

        if columns is not None:
            # Validate requested columns exist
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Columns {missing} not found in {file_path}. "
                    f"Available: {list(df.columns)}"
                )

            # Filter to time + requested columns
            keep = ([time_col] if time_col else []) + columns
            df = df[keep]

        df2 = _apply_hard_fault_to_df(df, sampling_rate, config=config)

        relative_path = Path(file_path).relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        df2.to_csv(output_file_path, index=False)
        
    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# =============================================================================
#
#
# MICRO INTERP
#
#
# =============================================================================
#

def _default_max_gap(
    sampling_rate: int,
    percentage_fill: float = 0.3,
    breath_rate_hz: float = 0.25,
) -> int:
    """
    Compute the max gap (samples) as a fraction of one respiratory cycle.
    
    Example: at 2000 Hz with a 0.25 Hz breath rate, 
    0.3 * (2000 / 0.25) = 2400 samples.
    """
    return int(round(percentage_fill * sampling_rate / breath_rate_hz))


def _nan_gap_indices(x: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Return (start, end, length) for each contiguous NaN gap in a 1D array.
    Indices are half-open: x[start:end] are all NaN.
    """
    x = np.asarray(x)
    is_nan = np.isnan(x)

    if not np.any(is_nan):
        return []

    # diff produces +1 where a NaN run starts and -1 where it ends
    d = np.diff(is_nan.astype(np.int8))
    starts = np.where(d == 1)[0] + 1   # +1 because diff shifts indices left by 1
    ends = np.where(d == -1)[0] + 1

    # diff misses runs that touch the edges, so handle those manually
    if is_nan[0]:
        starts = np.r_[0, starts]
    if is_nan[-1]:
        ends = np.r_[ends, len(x)]

    return [(s, e, e - s) for s, e in zip(starts.tolist(), ends.tolist())]


def _interpolate_nan_gaps(
    data: np.ndarray,
    method: str,
    max_gap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate NaN gaps in a 1-D array.

    Returns (filled_data, original_nan_mask). Only gaps with length <= max_gap
    are filled when max_gap is provided.
    """

    data = np.asarray(data, dtype=float)
    nan_mask = np.isnan(data)

    if not np.any(nan_mask):
        return data.copy(), nan_mask

    result = data.copy()
    gaps = _nan_gap_indices(data)

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
        raise ValueError(f"Unknown interpolation method: {method!r}")


    # Fill only the gaps we want to fill
    fill_idx = np.where(fill_mask)[0]
    result[fill_idx] = interp(fill_idx)

    return result, nan_mask


def _apply_micro_interp(
    signal: np.ndarray,
    sampling_rate: int,
    interp_method: str = "pchip",
    percentage_fill: float = 0.3,
    breath_rate_hz: float = 0.25,
) -> np.ndarray:
    """Interpolate small NaN gaps in a 1-D signal using the configured max gap."""
    signal = np.asarray(signal, dtype=float)
    max_gap = _default_max_gap(sampling_rate, percentage_fill=percentage_fill, breath_rate_hz=breath_rate_hz)
    
    # mask not needed here
    filled, _ = _interpolate_nan_gaps(signal, method=interp_method, max_gap=max_gap)
    return filled


def micro_interp_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    interp_method: str = "pchip",
    percentage_fill: float = 0.3,
    breath_rate_hz: float = 0.25,
) -> None:
    """
    Interpolate small NaN gaps in all signal columns across all CSV files.

    Gaps shorter than a fraction of one respiratory cycle are filled via
    interpolation. Folder structure from ``in_path`` is preserved in ``out_path``.

    Parameters
    ----------
    in_path : str
        Input directory path containing CSV files.
    out_path : str
        Output directory path for interpolated CSV files.
    sampling_rate : int
        Sampling rate in Hz.
    interp_method : str, optional
        Interpolation method: ``"pchip"`` (default) or ``"cubic_spline"``.
    percentage_fill : float, optional
        Fraction of one breath cycle used as the max gap to fill. Default 0.3.
    breath_rate_hz : float, optional
        Expected breathing rate in Hz. Default 0.25 (15 breaths/min).
        This is approx adult resting breathing rate.
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        for column in df.columns:
            if column.lower() != 'time':
                df[column] = _apply_micro_interp(
                    df[column].values, sampling_rate,
                    interp_method=interp_method,
                    percentage_fill=percentage_fill,
                    breath_rate_hz=breath_rate_hz,
                )

        relative_path = Path(file_path).relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# =============================================================================
#
#
# DETREND
#
#
# =============================================================================
#

def _apply_detrend(
    signal: np.ndarray,
    sampling_rate: int,
    window_size_seconds: int = 60, # 60-second window default based on BreathMetrics paper
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a rolling-median baseline from a 1-D signal."""
    num_samples = len(signal)
    signal_duration = num_samples / sampling_rate

    # if signal is shorter than window, use global median
    if signal_duration >= window_size_seconds:
        window_samples = seconds_to_samples(window_size_seconds, sampling_rate)
        
        # rolling with center=True requires an odd window to be truly centered
        if window_samples % 2 == 0: 
            window_samples += 1

        # Guard: do not use a window longer than the signal
        if window_samples <= num_samples:
            s = pd.Series(signal)

            # NaN-safe centered rolling median baseline
            # Require at least half the window to be filled before computing a median,
            # so edge estimates aren't based on just a handful of samples. The max(1, ...)
            # ensures min_periods is never 0 for very small windows.
            baseline = (
                s.rolling(window=window_samples, center=True, min_periods=max(1, window_samples // 2))
                .median()
            )

            # If baseline has NaNs (edges or long NaN runs), fill from nearest valid values
            baseline = baseline.ffill().bfill().to_numpy()
        else:
            baseline = np.full(num_samples, np.nanmedian(signal))
    else:
        baseline = np.full(num_samples, np.nanmedian(signal))

    # Detrend
    detrended_signal = signal - baseline

    return detrended_signal, baseline


def detrend_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    window_size_seconds: int = 60,
) -> None:
    """
    Apply rolling-median detrending to all signal columns in all CSV files.

    Subtracts a rolling-median baseline from each non-time column and writes
    the detrended data to ``out_path``, preserving folder structure.

    Parameters
    ----------
    in_path : str
        Input directory path containing CSV files.
    out_path : str
        Output directory path for detrended CSV files.
    sampling_rate : int
        Sampling rate in Hz.
    window_size_seconds : int, optional
        Window size for the rolling median in seconds. Default 60.
        If a signal is shorter than this window, a global median is used instead.
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        for column in df.columns:
            if column.lower() != 'time':
                detrended, _ = _apply_detrend(df[column].values, sampling_rate, window_size_seconds)
                df[column] = detrended

        relative_path = Path(file_path).relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file_path, index=False)
    
    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# =============================================================================
#
#
# BANDPASS
#
#
# =============================================================================
#

def _apply_bandpass(
    signal: np.ndarray,
    sampling_rate: int,
    lowcut: float = 0.05,
    highcut: float = 2.0,
    order: int = 2,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to a 1-D signal."""
    nyquist = 0.5 * sampling_rate
    
    # normalise to [0, 1] as required by butter
    low = lowcut / nyquist 
    high = highcut / nyquist

    # Design filter
    sos = butter(order, [low, high], btype='band', output='sos')

    # Apply zero-phase filter (sosfiltfilt) with padlen adjusted for short signals
    padlen = min(len(signal) - 1, 15)
    y = sosfiltfilt(sos, signal, padlen=padlen)

    return y


def _min_viable_length_sosfiltfilt(
    sampling_rate: int,
    lowcut: float = 0.05,
    highcut: float = 2.0,
    order: int = 2,
) -> dict:
    """
    Compute SciPy's default ``padlen`` for a Butterworth bandpass and the
    minimum segment length ``N_min`` such that ``padlen < N - 1``.

    Return dict keys: ``n_sections``, ``padlen_default``, ``min_sequence_length``.
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


def _nan_islands(x: np.ndarray) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs for contiguous non-NaN regions ("islands")
    in a 1D array x. Indices are half-open: [start, end).

    Example: if x[10:25] are non-NaN, returns (10, 25).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("_nan_islands expects a 1D array")

    valid = ~np.isnan(x)

    # Find rising edges (False->True) and falling edges (True->False)
    d = np.diff(valid.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1

    # Handle island starting at index 0
    if valid[0]:
        starts = np.r_[0, starts]
    
    # Handle island ending at last index
    if valid[-1]:
        ends = np.r_[ends, len(x)]

    return list(zip(starts.tolist(), ends.tolist()))


def _iter_nan_islands(x: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(start, end, segment)`` for each contiguous non-NaN run in a 1-D array."""
    x = np.asarray(x)
    for start, end in _nan_islands(x):
        yield start, end, x[start:end]


def _apply_bandpass_nan_safe(
    signal: np.ndarray,
    sampling_rate: int,
    lowcut: float = 0.05,
    highcut: float = 2.0,
    order: int = 2,
) -> np.ndarray:
    """
    Bandpass filter a signal that may contain NaNs.

    If no NaNs are present, filters directly. Otherwise filters each contiguous
    non-NaN island separately and leaves NaN regions untouched.
    """
    signal = np.asarray(signal, dtype=float)

    # Fast path: no NaNs
    if not np.any(np.isnan(signal)):
        return _apply_bandpass(signal, sampling_rate, lowcut, highcut, order)

    # NaNs present — filter each non-NaN island separately
    min_len = _min_viable_length_sosfiltfilt(sampling_rate, lowcut, highcut, order)["min_sequence_length"]
    result = np.full_like(signal, np.nan)

    for start, end, segment in _iter_nan_islands(signal):
        if len(segment) >= min_len:
            result[start:end] = _apply_bandpass(segment, sampling_rate, lowcut, highcut, order)

    return result


def bandpass_filter_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    passband: str | tuple = 'default',
    order: int = 2,
) -> None:
    """
    Apply a Butterworth bandpass filter to all signal columns in all CSV files.

    Filters every non-time column of each CSV under ``in_path`` and writes the
    results to ``out_path``, preserving folder structure. Filtering is NaN-safe:
    contiguous non-NaN runs are filtered independently and NaN regions are left
    untouched.

    Parameters
    ----------
    in_path : str
        Input directory path containing CSV files.
    out_path : str
        Output directory path for bandpass-filtered CSV files.
    sampling_rate : int
        Sampling rate in Hz.
    passband : str or tuple, optional
        Preset name or an explicit ``(lowcut, highcut)`` tuple in Hz. Presets:
        ``'default'`` (0.05-2.0 Hz), ``'resting_adult'`` (0.05-1.0 Hz),
        ``'narrow_band'`` (0.1-0.35 Hz), ``'wide_band'`` (0.05-3.0 Hz).
        Default ``'default'``.
    order : int, optional
        Butterworth filter order. Default 2.

    Returns
    -------
    None
    """
    PASSBANDS = {
        'default': (0.05, 2.0),
        'resting_adult': (0.05, 1),
        'narrow_band': (0.1, 0.35),
        'wide_band': (0.05, 3.0),
    }
    
    # Determine lowcut and highcut from passband argument
    if isinstance(passband, str):
        if passband not in PASSBANDS:
            raise ValueError(f"Unknown passband preset '{passband}'. Available: {list(PASSBANDS.keys())}")
        lowcut, highcut = PASSBANDS[passband]
    elif isinstance(passband, (tuple, list)) and len(passband) == 2:
        lowcut, highcut = passband
    else:
        raise ValueError("passband must be a string preset or a (lowcut, highcut) tuple")

    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        for column in df.columns:
            if column.lower() != 'time':
                df[column] = _apply_bandpass_nan_safe(
                    df[column].values, sampling_rate, lowcut, highcut, order
                )

        relative_path = Path(file_path).relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# =============================================================================
#
#
# ANOMALY DETECTION
#
#
# =============================================================================
#

def _detect_anomalies_iqr(
    signal_series: pd.Series,
    window_size: int = 60_000,
) -> pd.Series:
    """Flag samples outside the rolling IQR (1.5 * IQR) as anomalies."""
    rolling_q1 = signal_series.rolling(window=window_size, center=True, min_periods=1).quantile(0.25)
    rolling_q3 = signal_series.rolling(window=window_size, center=True, min_periods=1).quantile(0.75)
    rolling_iqr = rolling_q3 - rolling_q1

    lower_bound = rolling_q1 - 1.5 * rolling_iqr
    upper_bound = rolling_q3 + 1.5 * rolling_iqr

    anomalies = (signal_series < lower_bound) | (signal_series > upper_bound)
    return anomalies.fillna(False)


def _detect_anomalies_zscore(
    signal_series: pd.Series,
    window_size: int = 60_000,
    z_threshold: float = 2.0,
) -> pd.Series:
    """Flag samples more than ``z_threshold`` standard deviations from the rolling mean."""
    rolling_mean = signal_series.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = signal_series.rolling(window=window_size, center=True, min_periods=1).std()
    
    z_scores = (signal_series - rolling_mean) / rolling_std.replace(0, np.nan)
    
    anomalies = np.abs(z_scores) > z_threshold
    return anomalies.fillna(False)


def _detect_anomalies_energy_ratio(
    signal_series: pd.Series,
    short_window: int = 8_000,
    long_window: int = 60_000,
    upper_ratio: float = 3.0,
    lower_ratio: float = 0.1,
) -> pd.Series:
    """
    Flag samples where the short-to-long rolling variance ratio departs from 1.

    A ratio far above 1 indicates a local energy burst (motion, cough, artifact);
    a ratio far below 1 indicates local quiescence (apnea, signal dropout).
    """
    long_window = max(long_window, short_window + 1)
    short_var = signal_series.rolling(window=short_window, center=True, min_periods=1).var()
    long_var = signal_series.rolling(window=long_window, center=True, min_periods=1).var()

    # Where long_var is ~0 the signal is flatlined, which is itself anomalous.
    ratio = short_var / long_var.replace(0, np.nan)

    anomalies = (ratio > upper_ratio) | (ratio < lower_ratio)
    return anomalies.fillna(False)


def _detect_anomalies_ensemble(
    signal_series: pd.Series,
    window_size: int = 60_000,
    min_votes: int = 2,
    z_threshold: float = 2.0,
) -> np.ndarray:
    """Combine IQR / Z-score / energy-ratio detectors and flag samples with at least ``min_votes``."""
    iqr_flags = _detect_anomalies_iqr(signal_series, window_size=window_size)
    zscore_flags = _detect_anomalies_zscore(signal_series, window_size=window_size, z_threshold=z_threshold)
    energy_flags = _detect_anomalies_energy_ratio(signal_series, long_window=window_size)

    total_votes = iqr_flags.to_numpy(dtype=int) + zscore_flags.to_numpy(dtype=int) + energy_flags.to_numpy(dtype=int)

    ensemble_anomalies = total_votes >= min_votes
    return ensemble_anomalies   

def _merge_close_anomalies(
    anomaly_mask: np.ndarray,
    max_gap_samples: int = 6000,
) -> np.ndarray:
    """Merge anomaly runs separated by fewer than ``max_gap_samples`` samples via binary closing."""
    structure = np.ones(max_gap_samples, dtype=bool)

    # Binary closing bridges anomaly runs separated by fewer than max_gap_samples.
    merged_mask = binary_closing(anomaly_mask, structure=structure)

    return merged_mask


def _pad_anomaly_mask(
    anomaly_mask: np.ndarray,
    pad_samples: int,
) -> np.ndarray:
    """Expand each contiguous anomaly run by ``pad_samples`` on each side, clamped to array bounds."""
    mask = np.asarray(anomaly_mask, dtype=bool)
    padded = mask.copy()
    n = len(mask)

    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        padded[max(0, start - pad_samples): min(n, end + pad_samples)] = True

    return padded


def detect_anomalies(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    window_size_seconds: float = 30,
    min_votes: int = 2,
    z_threshold: float = 2.0,
    merge_gap_seconds: float = 1,
    pad_seconds: float = 2.5,
) -> None:
    """
    Detect anomalies in all signal columns via an ensemble of detectors.

    Runs an IQR / Z-score / energy-ratio ensemble on every non-time column of
    each CSV under ``in_path``. Samples flagged by at least ``min_votes``
    detectors are marked anomalous. Anomaly runs closer than
    ``merge_gap_seconds`` are merged, each run is padded by ``pad_seconds`` on
    each side, and the flagged samples in the original column are replaced with
    NaN. The boolean mask is stored alongside each column as ``<column>_anomaly``.

    Parameters
    ----------
    in_path : str
        Input directory path containing CSV files.
    out_path : str
        Output directory path for anomaly-screened CSV files.
    sampling_rate : int
        Sampling rate in Hz.
    window_size_seconds : float, optional
        Rolling window size in seconds for the detectors. Default 30.
    min_votes : int, optional
        Minimum number of detectors that must flag a sample. Default 2.
    z_threshold : float, optional
        Number of standard deviations from the rolling mean to flag as anomalous. Default 2.0.
    merge_gap_seconds : float, optional
        Maximum gap in seconds between anomaly runs to merge. Default 1.
        Set to 0 to disable merging.
    pad_seconds : float, optional
        Seconds of padding to add on each side of every anomaly run. Default 2.5.
        Set to 0 to disable padding.

    Returns
    -------
    None
    """
    window_samples = seconds_to_samples(window_size_seconds, sampling_rate)
    merge_gap_samples = seconds_to_samples(merge_gap_seconds, sampling_rate)
    pad_samples = seconds_to_samples(pad_seconds, sampling_rate)

    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        for column in df.columns:
            if column.lower() != 'time':
                mask = _detect_anomalies_ensemble(
                    df[column],
                    window_size=window_samples,
                    min_votes=min_votes,
                    z_threshold=z_threshold,
                )

                if merge_gap_samples > 0:
                    mask = _merge_close_anomalies(mask, max_gap_samples=merge_gap_samples)

                if pad_samples > 0:
                    mask = _pad_anomaly_mask(mask, pad_samples=pad_samples)

                df[f'{column}_anomaly'] = mask
                df.loc[mask, column] = np.nan

        relative_path = Path(file_path).relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# =============================================================================
#
#
# POST ANOMALY INTERP
#
#
# =============================================================================
#

def post_anomaly_interp_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    interp_method: str = "pchip",
    percentage_fill: float = 0.5,
    breath_rate_hz: float = 0.25,
) -> None:
    """
    Fill medium-sized NaN gaps introduced by anomaly detection.

    For every signal column that has a matching ``<column>_anomaly`` mask,
    interpolates across anomaly-induced NaN runs up to a maximum gap size set
    to ``percentage_fill`` of one breath cycle. Non-anomaly NaNs (e.g. hard-
    fault gaps that were already present or NaNs present in raw stage) 
    are preserved in the output.

    Parameters
    ----------
    in_path : str
        Input directory path (typically the anomaly-detection output).
    out_path : str
        Output directory path for interpolated CSV files.
    sampling_rate : int
        Sampling rate in Hz.
    interp_method : str, optional
        Interpolation method: ``"pchip"`` (default) or ``"cubic_spline"``.
    percentage_fill : float, optional
        Fraction of one breath cycle that sets the maximum gap to fill.
        Default 0.5 (50% of one cycle).
    breath_rate_hz : float, optional
        Expected breathing rate in Hz. Default 0.25 (15 breaths/min).

    Returns
    -------
    None
    """
    mapped_files = map_files(in_path, file_ext='csv')
    max_gap = _default_max_gap(
        sampling_rate,
        percentage_fill=percentage_fill,
        breath_rate_hz=breath_rate_hz,
    )

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

            filled, _ = _interpolate_nan_gaps(signal, method=interp_method, max_gap=max_gap)

            # Restore non-anomaly NaNs
            filled[non_anomaly_nans] = np.nan
            df[column] = filled

        relative_path = Path(file_path).relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# IMPUTE ANOMALY (CYCLE-SYNTHESIS)
# =============================================================================
#


def _resample_to_length(y: np.ndarray, M: int) -> np.ndarray:
    """Linear resample y (1D) to exactly M points."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.full(M, np.nan)
    if y.size == 1:
        return np.full(M, y[0])
    xi = np.linspace(0.0, 1.0, num=y.size)
    xo = np.linspace(0.0, 1.0, num=M)
    return np.interp(xo, xi, y)


def _raised_cosine_weights(B: int) -> np.ndarray:
    """0..1 raised-cosine ramp of length B."""
    if B <= 0:
        return np.array([], dtype=float)
    t = np.linspace(0.0, 1.0, B)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _pick_finite_average(a: float, b: float) -> float | None:
    """Return the average of a and b if both finite, else whichever is finite, else None."""
    a_ok, b_ok = np.isfinite(a), np.isfinite(b)
    if a_ok and b_ok:
        return 0.5 * (a + b)
    if a_ok:
        return float(a)
    if b_ok:
        return float(b)
    return None


def _extract_clean_context(
    x: np.ndarray, g0: int, g1: int, fs: float, context_s: float
) -> tuple[tuple[int, int, np.ndarray], tuple[int, int, np.ndarray]]:
    """Extract clean signal context on both sides of a gap [g0, g1)."""
    N = len(x)
    L = seconds_to_samples(context_s, fs)
    l0, l1 = max(0, g0 - L), g0
    r0, r1 = g1, min(N, g1 + L)
    xL = x[l0:l1]
    xR = x[r0:r1]
    return (l0, l1, xL), (r0, r1, xR)


def _detect_troughs(x_seg: np.ndarray, fs: float,
                    max_bpm: float = 60.0):
    """Detect troughs in a segment using find_peaks on -x."""
    x_seg = np.asarray(x_seg, dtype=float)
    valid = np.isfinite(x_seg)
    if valid.sum() < 10:
        return np.array([], dtype=int)

    y = x_seg.copy()
    y[~valid] = np.nan

    if np.isnan(y).any():
        s = pd.Series(y).interpolate(limit_direction="both")
        y = s.to_numpy()

    min_period_s = 60.0 / max_bpm
    min_dist = max(1, seconds_to_samples(min_period_s, fs))

    mad = _robust_mad(y)
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(y)
    if not np.isfinite(mad) or mad == 0:
        mad = 1.0

    troughs, _ = find_peaks(-y, distance=min_dist, prominence=0.25 * mad)
    return troughs.astype(int)


def _estimate_period_amp_from_cycles(x_ctx: np.ndarray, troughs: np.ndarray, fs: float):
    """Estimate period (seconds) and amplitude (peak-to-trough) from cycles."""
    if troughs.size < 3:
        return np.nan, np.nan, 0

    dt = np.diff(troughs) / fs
    T = float(np.median(dt))

    amps = []
    for i in range(troughs.size - 1):
        seg = x_ctx[troughs[i]:troughs[i + 1]]
        seg = seg[np.isfinite(seg)]
        if seg.size < 3:
            continue
        amps.append(np.nanmax(seg) - np.nanmin(seg))

    A = float(np.median(amps)) if len(amps) else np.nan
    return T, A, len(amps)


def _build_cycle_template(x_ctx: np.ndarray, troughs: np.ndarray,
                          M: int = 200, n_cycles_use: int = 5):
    """Build a robust median template from up to n_cycles_use clean cycles."""
    if troughs.size < 3:
        return None

    cycles = []
    start_i = max(0, (troughs.size - 1) - n_cycles_use)
    for i in range(start_i, troughs.size - 1):
        a = troughs[i]
        b = troughs[i + 1]
        seg = x_ctx[a:b]
        if np.isfinite(seg).sum() < max(5, (b - a) // 2):
            continue
        seg2 = seg.astype(float)
        seg2 = seg2 - float(np.nanmedian(seg2))
        seg_rs = _resample_to_length(seg2, M)
        if np.isfinite(seg_rs).all():
            cycles.append(seg_rs)

    if not cycles:
        return None

    return np.median(np.vstack(cycles), axis=0)


def _synthesize_from_template(template: np.ndarray, fs: float,
                              g_len: int, T: float,
                              phase0_frac: float, amp: float):
    """Tile a template over a gap of length g_len samples."""
    M = template.size
    if not np.isfinite(T) or T <= 0:
        return None

    tmin, tmax = float(np.min(template)), float(np.max(template))
    tA = tmax - tmin
    if tA <= 0 or not np.isfinite(tA):
        tA = 1.0

    scale = amp / tA if np.isfinite(amp) and amp > 0 else 1.0
    templ_scaled = template * scale

    samples_per_cycle = T * fs
    if samples_per_cycle <= 1:
        return None

    n = np.arange(g_len)
    frac = (phase0_frac + (n / samples_per_cycle)) % 1.0
    idx = frac * (M - 1)
    return np.interp(idx, np.arange(M), templ_scaled)


def _edge_crossfade(x: np.ndarray, y_gap: np.ndarray, g0: int, g1: int,
                    fs: float, blend_s: float):
    """Cross-fade y_gap to observed data at edges."""
    B = seconds_to_samples(blend_s, fs)
    if B <= 0:
        return y_gap

    # left edge
    left_start = max(0, g0 - B)
    xL = x[left_start:g0]
    B_left = B if np.isfinite(xL).sum() >= max(3, B // 2) else int(np.isfinite(xL).sum())

    # right edge
    right_end = min(len(x), g1 + B)
    xR = x[g1:right_end]
    B_right = B if np.isfinite(xR).sum() >= max(3, B // 2) else int(np.isfinite(xR).sum())

    if B_left > 2 and y_gap.size >= B_left:
        w = _raised_cosine_weights(B_left)
        xL_use = pd.Series(x[g0 - B_left:g0]).interpolate(limit_direction="both").to_numpy()
        y_gap[:B_left] = (1 - w) * xL_use + w * y_gap[:B_left]

    if B_right > 2 and y_gap.size >= B_right:
        w = _raised_cosine_weights(B_right)
        xR_use = pd.Series(x[g1:g1 + B_right]).interpolate(limit_direction="both").to_numpy()
        y_gap[-B_right:] = (1 - w[::-1]) * xR_use + w[::-1] * y_gap[-B_right:]

    return y_gap


# ── Main orchestrator ────────────────────────────────────────────────────────

def cycle_synthesis_impute(
    x: np.ndarray,
    fs: float,
    anomaly_mask: np.ndarray | None = None,
    context_s: float = 20.0,
    blend_s: float = 0.5,
    min_cycles_each_side: int = 2,
    max_gap_cycles: int = 10,
    template_points: int = 200,
    template_cycles_use: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill NaN gaps using cycle-template synthesis.
    Gaps that can't be template-filled are left as NaN.

    Parameters
    ----------
    x : np.ndarray
        1D signal with NaN gaps.
    fs : float
        Sampling rate in Hz.
    anomaly_mask : np.ndarray or None
        Boolean mask where True = anomaly-flagged sample. If provided,
        only NaN gaps overlapping the mask are processed.
    context_s : float
        Seconds of clean context on each side of a gap.
    blend_s : float
        Seconds for raised-cosine cross-fade at gap edges.
    min_cycles_each_side : int
        Minimum clean cycles required on each side.
    max_gap_cycles : int
        Maximum gap length in breathing cycles to attempt.
    template_points : int
        Points in the resampled cycle template.
    template_cycles_use : int
        Clean cycles to use for the median template.

    Returns
    -------
    (x_filled, mask_imputed) : tuple[np.ndarray, np.ndarray]
    """
    x = np.asarray(x, dtype=float)
    x_filled = x.copy()
    mask_imputed = np.zeros(len(x), dtype=bool)

    gaps = _nan_gap_indices(x_filled)

    # Only process gaps that overlap the anomaly mask
    if anomaly_mask is not None:
        anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
        gaps = [(g0, g1, G) for g0, g1, G in gaps
                if np.any(anomaly_mask[g0:g1])]

    for g0, g1, G in gaps:
        if G <= 0:
            continue

        (l0, l1, xL), (r0, r1, xR) = _extract_clean_context(x_filled, g0, g1, fs, context_s)

        # Need at least 3 s of clean data per side to estimate breathing cycles
        if np.isfinite(xL).sum() < seconds_to_samples(3, fs) or np.isfinite(xR).sum() < seconds_to_samples(3, fs):
            continue

        # Detect troughs and estimate period/amplitude
        tL = _detect_troughs(xL, fs)
        tR = _detect_troughs(xR, fs)

        TL, AL, nCL = _estimate_period_amp_from_cycles(xL, tL, fs)
        TR, AR, nCR = _estimate_period_amp_from_cycles(xR, tR, fs)

        if nCL < min_cycles_each_side and nCR < min_cycles_each_side:
            continue

        # Select period
        T_used = _pick_finite_average(TL, TR)
        if T_used is None or T_used <= 0:
            continue

        # Gate on gap length
        n_cycles_gap = (G / fs) / T_used
        if n_cycles_gap > max_gap_cycles:
            continue

        # Select amplitude
        A_used = _pick_finite_average(AL, AR)
        if A_used is None:
            A_used = 2.0 * _robust_mad(np.r_[xL, xR])
            if not np.isfinite(A_used) or A_used <= 0:
                A_used = 1.0

        # Phase alignment from last trough in left context
        phase0_frac = 0.0
        if tL.size >= 2:
            last_trough_abs = l0 + tL[-1]
            dt_samples = max(0, g0 - last_trough_abs)
            phase0_frac = (dt_samples / (T_used * fs)) % 1.0

        # Build template (prefer left context, fallback to right)
        template = _build_cycle_template(xL, tL, M=template_points, n_cycles_use=template_cycles_use)
        if template is None:
            template = _build_cycle_template(xR, tR, M=template_points, n_cycles_use=template_cycles_use)

        if template is None or not np.isfinite(template).all():
            continue

        # Synthesize and crossfade
        y_gap = _synthesize_from_template(template, fs, G, T_used, phase0_frac, A_used)
        if y_gap is None or not np.isfinite(y_gap).all():
            continue

        y_gap = _edge_crossfade(x_filled, y_gap, g0, g1, fs, blend_s)

        x_filled[g0:g1] = y_gap
        mask_imputed[g0:g1] = True

    return x_filled, mask_imputed


def impute_anomaly_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    context_s: float = 20.0,
    blend_s: float = 0.5,
    min_cycles_each_side: int = 2,
    max_gap_cycles: int = 10,
    template_points: int = 200,
    template_cycles_use: int = 5,
) -> None:
    """
    Cycle-synthesis imputation for NaN gaps caused by anomaly detection.

    Uses a median breathing-cycle template built from clean context around
    each gap to synthesise a plausible fill, then cross-fades at the edges.
    Only gaps that overlap the ``{column}_anomaly`` mask are processed;
    hard-fault NaNs are left untouched.

    The ``_anomaly`` mask columns are dropped after imputation.

    Parameters
    ----------
    in_path : str
        Input directory (typically the post_anomaly_interp output).
    out_path : str
        Output directory.
    sampling_rate : int
        Sampling rate in Hz.
    context_s : float, optional
        Seconds of clean context on each side of a gap (default 20).
    blend_s : float, optional
        Seconds for raised-cosine cross-fade at gap edges (default 0.5).
    min_cycles_each_side : int, optional
        Minimum clean cycles required on each side (default 2).
    max_gap_cycles : int, optional
        Maximum gap length in breathing cycles to attempt (default 10).
    template_points : int, optional
        Points in the resampled cycle template (default 200).
    template_cycles_use : int, optional
        Clean cycles to use for the median template (default 5).

    Notes
    -----
    A gap is left as NaN (skipped) when any of the following apply:

    - The gap does not overlap the ``{column}_anomaly`` mask (i.e. it is
      not an anomaly NaN).
    - Fewer than 3 seconds of finite data exist in the context window on
      either side (not enough signal to estimate breathing parameters).
    - Fewer than ``min_cycles_each_side`` complete breathing cycles are
      detectable on both sides simultaneously.
    - Period estimation fails (no finite period can be derived from either
      side's trough spacing).
    - The gap is longer than ``max_gap_cycles`` breathing cycles.
    - Template construction fails (too few finite samples in the context
      cycles to build a reliable median shape).
    - The synthesised fill contains non-finite values.
    """
    mapped_files = map_files(in_path, file_ext='csv')

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
            anomaly_mask = (
                df[anomaly_col].values.astype(bool)
                if anomaly_col in df.columns
                else None
            )

            filled, _ = cycle_synthesis_impute(
                df[column].values,
                fs=sampling_rate,
                anomaly_mask=anomaly_mask,
                context_s=context_s,
                blend_s=blend_s,
                min_cycles_each_side=min_cycles_each_side,
                max_gap_cycles=max_gap_cycles,
                template_points=template_points,
                template_cycles_use=template_cycles_use,
            )
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
#
# GAUSSIAN SMOOTH
#
#
# =============================================================================
#

def _apply_gaussian_smooth(
    signal: np.ndarray,
    sampling_rate: int,
    sigma_seconds: float = 0.05,
) -> np.ndarray:
    """Apply a Gaussian smoothing filter (σ in seconds) to a 1-D signal."""
    sigma_samples = seconds_to_samples(sigma_seconds, sampling_rate)
    if sigma_samples < 1:
        sigma_samples = 1
    return gaussian_filter1d(signal, sigma=sigma_samples)


def _apply_gaussian_smooth_nan_safe(
    signal: np.ndarray,
    sampling_rate: int,
    sigma_seconds: float = 0.05,
) -> np.ndarray:
    """
    Gaussian-smooth a signal that may contain NaNs.

    If no NaNs are present, smooths directly. Otherwise smooths each contiguous
    non-NaN island separately and leaves NaN regions untouched.
    """
    signal = np.asarray(signal, dtype=float)

    # Fast path: no NaNs
    if not np.any(np.isnan(signal)):
        return _apply_gaussian_smooth(signal, sampling_rate, sigma_seconds)

    # NaNs present: smooth each non-NaN island separately
    sigma_samples = seconds_to_samples(sigma_seconds, sampling_rate)
    min_len = max(2 * sigma_samples + 1, 3) # floor of 3: minimum samples for smoothing to be meaningful
    result = np.full_like(signal, np.nan)

    for start, end, segment in _iter_nan_islands(signal):
        if len(segment) >= min_len:
            result[start:end] = _apply_gaussian_smooth(segment, sampling_rate, sigma_seconds)

    return result


def gaussian_smooth_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    sigma_seconds: float = 0.05,
) -> None:
    """
    Apply Gaussian smoothing to all signal columns in all CSV files.

    Smooths every non-time column of each CSV under ``in_path`` and writes the
    results to ``out_path``, preserving folder structure. Smoothing is NaN-safe:
    contiguous non-NaN runs are smoothed independently and NaN regions are left
    untouched.

    Parameters
    ----------
    in_path : str
        Input directory path containing CSV files.
    out_path : str
        Output directory path for smoothed CSV files.
    sampling_rate : int
        Sampling rate in Hz.
    sigma_seconds : float, optional
        Standard deviation of the Gaussian kernel in seconds. Default 0.05.

    Returns
    -------
    None
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        for column in df.columns:
            if column.lower() != 'time':
                df[column] = _apply_gaussian_smooth_nan_safe(
                    df[column].values, sampling_rate, sigma_seconds
                )

        relative_path = Path(file_path).relative_to(in_path_obj)
        output_file_path = out_path_obj / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file_path, index=False)

    print(f"Processed {len(mapped_files)} files from {in_path} to {out_path}")

#
# =============================================================================
#
# OTHER
#
# =============================================================================
#

def seconds_to_samples(seconds: float, sampling_rate: float) -> int:
    """Convert a duration in seconds to the nearest whole number of samples."""
    return int(round(seconds * sampling_rate))

def samples_to_seconds(samples: int, sampling_rate: float, decimal_places: int) -> float:
    """Convert a number of samples to a duration in seconds, rounded to the given decimal places."""
    return round(samples / sampling_rate, decimal_places)

def clean_signals(
    path_names: dict,
    sampling_rate: int,
    columns: list[str] | None = None,
    hard_fault_config: HardFaultConfig | None = None,
    detrend_window_s: int = 60,
    passband: str | tuple = 'default',
    do_anomaly: bool = True,
    do_smooth: bool = True,
    sigma_seconds: float = 0.05,
) -> None:
    """
    Apply all respiratory preprocessing steps to all signal files in a
    folder and its subfolders. Uses the ``path_names`` dictionary, starting
    with files in the ``'raw'`` path and moving through each stage as
    filters are applied.

    Optionally, ``do_anomaly`` and ``do_smooth`` can be set to True to
    perform those steps.

    Parameters
    ----------
    path_names : dict[str, str]
        A dictionary of file locations with keys for each stage in the
        processing pipeline. Required keys: ``'raw'``, ``'hard_fault'``,
        ``'micro_interp'``, ``'detrend'``, ``'bandpass'``.
        The dictionary can be created with the ``make_paths`` function.
    sampling_rate : int
        The sampling rate of the signal files in Hz.
    columns : list[str], optional
        Column names to process. Defaults to all non-time columns.
    hard_fault_config : HardFaultConfig, optional
        Configuration for hard-fault detection. Uses defaults if None.
    detrend_window_s : int, optional
        Window size in seconds for rolling-median detrending. Default 60.
    passband : str or tuple, optional
        Bandpass preset name or ``(lowcut, highcut)`` tuple in Hz.
        Default ``'default'`` (0.05-2.0 Hz).
    do_anomaly : bool, optional
        Whether to run anomaly detection, post-anomaly interpolation, and
        cycle-synthesis imputation. Default True.
    do_smooth : bool, optional
        Whether to apply Gaussian smoothing. Default True.
    sigma_seconds : float, optional
        Standard deviation of the Gaussian kernel in seconds for the
        smoothing step. Default 0.05.

    Raises
    ------
    KeyError
        If a required key is missing from ``path_names``, or if an optional
        step is enabled but its key is missing.
    """
    # --- validate required paths ---
    required = ['raw', 'hard_fault', 'micro_interp', 'detrend', 'bandpass']
    for key in required:
        if key not in path_names:
            raise KeyError(
                f"'{key}' path not detected in provided dictionary (path_names)."
            )

    # --- required steps ---
    hard_fault_signals(
        path_names['raw'], path_names['hard_fault'],
        sampling_rate, config=hard_fault_config, columns=columns,
    )
    micro_interp_signals(
        path_names['hard_fault'], path_names['micro_interp'],
        sampling_rate,
    )
    detrend_signals(
        path_names['micro_interp'], path_names['detrend'],
        sampling_rate, window_size_seconds=detrend_window_s,
    )
    bandpass_filter_signals(
        path_names['detrend'], path_names['bandpass'],
        sampling_rate, passband=passband,
    )

    last = 'bandpass'

    # --- optional: anomaly detection + imputation ---
    if do_anomaly:
        for key in ('anomaly', 'post_anomaly_interp', 'impute_anomaly'):
            if key not in path_names:
                raise KeyError(
                    f"'{key}' path not detected in provided dictionary "
                    f"(path_names). Required when do_anomaly=True."
                )
        detect_anomalies(
            path_names[last], path_names['anomaly'],
            sampling_rate,
        )
        post_anomaly_interp_signals(
            path_names['anomaly'], path_names['post_anomaly_interp'],
            sampling_rate,
        )
        impute_anomaly_signals(
            path_names['post_anomaly_interp'], path_names['impute_anomaly'],
            sampling_rate,
        )
        last = 'impute_anomaly'

    # --- optional: smoothing ---
    if do_smooth:
        if 'smooth' not in path_names:
            raise KeyError(
                "'smooth' path not detected in provided dictionary "
                "(path_names). Required when do_smooth=True."
            )
        gaussian_smooth_signals(
            path_names[last], path_names['smooth'],
            sampling_rate, sigma_seconds=sigma_seconds,
        )