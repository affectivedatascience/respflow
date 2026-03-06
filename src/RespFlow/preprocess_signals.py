from RespFlow.access_files import map_files
from scipy.signal import butter, sosfiltfilt
import pandas as pd
from pathlib import Path
from scipy.ndimage import median_filter
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
    flat_eps_abs: float = 0.0            # absolute threshold on |dx| (optional)
    flat_eps_frac_mad_x: float = 1e-4    # eps component as fraction of MAD(x)
    flat_eps_k_mad_dx: float = 0.05      # eps component as fraction of MAD(dx)

    # Clipping / saturation (data-driven rails)
    clip_low_pct: float = 0.1
    clip_high_pct: float = 99.9
    clip_tol_frac_mad_x: float = 0.01
    clip_min_run_s: float = 0.25

    # Step/discontinuity spikes
    step_k_mad_dx: float = 12.0
    step_pad_s: float = 0.05             # pad around steps (seconds)
    step_min_thr_frac_mad_x: float = 0.5 # floor: threshold >= this * MAD(x)
    step_verify_window_s: float = 0.5    # window (seconds) to check sustained level shift
    step_verify_min_shift_frac_mad_x: float = 0.3  # minimum median shift to confirm step
    step_spike_bypass_k: float = 3.0     # skip verification if |dx| exceeds threshold by this factor

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
    return_info: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Detect hard faults and set them to NaN.

    Returns:
        signal_out: signal with hard-fault samples set to NaN
        info: dict with masks and optional thresholds used
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
    eps = max(
        float(config.flat_eps_abs),
        float(config.flat_eps_frac_mad_x) * mad_x,
        float(config.flat_eps_k_mad_dx) * mad_dx,
        np.finfo(float).eps,
    )

    dx_abs = np.abs(dx)
    mask_dx_small = (dx_abs <= eps) & ~np.isnan(dx)

    min_flat_samples = int(np.ceil(config.flat_min_s * fs))
    mask_flatline = np.zeros(N, dtype=bool)

    # A run of dx_small from [a, b) implies constant samples [a, b+1)
    for a, b in _runs_from_mask(mask_dx_small):
        sa, sb = a, min(N, b + 1)
        if (sb - sa) >= min_flat_samples:
            mask_flatline[sa:sb] = True

    mask_flatline &= ~mask_nan

    # -------------------------------------------------------------------------
    # Clipping / saturation (runs near inferred rails)
    # -------------------------------------------------------------------------
    xv = x[~mask_nan]
    lo = np.percentile(xv, config.clip_low_pct)
    hi = np.percentile(xv, config.clip_high_pct)
    tol = float(config.clip_tol_frac_mad_x) * mad_x

    mask_clip_raw = (~mask_nan) & ((x <= lo + tol) | (x >= hi - tol))
    min_clip_samples = int(np.ceil(config.clip_min_run_s * fs))
    mask_clip = _apply_min_run_length(mask_clip_raw, min_clip_samples)

    # -------------------------------------------------------------------------
    # Step/discontinuity spikes (robust threshold on dx)
    # -------------------------------------------------------------------------
    dxv = dx[~np.isnan(dx)]
    dx_med = float(np.median(dxv)) if dxv.size else 0.0

    # Fix: floor the threshold so it never collapses for smooth signals
    thr_dx = float(config.step_k_mad_dx) * mad_dx
    thr_floor = float(config.step_min_thr_frac_mad_x) * mad_x
    thr = max(thr_dx, thr_floor)

    step_candidates = np.where(np.abs(dx - dx_med) > thr)[0]
    mask_step = np.zeros(N, dtype=bool)
    step_pad = int(np.ceil(config.step_pad_s * fs))

    # Fix: verify each candidate by checking for a sustained level shift
    verify_win = int(np.ceil(config.step_verify_window_s * fs))
    min_shift = float(config.step_verify_min_shift_frac_mad_x) * mad_x

    spike_bypass_thr = config.step_spike_bypass_k * thr

    for i in step_candidates:
        dx_mag = abs(float(dx[i]) - dx_med)

        # Massive spike — flag unconditionally, no verification needed
        if dx_mag >= spike_bypass_thr:
            a = max(0, i - step_pad)
            b = min(N, i + 2 + step_pad)
            mask_step[a:b] = True
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

        a = max(0, i - step_pad)
        b = min(N, i + 2 + step_pad)
        mask_step[a:b] = True

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
        "flat_eps": eps,
        "clip_lo": lo,
        "clip_hi": hi,
        "clip_tol": tol,
        "step_thr": thr,
    }
    return x, info


def apply_hard_fault_to_df(
    df: pd.DataFrame,
    sampling_rate: int,
    config: HardFaultConfig | None = None,
    time_col: str = "time",
    add_mask_cols: bool = True,
) -> pd.DataFrame:
    """
    Apply hard-fault detection to all non-time columns of a dataframe.

    - Replaces hard-fault samples with NaN in each signal column.
    - Optionally writes mask columns:
        <col>_mask_hardfault
        <col>_mask_flatline
        <col>_mask_clip
        <col>_mask_step
    """
    out = df.copy()

    for column in out.columns:
        if column.lower() == time_col.lower():
            continue

        x = out[column].to_numpy(dtype=float)
        x_hf, info = apply_hard_fault(x, sampling_rate, config=config, return_info=True)
        out[column] = x_hf

        if add_mask_cols:
            out[f"{column}_mask_hardfault"] = info["mask_hardfault"].astype(bool)
            out[f"{column}_mask_flatline"] = info["mask_flatline"].astype(bool)
            out[f"{column}_mask_clip"] = info["mask_clip"].astype(bool)
            out[f"{column}_mask_step"] = info["mask_step"].astype(bool)

    return out


def hard_fault_signals(
    in_path: str,
    out_path: str,
    sampling_rate: int,
    config: HardFaultConfig | None = None,
    add_mask_cols: bool = False,
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
    add_mask_cols : bool, optional
        Whether to include boolean mask columns in output CSVs (default: False).
        Set to False to avoid propagating mask columns to downstream steps.
    """
    mapped_files = map_files(in_path, file_ext='csv')

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    for file_path in mapped_files.values():
        df = pd.read_csv(file_path)

        df2 = apply_hard_fault_to_df(df, sampling_rate, config=config, time_col="time", add_mask_cols=add_mask_cols)

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

# Physiological constant: typical resting respiratory rate
RESTING_RR_HZ = 0.5  # 0.5 Hz ≈ 30 breaths/min (upper bound for resting adults)


def default_max_gap(sampling_rate: int, rr: float = RESTING_RR_HZ) -> int:
    """
    Compute a default max_gap (in samples) for NaN interpolation before bandpass.

    Rule: 10% of one respiratory cycle length.
    At 2000 Hz, 0.5 Hz: 0.1 * (2000 / 0.5) = 400 samples.
    """
    return int(round(0.1 * sampling_rate / rr))


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
        Maximum gap size (in samples) to interpolate over. If None, uses
        default_max_gap(sampling_rate) based on resting respiratory rate.
        Gaps larger than max_gap remain as NaN in the output.
    interp_method : Specified interpolation method. Defaults to pchip.
        Alternatively user can specify "cubic_spline".
    """

    # Apply physiological default if max_gap not specified
    if max_gap is None:
        max_gap = default_max_gap(sampling_rate)

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

