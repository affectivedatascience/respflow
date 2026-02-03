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
# =============================================================================
#

#
# =============================================================================
#

