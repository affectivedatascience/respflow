from RespFlow.access_files import map_files
from scipy.signal import butter, sosfiltfilt
import pandas as pd
from pathlib import Path
from scipy.ndimage import median_filter

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

def apply_detrend(signal, sampling_rate, window_size_seconds=60):
    # IMPORTANT: 
    # 60-second window
    window_size = int(window_size_seconds * sampling_rate)
    
    # Calculate rolling median
    baseline = median_filter(signal, size=window_size)
    
    # Detrend
    detrended_signal = signal - baseline
    
    return detrended_signal, baseline

def detrend_signals(in_path: str, out_path: str, sampling_rate, window_size_seconds=60):
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

def apply_bandpass(data, fs, lowcut=0.05, highcut=2.0, order=2):
    """
    Applies a zero-phase Butterworth bandpass filter.
    Standard: 0.05-2.0 Hz for RIP belt data.
    """
    nyquist = 0.5 * fs
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
def bandpass_filter_signals(in_path: str, out_path: str, sampling_rate, lowcut=0.05, highcut=2.0, order=2):
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
    lowcut : float, optional
        Low cutoff frequency in Hz (default: 0.05)
    highcut : float, optional
        High cutoff frequency in Hz (default: 2.0)
    order : int, optional
        Filter order (default: 2)
    """
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

