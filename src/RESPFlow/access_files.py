import os

#
# =============================================================================
#

"""
A collection of functions for accessing files.
"""

#
# =============================================================================
#

def make_paths(root: str | None = None, raw: str | None = None) -> dict[str, str]:
    """
    Generates a file structure for a RESP workflow, and returns a dictionary of
    the locations for these files for easy use with EMG processing functions.
    
    Creates subfolders for each stage of the processing pipeline.
    
    If no path is given, will create a `data` folder in the current working
    directory, with these subfolders inside.
    
    Taken from EMGFlow package by William Conely.

    Parameters
    ----------
    root : str, optional
        The root where the folders are generated. If not specified, it uses the cwd by default.
    raw : str, optional
        The absolute path for the raw data. If not specified, it creates a folder inside `root/data`.

    Returns
    -------
    path_names : dict[str, str]
        A dictionary of file locations with keys for stage in the processing
        pipeline.

    """
    
    root = os.path.abspath(root or os.path.join(os.getcwd(), 'data'))
    raw = os.path.abspath(raw or os.path.join(root, '1_raw'))
    
    # Create dictionary
    path_names = {
        'raw':raw,
        'notch':os.path.join(root, '2_notch'),
        'bandpass':os.path.join(root, '3_bandpass'),
        'fwr':os.path.join(root, '4_fwr'),
        'screened':os.path.join(root, '5_screened'),
        'filled':os.path.join(root, '6_filled'),
        'smooth':os.path.join(root, '7_smoothed'),
        'feature':os.path.join(root, '8_feature')
    }
    
    # Create folders
    for value in path_names.values():
        os.makedirs(value, exist_ok=True)
    
    # Return dictionary
    return path_names

#
# =============================================================================
#