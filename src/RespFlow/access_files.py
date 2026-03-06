import os, re
import pandas as pd

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
    
    Adapted from EMGFlow package by William Conely.

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
        'hard_fault':os.path.join(root, '2_hard_fault'),
        'detrend':os.path.join(root, '3_detrend'),
        'bandpass':os.path.join(root, '4_bandpass'),
        'fwr':os.path.join(root, '5_fwr'),
        'screened':os.path.join(root, '6_screened'),
        'filled':os.path.join(root, '7_filled'),
        'smooth':os.path.join(root, '8_smoothed'),
        'feature':os.path.join(root, '9_feature')
    }
    
    # Create folders
    for value in path_names.values():
        os.makedirs(value, exist_ok=True)
    
    # Return dictionary
    return path_names

#
# =============================================================================
#

def map_files(in_path: str, 
              file_ext: str = 'csv', 
              expression: str | None = None, 
              base: str | None = None) -> dict[str, str]:
    """
    Generate a dictionary of file names and locations (keys/values) from the
    subfiles of a folder.
    
    Adapted from EMGFlow package by William Conely.
    
    Parameters
    ----------
    in_path : str
        The filepath to a directory to read files.
    file_ext : str, optional
        The file extension for files to read. Only reads files with this
        extension. The default is 'csv'.
    expression : str, optional
        A regular expression. If provided, will only count files whose relative
        paths from 'base' match the regular expression. The default is None.
    base : str, optional
        The path of the root folder the path keys should start from. Used to
        track the relative path during recursion. The default is None.
    
    Raises
    ------
    Exception
        An exception is raised if 'expression' is not None or a valid regular
        expression.

    Returns
    -------
    file_dirs : dict-str
        A dictionary of file name keys and file path location values.
        
    """
    
    # Throw error if Regex does not compile
    if expression is not None:
        try:
            re.compile(expression)
        except:
            raise Exception("Invalid regex expression provided")
    
    # Set base path and ensure in_path is absolute
    if base is None:
        if not os.path.isabs(in_path):
            in_path = os.path.join(os.getcwd(), in_path)
        base = in_path
    
    # Build file directory dictionary
    file_dirs = {}
    for directory in os.listdir(in_path):
        new_path = os.path.join(in_path, directory)
        fileName = os.path.relpath(new_path, base)
        
        # Recursively check folders
        if os.path.isdir(new_path):
            subDir = map_files(new_path, file_ext=file_ext, expression=expression, base=base)
            file_dirs.update(subDir)
        
        # Record the file path (from base to current folder) and absolute path
        elif (directory.endswith(file_ext)) and ((expression is None) or (re.match(expression, fileName)!=None)):
            file_dirs[fileName] = new_path
            
    return file_dirs

# 
# =============================================================================
# 

from pathlib import Path
from importlib import resources as ir
import shutil

def make_sample_data(path_names: dict[str, str]) -> None:
    """
    Copies sample data files from the RespFlow package to the raw data folder
    specified in `path_names`.
    
    Parameters
    ----------
    path_names : dict[str, str]
        A dictionary of file locations with keys for stage in the processing
        pipeline. Must contain a 'raw' key specifying the destination folder
        for the sample data.
    
    Raises
    ------
    ValueError
        If 'raw' key is not found in `path_names`.
    
    Returns
    -------
    None
        The function performs file operations only; it does not return a value.
    """
    if "raw" not in path_names:
        raise ValueError("Raw path not detected in path_names.")

    dest = Path(path_names["raw"])
    dest.mkdir(parents=True, exist_ok=True)

    with ir.as_file(ir.files("RespFlow").joinpath("data")) as src:
        shutil.copytree(src, dest, dirs_exist_ok=True)

