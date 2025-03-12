"""
This module sets up the configuration for the CT Segmentator.
It defines the directory structure and environment variables needed
for the model weights and results.
"""

import os
from pathlib import Path
from importlib.metadata import version
__version__ = version("CTSegmentator")

def get_ctsegmentator_dir(version):
    """
    Get the path to the ctsegmentator directory, containing the model weights
    that have been downloaded.
    """
    
    if "CT_SEGMENTATOR_HOME" in os.environ:
        base_dir = Path(os.environ["CT_SEGMENTATOR_HOME"])
    else:
        base_dir = Path.home() / ".ctsegmentator"
    return base_dir / version

def setup_ctsegmentator():
    current_version = __version__
    print(current_version)
    print(f"Setting ctsegmentator version {current_version}")
    version_dir = get_ctsegmentator_dir(current_version)
    weights_dir = version_dir / "results"
    print(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(weights_dir)
    os.environ["nnUNet_preprocessed"] = str(weights_dir)
    os.environ["nnUNet_results"] = str(weights_dir)
    return weights_dir