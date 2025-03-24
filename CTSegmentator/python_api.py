import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctsegmentator.run_inference import nnUNet_predict_image
from pathlib import Path
from importlib.metadata import version
from ctsegmentator.post_processing import postprocessing
__version__ = version("CTSegmentator")

def setup_ctsegmentator(weights_dir: str):
    """
    Sets up the configuration for the CT Segmentator.
    Defines the directory structure and environment variables needed
    for the model weights and results.
    """
    print(f"Setting up CT segmentator Version {__version__}")
    os.environ["nnUNet_raw"] = str(weights_dir)
    os.environ["nnUNet_preprocessed"] = str(weights_dir)
    os.environ["nnUNet_results"] = str(weights_dir)


def ctsegmentator(weights_dir=r"model_weights", 
                     ct_dir = None,
                     output_dir=None, 
                     device="cpu",
                     file_format: str = "dicom"):
    """
    Runs the full CT segmentation pipeline, along with post processing steps. 

    Args:
        weights_dir (str): Directory containing trained model weights. 
        ct_dir (str): Directory containing CT files.
        output_dir (str): Directory to save the segmentation results.
        device (str): 
        file_format (str): Input file format, either "dicom" or "nifti". Defaults to "dicom".
    """
    setup_ctsegmentator(weights_dir)

    for case in os.listdir(ct_dir):
        case_path = os.path.join(ct_dir, case)
        if not os.path.isdir(case_path):
            continue
        data = os.path.join(case_path, "DATA", "DICOM")
        nnUNet_predict_image(weights_dir, data, output_dir, file_format, 
                            device, step_size=0.5, use_tta=True, verbose=False)
        print('Inference for available Ct scans is complete.')
        
    print("Beginning post processing...")
    postprocessing(ct_dir, output_dir, file_format)

    # post processing step
    # input is ct_dir, output 


if __name__ == '__main__':


    dir = r"/Users/saffihunt/Library/CloudStorage/OneDrive-UWA/00 THESIS/testing_CTseg/root"
    out = r'/Users/saffihunt/Library/CloudStorage/OneDrive-UWA/00 THESIS/testing_CTseg/root_output'


    ctsegmentator(ct_dir = dir, output_dir = out, device = "cpu", file_format="dicom")