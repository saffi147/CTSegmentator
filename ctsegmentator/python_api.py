import os
from pathlib import Path
from ctsegmentator.run_inference import nnUNet_predict_image
from importlib.metadata import version
from ctsegmentator.post_processing import postprocessing
from ctsegmentator.download_weights import download_fold_weights_via_api as dw
import csv
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

    dw(weights_dir)

    # figure out how many cases there are
    patient_dirs = [d for d in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, d))]
    num_patients = len(patient_dirs)
    print(f"Number of cases found: {num_patients}")

    # create map for old patient directory to new patient directory 
    mapping = []

    
    i = 1
    for case in os.listdir(ct_dir):
        case_path = os.path.join(ct_dir, case)
        if not os.path.isdir(case_path):
            continue
        data = os.path.join(case_path, "DATA", "DICOM")
        print(f"Processing {case} ({i}/{num_patients}). Cases left: {num_patients - i}")
        
        old_patient_id, new_patient_id = nnUNet_predict_image(i, weights_dir, data, output_dir, file_format, 
                            device, step_size=0.5, use_tta=True, verbose=False)
        mapping.append({'new_patient_id': new_patient_id, 'original_patient_id': old_patient_id})
        i = i + 1
    
    print(mapping)

    # Save mapping.csv
    mapping_file = os.path.join(output_dir, "patient_id_mapping.csv")
    with open(mapping_file, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['new_patient_id', 'original_patient_id'])
        writer.writeheader()
        writer.writerows(mapping)

    print('Inference for available Ct scans is complete.')