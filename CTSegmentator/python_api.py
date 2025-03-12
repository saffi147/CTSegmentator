from CTSegmentator.download_weights import download_fold_weights_via_api
from CTSegmentator.inference import nnUNet_predict_image
from CTSegmentator.config import setup_ctsegmentator


def ctsegmentator (pet_dir=None, 
                     ct_dir = None,
                     output_dir=None):
    """
    Runs the full CT segmentation pipeline.

    Args:
        input_dir (str): Directory containing CT files.
        output_dir (str): Directory to save the segmentation results.
        file_format (str): Input file format, either "dicom" or "nifti". Defaults to "dicom".
        suv_threshold (float): SUV threshold for segmentation refinement. Defaults to 3.

    """
    
    weights_dir = setup_ctsegmentator() 

    download_fold_weights_via_api(output_dir = weights_dir) #downloads dataset.json, plans.json etc to be able to run inference. 

    nnUNet_predict_image(weights_dir, ct_dir, output_dir,
                         device='cuda', step_size=0.5, use_tta=True, verbose=False)
    