# import pydicom
import tempfile
from pathlib import Path
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import pydicom 
import nibabel as nb
import shutil

import os

def is_dicom(input_path):
    """
    Determines if the input is a DICOM file or directory.

    Args:
        input_path (str or Path): Path to the input file or directory.

    Returns:
        bool: True if the input is in DICOM format, False otherwise.
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        # Check if any file in the directory is a valid DICOM file
        for file in input_path.iterdir():
            try:
                pydicom.dcmread(file, stop_before_pixels=True)
                return True
            except Exception:
                continue
    else:
        try:
            pydicom.dcmread(input_path, stop_before_pixels=True)
            return True
        except Exception:
            pass

    return False

def read_dicom_image(dicom_path):
    """
    Read a DICOM image series, convert to nifti
    
    Args:
        dicom_path (str|pathlib.Path): Path to the DICOM series to read
    Returns:
        sitk.Image: The image as a SimpleITK Image (nii.gz) 
                    This is required for nnUNet inference to work. 
    """
    print(dicom_path)
    dicom_images = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(dicom_path))
    return sitk.ReadImage(dicom_images)

def nnUNet_predict_image(model_folder, ct_input, output_path,
                         device='cpu', step_size=0.5, use_tta=False, verbose=False):
    """
    Runs inference using nnUNet for a single CT image.

    Args:
        model_folder (str): Path to the trained model folder.
        ct_input (str): Path to the input CT image.
        output_path (str): Path to the output directory where segmentation results are saved
                           If empty, will return the segmentation results.
        device (str): Device for inference ('cuda', 'cpu', or 'mps'). Defaults to 'cuda'.
        step_size (float): Step size for sliding window prediction. Defaults to 0.5.
        use_tta (bool): Whether to use test-time augmentation (mirroring). Defaults to False.
        verbose (bool): Whether to enable verbose output. Defaults to False.

    """
    assert device in ['cuda', 'cpu', 'mps'] or isinstance(device, torch.device), (
        f"Invalid device specified: {device}. Must be 'cuda', 'cpu', 'mps', or a valid torch.device."
    )

    if device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        torch.set_num_threads(1)
        device = torch.device('cuda')
    elif isinstance(device, torch.device):
        torch.set_num_threads(1)
        device = device
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(model_folder, 
                                                   use_folds=None,
                                                   checkpoint_name = "checkpoint_final.pth")
    
    ct_input = Path(ct_input)
    print(ct_input)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        #have to make a temporary directory to save the nifti file - cannot read into a zip file directly. 
        tmp_dir = Path(tmp_dir)
        print(f"Temporary directory: {tmp_dir}")

        temp_ct_path = shutil.copy(ct_input, tmp_dir)

        # Process CT image
        # print("Reading CT NIfTI image...")
        # ct_image = sitk.ReadImage(str(nifti_file_path))

        ######## TO DO CHECK IF IS A DICOM IMAGE 
        # if is_dicom(ct_input):
        #     print("Reading CT DICOM image...")
        #     ct_image = read_dicom_image(ct_input)
        # else:
        #     print("Reading CT NIfTI image...")
        #     ct_image = sitk.ReadImage(str(ct_input))


        print("Running prediction...")
        
        dir_in = str(tmp_dir)
        print(dir_in)
        if not os.path.exists(tmp_dir):
            raise ValueError(f"Input directory '{dir_in}' does not exist!")

        files = os.listdir(dir_in)
        print(files)
        if not files:
            raise ValueError(f"Input directory '{dir_in}' is empty!")
        
        dir_out = str(output_path)

        # Ensure dir_in contains valid files
        files = [os.path.join(dir_in, f) for f in os.listdir(dir_in) if f.endswith('.nii.gz')]

        # Convert to list-of-lists format
        list_of_lists_or_source_folder = [[f] for f in files] if files else []



        
        predictor.predict_from_files(list_of_lists_or_source_folder=list_of_lists_or_source_folder,
                                     output_folder_or_list_of_truncated_output_files=dir_out,
                                     save_probabilities=False,
                                     overwrite=True,
                                     )


# note that images for inference must be in the form: case_000_00_0000.nii.gz

##############testing remove later bro

def ctsegmentator (ct_dir, pred_dir, file_format="nifti"):
    """
    Runs the full CT segmentation pipeline.

    Args:
        ct_dir (str): Directory containing CT files.
        pred_dir (str): Directory to save the segmentation predictions.
        file_format (str): Input file format, either "dicom" or "nifti". Defaults to "nifti".
    """
    print(ct_dir)
    print(pred_dir)
    #if files are dicom, convert to nifti
    #TO DO 
    
    # weights_dir = setup_ctsegmentator()

    # download_fold_weights_via_api(output_dir = weights_dir)
    # for now set the fold weights manually

    weights_dir = r'C:\Users\SaffiHunt\OneDrive - UWA\00 THESIS\testing_CTseg\model_weights'

    os.environ["nnUNet_raw"] = str(weights_dir)
    os.environ["nnUNet_preprocessed"] = str(weights_dir)
    os.environ["nnUNet_results"] = str(weights_dir)

    nnUNet_predict_image(weights_dir, ct_dir, pred_dir,
                        device='cpu', step_size=0.5, use_tta=True, verbose=False)
    

if __name__ == '__main__':
    input_dir = r"C:\Users\SaffiHunt\OneDrive - UWA\00 THESIS\testing_CTseg\ct_dir"
    output_dir = r"C:\Users\SaffiHunt\OneDrive - UWA\00 THESIS\testing_CTseg\pred_dir"

    input_dir = r"C:\Users\SaffiHunt\OneDrive - UWA\00 THESIS\testing_CTseg\ct_dir\case_010_30.nii.gz"


    ctsegmentator(ct_dir = input_dir, pred_dir = output_dir, file_format="nifti")


