import os
import tempfile
from pathlib import Path
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
import shutil  
import pydicom

def dicom_nifti_conversion(dicom_series_path: str, nifti_output_dir:str):
    """
    Converts dicom input into nifti output, saves in specified folder. 

    Args:  
        dicom_input_path(str ): path to dicom file
        nifti_output_dir (str): path where nifti file will be saved (directory)
    """
    # Ensure the output directory exists
    os.makedirs(nifti_output_dir, exist_ok=True)

    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_series_path)
    reader.SetFileNames(dicom_series)

    # Load the volume and orient it
    volume = reader.Execute()
    volume = sitk.DICOMOrient(volume, "LPS")

    patient_folder_name = os.path.basename(os.path.dirname(os.path.dirname(dicom_series_path)))
    nifti_filename = f"{patient_folder_name}.nii.gz"    
    output_nifti_path = os.path.join(nifti_output_dir, nifti_filename)

    # Write NIfTI
    sitk.WriteImage(volume, output_nifti_path)
    return output_nifti_path

def get_patient_id(input):
    """
    Retrieve the pateint ID from the first dicom in the folder of inputs
    Returns: 
        patient_id (str)
    """
    for file in os.listdir(input):
        if file.lower().endswith(('.dcm', '')):  # Often no extension or .dcm
            file_path = os.path.join(input, file)
            try:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                return dcm.PatientID
            except Exception:
                continue
    raise ValueError(f"No valid DICOM files found in {input}")


def nnUNet_predict_image(patient_count, model_folder: str, ct_input: str, output_path: str, image_type: str,
                         device='cpu', step_size=0.5, use_tta=False, verbose=False):
    """
    Runs inference using nnUNet for a directory of CT images. Can be dicom or nifti. 

    Args:
        model_folder (str): Path to the trained model folder.
        ct_input (str): Path to the input CT image firectory.
        output_path (str): Path to the output directory where segmentation results are saved
                           If empty, will return the segmentation results.
        image_type (str): "dicom" or "nifti". defaults to dicom
        device (str): Device for inference ('cuda', 'cpu', or 'mps'). Defaults to 'cuda'.
        step_size (float): Step size for sliding window prediction. Defaults to 0.5.
        use_tta (bool): Whether to use test-time augmentation (mirroring). Defaults to False.
        verbose (bool): Whether to enable verbose output. Defaults to False.

    Returns: 
        original_patient_id (str)
        new_patient_id (str)
    """
    # check for valid device 
    assert device in ['cuda', 'cpu', 'mps'] or isinstance(device, torch.device), (
        f"Invalid device specified: {device}. Must be 'cuda', 'cpu', 'mps', or a valid torch.device."
    )
    # Check for valid image_type
    assert image_type in ['dicom', 'nifti'], (
        f"Invalid image type specified: {image_type}. Must be 'dicom' or 'nifti'."
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

    # initialise nnUnet from model folder
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=None,checkpoint_name = "checkpoint_final.pth")
    
    # setup output path 
    ct_input = Path(ct_input)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    new_patient_id = f"patient{patient_count:03d}"
    output_dir = output_path / new_patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        #have to make a temporary directory to save the nifti file - cannot read into a zip file directly. 
        tmp_dir = Path(tmp_dir)

        # if input files are dicom, convert all to nifti
        if image_type == "dicom":
            print("Converting DICOMs to NIfTI...")
            # ct_input == the path to the CT dicoms 
            old_patient_id = get_patient_id(ct_input)
            try:
                nifti_file = dicom_nifti_conversion(ct_input, tmp_dir)
            except Exception as e:
                    print(f"Exception occurred before calling dicom_nifti_conversion: {e}")
            # Also save a copy into CT_scans folder:
            destination = os.path.join(output_dir, "ct_scan.nii.gz")
            shutil.copy(nifti_file, destination)
        else: 
            #iterate over files in ct_input and copy only those ending in .nii.gz into tmp_dir
            for file in Path(ct_input).glob("*.nii.gz"):
                if file.is_file():
                    shutil.copy(file, tmp_dir)
        
        print("Running prediction...")
        
        dir_in = str(tmp_dir)
        if not os.path.exists(tmp_dir):
            raise ValueError(f"Input directory '{dir_in}' does not exist!")

        files = os.listdir(dir_in)
        if not files:
            raise ValueError(f"Input directory '{dir_in}' is empty!")
        
        dir_out = str(output_dir)

        # Ensure dir_in contains valid files
        files = [os.path.join(dir_in, f) for f in os.listdir(dir_in) if f.endswith('.nii.gz')]

        # Convert to list-of-lists format STRING INPUT DOES NOT WORK HERE
        list_of_lists_or_source_folder = [[f] for f in files] if files else []

        predictor.predict_from_files(list_of_lists_or_source_folder=list_of_lists_or_source_folder,
                                     output_folder_or_list_of_truncated_output_files=dir_out,
                                     save_probabilities=False,
                                     overwrite=True,
                                     )
        return old_patient_id, new_patient_id