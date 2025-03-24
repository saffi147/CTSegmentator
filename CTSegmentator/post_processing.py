import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import SimpleITK as sitk
import numpy as np
import pandas as pd
from ctsegmentator.run_inference import dicom_nifti_conversion
import tempfile
from pathlib import Path
import shutil
import os
import nibabel as nib

def read_image(ct_path, pred_path):
    """
    Reads the CT and prediction images from the given paths. 

    Args: 
        ct_path (str): Path to the CT image.
        pred_path (str): Path to the prediction image.
    """
    image_CT = sitk.ReadImage(ct_path)
    image_prediction = sitk.ReadImage(pred_path)
    return image_CT, image_prediction


def compute_parameters(ct_path, pred_path):
    """
    Computes the volume of each segmentation, and average HU of each label. 

    Args: 
        ct_path (str): Path to the CT image.
        pred_path (str): Path to the prediction image.
    """
    # load in the CT scan and segmentation
    ct_image = sitk.ReadImage(ct_path)
    spacing = ct_image.GetSpacing()
    sitk_seg = sitk.ReadImage(pred_path)

    # Get a numpy array from the label map image
    npy_label = sitk.GetArrayFromImage(sitk_seg)
    npy_ct = sitk.GetArrayFromImage(ct_image)

    # Get a list of the IDs
    label_ids = np.unique(npy_label)

    label_list = []
    volume_list = []
    average_hu_list = []
    stddev_hu_list = []

    patient_id = Path(ct_path).name.replace('.nii.gz', '')

    for i in label_ids:

        #skip the background label
        if i == 0:
            continue
        
        label_mask = npy_label == i

        # number of voxels
        voxel_count = np.count_nonzero(npy_label == i)
        # Convert to mm^3
        volume_mm3 = voxel_count*(spacing[0]*spacing[1])

        #average HU
        hu_vals = npy_ct[label_mask]
        average_hu = np.mean(hu_vals)
        std_dev_hu = np.std(hu_vals)

        label_list.append(i)
        volume_list.append(volume_mm3)
        average_hu_list.append(average_hu)
        stddev_hu_list.append(std_dev_hu)
    
    # Create a DataFrame from the results
    df = pd.DataFrame({
        'label': label_list,
        'volume_mm3': volume_list,
        'average_hu': average_hu_list,
        'std_dev_hu': stddev_hu_list
    })

    df['case'] = patient_id
    df['spacing_x_mm'] = spacing[0]
    df['spacing_y_mm'] = spacing[1]
    df['spacing_z_mm'] = spacing[2]
    
    return df

def postprocessing(ct_input, pred_dir, image_type):
    """
    Compute the volumes of the muscle and fat from the given CT and prediction files.
    
    Args: 
        ct_file (sitk): CT image.
        pred_file (sitk): Prediction image.
    """
    # setup output path 
    ct_input = Path(ct_input)
    pred_dir = Path(pred_dir)

    summary_df = []

    # iterate over each case in the directory 
    # this assumes the dataset is set up 
    for case in os.listdir(ct_input):
        case_path = os.path.join(ct_input, case)
        if not os.path.isdir(case_path):
            continue
        data = os.path.join(case_path, "DATA", "DICOM")

        with tempfile.TemporaryDirectory() as tmp_dir:
        #have to make a temporary directory to save the nifti file - cannot read into a zip file directly. 
            tmp_dir = Path(tmp_dir)
            print(f"Temporary directory: {tmp_dir}")

            # if input files are dicom, convert all to nifti
            if image_type == "dicom":
                print("Converting DICOMs to NIfTI...")
                dicom_nifti_conversion(data, tmp_dir)

            else: 
                # TODO
                # adapt for nifti files
                for file in Path(ct_input).glob("*.nii.gz"):
                    if file.is_file():
                        shutil.copy(file, tmp_dir)
            
            for ct_file in os.listdir(tmp_dir):
                if ct_file.endswith("nii.gz"):
                    ct_path = os.path.join(tmp_dir, ct_file)
                    pred_path = os.path.join(pred_dir, ct_file) # assumes prediction file and ct file are named the exact same thing
                if os.path.exists(pred_path):
                    patient_df = compute_parameters(ct_path, pred_path)
                else:
                    print(f"Prediction file not found for {ct_file}") 
                if patient_df is not None:
                    summary_df.append(patient_df)

    # save df to csv
    final_df = pd.concat(summary_df, ignore_index=True)
    csv_path = os.path.join(output, "postprocessing_results.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Postprocessing results saved to: {csv_path}")
            


# def compute_3d_smi(height, df):
#     """
#     Compute the skeletal muscle index (SMI) from the given height and volume data.

#     Args: 
#         height (float): Height of the patient.
#         df (pd.DataFrame): DataFrame containing the volume and HU data.
#     """

#     muscle_volume = df[df['label'] == 2]['volume_mm3'].values

#     smi = muscle_volume/((height)*(height))
    
#     return smi

if __name__ == "__main__":
    ct_path = r"/Users/saffihunt/Library/CloudStorage/OneDrive-UWA/00 THESIS/testing_CTseg/root"
    output = r'/Users/saffihunt/Library/CloudStorage/OneDrive-UWA/00 THESIS/testing_CTseg/root_output'

    postprocessing(ct_path, output, "dicom")