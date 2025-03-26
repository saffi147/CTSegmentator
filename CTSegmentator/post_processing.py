import os
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

    # get the shape of the image
    size = npy_ct.shape

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
    df['slices_z'] = size[0]
    df['slices_y'] = size[1]
    df['slices_x'] = size[2]
    
    return df

def postprocessing(ct_input, pred_dir, image_type, output_path):
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
    for case in os.listdir(ct_input):
        if case.endswith("nii.gz"):
            ct_path = os.path.join(ct_input, case)
            pred_path = os.path.join(pred_dir, case) # assumes prediction file and ct file are named the exact same thing
        if os.path.exists(pred_path):
            patient_df = compute_parameters(ct_path, pred_path)
        else:
            print(f"Prediction file not found for {case}") 
        if patient_df is not None:
            summary_df.append(patient_df)

    # save df to csv
    final_df = pd.concat(summary_df, ignore_index=True)
    csv_path = os.path.join(output_path, "postprocessing_results.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Postprocessing results saved to: {csv_path}")
          