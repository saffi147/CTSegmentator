import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

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

def get_height(pred_file):
    """
    Calculate the height of the patient from the prediction file.
    """
    #TO DO

    return height

def compute_volumes(ct_file, pred_file):
    """
    Compute the volumes of the muscle and fat from the given CT and prediction files.
    
    Args: 
        ct_file (sitk): CT image.
        pred_file (sitk): Prediction image.
    """
    spacing = ct_file.GetSpacing()
    # Get a numpy array from the label map image
    npy_label = sitk.GetArrayFromImage(pred_file)
    npy_ct = sitk.GetArrayFromImage(ct_file)

    # Get a list of the IDs
    label_ids = np.unique(npy_label)
    #print("Number of unique labels (0 is the background):")
    #print(len(np.unique(label_ids)))

    label_list = []
    volume_list = []
    average_hu_list = []

    for i in label_ids:

        #skip the bakcground label - we dont care!
        if i == 0:
            continue
        
        label_mask = npy_label == i

        voxel_count = np.count_nonzero(npy_label == i)
        #print("Muscle voxel count:"+str(v1_voxel_count))
        # Convert to mm^3
        volume = voxel_count*(spacing[0]*spacing[1]*spacing[2])
        #print("Muscle size mm^3: "+str(round(v1_size,2)))

        #average HU
        hu_vals = npy_ct[label_mask]
        average_hu = np.mean(hu_vals)

        label_list.append(i)
        volume_list.append(volume)
        average_hu_list.append(average_hu)

        # Create a DataFrame from the results
    df = pd.DataFrame({
        'label': label_list,
        'volume_mm3': volume_list,
        'average_hu': average_hu_list
    })

    return df

def compute_2d_smi(height):
    """
    Compute the skeletal muscle index (SMI) from the given height and area data.   
    """

    #TO DO
    smi = None

    return smi


def compute_3d_smi(height, df):
    """
    Compute the skeletal muscle index (SMI) from the given height and volume data.

    Args: 
        height (float): Height of the patient.
        df (pd.DataFrame): DataFrame containing the volume and HU data.
    """

    muscle_volume = df[df['label'] == 2]['volume_mm3'].values

    smi = muscle_volume/((height)*(height))
    
    return smi

ct_path = r'xxx'
pred_path = r'xxx'

ct_files = [os.path.join(ct_path, f) for f in os.listdir(ct_path) if f.endswith('.nii.gz')]
pred_files = [os.path.join(pred_path, f) for f in os.listdir(pred_path) if f.endswith('.nii.gz')]

gt_dict = {os.path.basename(gt_file): gt_file for gt_file in ct_files}
for pred_file in tqdm(pred_files, desc="Processing Files"):
    pred_id = os.path.basename(pred_file)  # Get just the filename, assuming the file name is the patient UID
    if pred_id in gt_dict:
        gt_file = gt_dict[pred_id]
    try:
        height = get_height(pred_file)
        volume_hu = compute_volumes(pred_file, gt_file)
        twod_smi = compute_2d_smi(height)
        smi_3d = compute_3d_smi(height, volume_hu)

    except:
        print('no matching ct file for pred file')

        #still to do: 
        #- save everything into one big df. 