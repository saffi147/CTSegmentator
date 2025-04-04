# CTSegmentator 
Package for the automatic segmentation of CT images using AI.

## Installation Instructions

Clone the github repository by running the following: 

```
git clone https://github.com/saffi147/CTSegmentator.git
cd CTSegmentator
```

Create and activate the conda environment by running the following:

```
conda env create -f environment.yaml
conda activate CTSegmentator
```

To install all dependencies into the conda environment:

```
poetry install
```

To generate a new `poetry.lock` file (and install any new dependencies listed in `pyproject.toml`), run:
```
poetry update
```

## Dataset structure

The dataset MUST be set up with the following structure:

```
root/
 ├── Patient001/
 │   ├── DICOMDIR
 │   ├── DATA/
 │   │   ├── DICOM/
 │   │   │   ├── image1
 │   │   │   ├── image2
 ├── Patient002/
 │   ├── DICOMDIR
 │   ├── DATA/
 │   │   ├── DICOM/
 │   │   │   ├── image1
 │   │   │   ├── image2
...
```


## How to use: 

Run the following from the CTSegmentator directory: 
```
python -m ctsegmentator.cli -i [path to input directory] -p [github personal access token] -o [path to output directory] -f "dicom" -d ["cpu", "cuda"]

```


## Output:

CTSegmentator de-identifies input dicom and NIfTI CT scans, and saves the model derived segmentations into the following file structure: 

```
Output_folder/
 ├── Patient001/
 │   ├── ct_scan.nii.gz
 │   ├── mask.nii.gz
 ├── Patient002/
 │   ├── ct_scan.nii.gz
 │   ├── mask.nii.gz
 ...
 ├──dataset.json
 ├──plans.json
 ├──predict_from_raw_data_args.json
 ├──patient_id_mapping.csv
```

Json files in the output folder are copied over from the trained nnU-Net model. 

`patient_id_mapping.csv` contains the original naming convention of input files, mapped to the new patient id's assigned during processing. 