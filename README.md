# CTSegmentator 
Package for the automatic segmentation of CT images using AI.

## Installation Instructions

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

### Setting up model weights

Ensure the model_weights zip file is extracted directly into the top level CTSegmentator directory ie:

```
CTSegmentator/
 ├── ctsegmentator/
 │   ├── __init__.py
 │   ├── cli.py
 │   ├── ...
 ├── model_weights/
 │   ├── fold_0/
 │   │   │   ├── checkpoint_final.pth
 │   │   │   ├── debug.json
 │   │   │   ├── progress.png
 │   ├── fold_1\
 │   │   │   ├── ...
 │   ├── fold_2\
 │   │   │   ├── ...
 │   ├── fold_3\
 │   │   │   ├── ...
 │   ├── fold_4\
 │   │   │   ├── ...
 ...
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
python -m CTsegmentator.cli -i [path to input directory] -o [path to output directory] -f "nifti" -d ["cpu", "gpu"]

```