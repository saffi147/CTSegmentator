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
python -m ctsegmentator.cli -i [path to input directory] -p [github personal access token] -o [path to output directory] -f "dicom" -d "cpu"

```