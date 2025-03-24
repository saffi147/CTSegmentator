# CTSegmentator 
Package for the automatic segmentation of CT images using AI.

## Installation Instructions

Create and activate the conda environment by running the following:

```
conda env create -f environment.yaml
conda activate CT_segmentator
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

The dataset MUST be set up following:


## How to use: 

Run the following from the CTSegmentator directory: 
```
python -m CTsegmentator.cli -i [path to input directory] -o [path to output directory] -f "nifti" -d "cpu"

```