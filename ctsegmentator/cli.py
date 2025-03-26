import argparse
from ctsegmentator.python_api import ctsegmentator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT Auto-Segmentation Tool")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing CT files")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save segmentation results")
    parser.add_argument("-f", "--file_format", choices=["dicom", "nifti"], required=True, help="Input file format")
    parser.add_argument("-d", "--device", choices=["cpu", "gpu"], required = True, help = "Device to use for processing")
    args = parser.parse_args()

    ctsegmentator(ct_dir = args.input_dir, output_dir = args.output_dir, device = args.device, file_format= args.file_format)