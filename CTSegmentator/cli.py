# this is the command line interface 


########### come back to this 

# import argparse

# from CTSegmentator.read_dicom import read_dicom_image
# from CTsegmentator.segmentation import run_segmentation

# def main():
#     parser = argparse.ArgumentParser(description="CT Auto-Segmentation Tool")
#     parser.add_argument("--input_dir", required=True, help="Directory containing CT files")
#     parser.add_argument("--file_format", choices=["dicom", "nifti"], required=True, help="Input file format")
#     parser.add_argument("--output_dir", required=True, help="Directory to save segmentation results")
    
#     args = parser.parse_args()
    
#     if args.file_format == "dicom":
#         images = read_dicom_image(args.input_dir)
#     elif args.file_format == "nifti":
#         pass
#     else:
#         raise ValueError("Unsupported file format")
    
#     run_segmentation(images, args.output_dir)

# if __name__ == "__main__":
#     main()