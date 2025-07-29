import cv2
import numpy as np
import os
from glob import glob
import argparse
#python ml_segmentation/ml_filter.py -i ml_segmentation/merged_columns_Ml_Sample -o output/filtered_columns
def enhance_fingerprints(image_path, output_path):
    """
    Reads an image, applies a series of filters to enhance fingerprint ridges,
    and saves the result.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Local contrast enhancement for clearer ridges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Aggressive adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 9  # Larger window size helps in "empty" regions
    )

    # Line removal
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    all_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)

    no_lines = cv2.bitwise_and(thresh, cv2.bitwise_not(all_lines))

    # Fill in the cores of the fingerprints using closing
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled = cv2.morphologyEx(no_lines, cv2.MORPH_CLOSE, kernel_fill)

    # Final smoothing: remove isolated noise without eroding edges
    smoothed = cv2.medianBlur(filled, 3)

    cv2.imwrite(output_path, smoothed)
    return smoothed

def main(input_dir, output_dir):
    """
    Processes all images in the input directory and saves the filtered
    results to the output directory.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: '{output_dir}'")

    # Process all common image types
    image_paths = glob(os.path.join(input_dir, "*.jpg")) + \
                  glob(os.path.join(input_dir, "*.png")) + \
                  glob(os.path.join(input_dir, "*.bmp"))
                  
    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"Found {len(image_paths)} images to process.")

    for path in image_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, f"{filename}")
        # print(f"Processing {filename} -> {output_path}")
        enhance_fingerprints(path, output_path)

    print(f"\nProcessing complete. Filtered images saved to {output_dir}")

if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Apply fingerprint enhancement filter to a directory of images.")
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        required=True,
        help="Path to the directory containing the images to be filtered."
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        required=True,
        help="Path to the directory where filtered images will be saved."
    )
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)
