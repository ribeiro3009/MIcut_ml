import os
import glob
import cv2
from src.features.merge_column import merge_and_rotate_fingerprints
from collections import defaultdict

# --- Configuration ---
INPUT_DIR = "C:\Ml_dir\Ml_Sample" 
# Set the output directory where the final merged columns will be saved.
OUTPUT_DIR = "C:\Ml_dir\merged_columns_Ml_Sample"
def create_fingerprint_columns():
    """
    Main function to find, group, sort, and merge fingerprint images into columns.
    This is the primary script to generate data for the ML annotation task.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Find all images and group them by their unique ID
    all_image_paths = glob.glob(os.path.join(INPUT_DIR, "*.bmp"))
    images_by_id = defaultdict(list)

    for path in all_image_paths:
        filename = os.path.basename(path)
        try:
            # Assumes filename format: 'ID_dedoX.bmp'
            img_id = filename.split('_')[0]
            images_by_id[img_id].append(path)
        except IndexError:
            print(f"Warning: File {filename} does not match 'ID_dedoX.bmp' format. Skipping.")
            continue

    if not images_by_id:
        print(f"Error: No valid BMP images found in {INPUT_DIR}.")
        return

    # 2. Process each person's set of fingerprints
    for img_id, paths in images_by_id.items():
        print(f"--- Processing ID: {img_id} ---")
        try:
            # IMPORTANT: Sort paths numerically based on the finger number (dedoX)
            paths.sort(key=lambda p: int(os.path.basename(p).split('_dedo')[1].split('.')[0]))
        except (IndexError, ValueError):
            print(f"Warning: Could not sort files for ID {img_id}. Check file naming. Skipping.")
            continue

        # 3. Split into two groups: hand1 (fingers 1-5) and hand2 (fingers 6-10)
        hand1_paths = [p for p in paths if int(os.path.basename(p).split('_dedo')[1].split('.')[0]) <= 5]
        hand2_paths = [p for p in paths if int(os.path.basename(p).split('_dedo')[1].split('.')[0]) > 5]

        # 4. Process and save the column for each hand
        if hand1_paths:
            process_and_save_hand(img_id, hand1_paths, "hand1")
        else:
            print(f"Info: No images found for hand1 for ID {img_id}.")

        if hand2_paths:
            process_and_save_hand(img_id, hand2_paths, "hand2")
        else:
            print(f"Info: No images found for hand2 for ID {img_id}.")

def process_and_save_hand(img_id, hand_paths, hand_name):
    """
    Takes a list of paths for a specific hand, merges them, and saves the result.
    """
    # The merge function can handle variable numbers of fingers
    print(f"Processing {hand_name} with {len(hand_paths)} images...")
    
    # Call the corrected merge function
    column_image = merge_and_rotate_fingerprints(hand_paths)
    
    if column_image is not None:
        output_filename = f"column_{img_id}_{hand_name}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, column_image)
        print(f"Successfully created column: {output_path}")
    else:
        print(f"Error: Failed to create column for ID {img_id}, {hand_name}.")

if __name__ == "__main__":
    create_fingerprint_columns()