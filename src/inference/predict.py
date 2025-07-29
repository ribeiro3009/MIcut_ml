

import os
import cv2
import torch
import numpy as np
import argparse
import time
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Constants ---
# Adjust batch size based on your GPU memory. 4 or 8 is a good starting point.
BATCH_SIZE = 4

# --- Model Definition (Copied from train_detector.py) ---
def get_model(num_classes):
    """Loads a pre-trained Faster R-CNN model and replaces the classifier head."""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- Utility to create batches ---
def create_batches(items, batch_size):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# --- Main Prediction Logic ---
def run_prediction(model_path, input_dir, output_dir, threshold):
    """Loads the model and runs prediction on all images in the input directory."""
    # --- Setup ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU. This will be slow.")

    num_classes = 2
    model = get_model(num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"\nFound {len(image_files)} images to process...")
    start_time = time.time()

    # --- Batch Prediction Loop ---
    for batch_files in create_batches(image_files, BATCH_SIZE):
        images_tensors = []
        original_images = []
        
        for filename in batch_files:
            image_path = os.path.join(input_dir, filename)
            try:
                with open(image_path, 'rb') as f:
                    img_np = np.frombuffer(f.read(), np.uint8)
                    original_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if original_image is None: raise IOError("Failed to decode image")
                
                original_images.append(original_image)
                img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                images_tensors.append(F.to_tensor(img_rgb).to(device))
            except Exception as e:
                print(f"Could not read or decode {filename}: {e}. Skipping.")
                continue
        
        if not images_tensors: continue

        with torch.no_grad():
            predictions = model(images_tensors)

        # --- Process and Save Results for the Batch ---
        for i, pred in enumerate(predictions):
            filename = batch_files[i]
            original_image = original_images[i]

            for j in range(len(pred['scores'])):
                score = pred['scores'][j].item()
                if score > threshold:
                    box = pred['boxes'][j].detach().cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{score:.2f}"
                    cv2.putText(original_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            output_path = os.path.join(output_dir, f"predicted_{filename}")
            cv2.imwrite(output_path, original_image)
            print(f"  - Processed '{filename}' -> Saved result to '{output_path}'")

    end_time = time.time()
    print(f"\nPrediction complete in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect fingerprints in column images using a trained Faster R-CNN model.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained model (.pth) file.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to the directory containing column images to test.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to the directory where marked images will be saved.")
    parser.add_argument("-t", "--threshold", type=float, default=0.8, help="Confidence threshold for displaying detections (default: 0.8).")
    parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE, help=f"Number of images to process in parallel (default: {BATCH_SIZE}).")

    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    
    run_prediction(args.model_path, args.input_dir, args.output_dir, args.threshold)


