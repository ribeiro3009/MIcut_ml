
import os
import json
import cv2
import numpy as np
import torch
import torch.utils.data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
import time

# --- Utility Functions ---
def collate_fn(batch):
    return tuple(zip(*batch))

# --- 1. Dataset Class ---
class FingerprintDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, annotations, transforms=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Fix for Unicode paths on Windows
            with open(img_path, 'rb') as f:
                img_np = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if img is None:
                raise IOError(f"Image not found or failed to load: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # imdecode loads as BGR
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample or skip
            return None, None

        filename = os.path.basename(img_path)
        annotation = self.annotations[filename]
        rects = annotation['rectangles']

        boxes = []
        for r in rects:
            x1, y1 = r[0]
            x2, y2 = r[1]
            # Ensure box has a positive area
            if x1 >= x2 or y1 >= y2:
                continue
            boxes.append([x1, y1, x2, y2])

        if not boxes: # Skip if no valid boxes
            return None, None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes, "labels": labels, "image_id": image_id,
            "area": area, "iscrowd": iscrowd
        }

        img = F.to_tensor(img)

        # TODO: Add data augmentation transforms here if needed

        return img, target

    def __len__(self):
        return len(self.image_paths)

# --- 2. Model Definition ---
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 3. Training & Validation Functions ---
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = [img.to(device) for img, tgt in zip(images, targets) if img is not None]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]

        if not images: continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    return total_loss / len(data_loader)

def validate(model, data_loader, device):
    # To get validation loss, we keep the model in training mode
    # but use torch.no_grad() to prevent gradient calculations.
    model.train()
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img, tgt in zip(images, targets) if img is not None]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]

            if not images: continue

            # When in train() mode, the model returns a loss dictionary
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            
    return total_loss / len(data_loader)

# --- 4. Main Execution ---
def main():
    # --- Config ---
    base_dir = "C:/Ml_dir"
    image_dir = os.path.join(base_dir, "merged_columns_Ml_Sample")
    annotations_path = os.path.join(base_dir, "annotations.json")
    output_model_path = os.path.join(base_dir, "model/best_detector_model_v2.pth")
    
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 2 # Use a small batch size due to large image sizes

    # --- Data Loading ---
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    all_filenames = list(annotations.keys())
    person_ids = sorted(list(set([name.split('_')[1] for name in all_filenames])))
    
    train_ids, test_ids = train_test_split(person_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    def get_files_for_ids(ids):
        return [f for f in all_filenames if any(f"_{pid}_" in f for pid in ids)]

    train_files = get_files_for_ids(train_ids)
    val_files = get_files_for_ids(val_ids)

    train_image_paths = [os.path.join(image_dir, f) for f in train_files]
    val_image_paths = [os.path.join(image_dir, f) for f in val_files]

    print(f"Training samples: {len(train_image_paths)}")
    print(f"Validation samples: {len(val_image_paths)}")

    # --- Datasets and DataLoaders ---
    dataset_train = FingerprintDataset(train_image_paths, annotations)
    dataset_val = FingerprintDataset(val_image_paths, annotations)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # --- Model, Optimizer, Scheduler ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Our model has 2 classes: background and fingerprint
    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # --- Training Loop ---
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device)
        val_loss = validate(model, data_loader_val, device)
        
        lr_scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"  -> New best model saved to {output_model_path}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    # Ensure the output directory exists
    output_dir = "C:/Users/luizs/Área de Trabalho/Montreal/MIcut_biometry/ml_segmentation/model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main()
