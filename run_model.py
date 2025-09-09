# ----------------------------------------------------------------------->
# Author: Connor Prikkel
# Applied Sensing Lab, University of Dayton
# 9/6/2025
# ----------------------------------------------------------------------->

#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm as std_tqdm
from functools import partial
from torchinfo import summary
import time
import numpy as np

from dataset import MultiModalASLDataset 
from UltraLight_VM_UNet import UltraLight_VM_UNet
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import matplotlib.pyplot as plt
from torchinfo import summary

import torch.nn.functional as F

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

tqdm = partial(std_tqdm, dynamic_ncols=True)#!/usr/bin/env python3

def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Mamba on ASL data.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--stack-modalities", action="store_true")
    parser.add_argument("--modalities", nargs="+", default=["s0", "dolp", "aop"],
                        help="List of modalities to use; " \
                        "options: s0, s1, s2, dolp, aop, enhanced_s0, s0e1, s0e2")
    parser.add_argument("--model-name", type=str, default="asl-vm-unet")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of dataset for validation (default=0.2)")
    parser.add_argument("--model-name", type=str, required=True)
    
    # --- For dataset specifications ---
    parser.add_argument("--raw_scale", action="store_true", help="Apply min-max scaling on raw intensity data.")
    parser.add_argument("--min_max", action="store_true", help="Apply min-max normalization to s0.")
    parser.add_argument("--aop_rotate", action="store_true", help="Apply aop rotations to s0.")
    parser.add_argument("--hist_shift", action="store_true", help="Apply histogram shifting to s0.")
    parser.add_argument("--stack_modalities", action="store_true", help="Stack output modalities -> represent as a tensor.")
    parser.add_argument("--debug", action="store_true", help="Add debug print statements to see intermediate output.")
    return parser.parse_args()


# --- Training and Testing Functions ---
def train_model(args, model, dataloader, device, val_dataloader=None, model_name="model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss().to(device)  # expects logits and class indices

    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        model.train()
        running_loss = 0.0
        total_correct = 0
        total_pixels = 0

        # Training loop
        # Compute loss + accuracy across every mini-batch
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{args.epochs}") as t:
            #for images, onehot_masks in dataloader:
            for batch in dataloader:
                if args.stack_modalities:
                    data, masks, _ = batch
                else:
                    modalities_dict, masks, _ = batch
                    data = torch.cat([modalities_dict[i] for i in args.modalities], dim=1)

                data = data.to(device, non_blocking=True).float()
                masks = masks.to(device, non_blocking=True).long()

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == masks).sum().item()
                total_pixels += torch.numel(masks)

                avg_loss = running_loss / (t.n + 1)
                train_accuracy = total_correct / total_pixels * 100.0

                # Update the progress bar with the current loss and accuracy
                t.set_postfix(loss=avg_loss, accuracy=train_accuracy)
                t.update(1)  # Increment the progress bar

        # Compute accuracy and loss by epoch
        epoch_avg_loss = running_loss / len(dataloader)
        epoch_train_accuracy = total_correct / total_pixels * 100.0

        # Validation loop
        val_loss = 0.0
        val_total_correct = 0
        val_total_pixels = 0

        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                #for images, onehot_masks in val_dataloader:
                for batch in val_dataloader:
                    if args.stack_modalities:
                        data, masks, _ = batch
                    else:
                        modalities_dict, masks, _ = batch
                        data = torch.cat([modalities_dict[i] for i in args.modalities], dim=1)
                    images = data.to(device, non_blocking=True)
                    masks = masks.to(
                        device, non_blocking=True
                    )

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    val_total_correct += (preds == masks).sum().item()
                    val_total_pixels += torch.numel(masks)

            avg_val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_total_correct / val_total_pixels * 100.0

            # Print metrics
            print(
                f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_train_accuracy:.2f}%, "
                f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

            # Save the model if it has the best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(
                    model.state_dict(), f"{model_name}_best_accuracy_model.pt"
                )
                print(
                    f"Saved model with best validation accuracy: {best_val_accuracy:.2f}%"
                )

            # Save the model if it has the lowest validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{model_name}_best_loss_model.pt")
                print(
                    f"Saved model with lowest validation loss: {best_val_loss:.4f}"
                )

        # If no validation set, just print training metrics
        else:
            print(
                f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
            )


def compute_iou(preds, targets, num_classes):
    iou = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        iou.append(intersection / union if union > 0 else 0)
    return iou


def test_model(args, model, dataloader, device, num_classes):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()  # Use the same loss function as during training
    test_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_inference_time = 0.0

    all_preds = []
    all_masks = []

    with torch.no_grad():  # Disable gradient calculation for testing
        #for images, onehot_masks in dataloader:
        for batch in dataloader:
            if args.stack_modalities:
                data, masks, _ = batch
            else:
                modalities_dict, masks, valid = batch
                data = torch.cat([modalities_dict[i] for i in args.modalities], dim=1)
            data = data.to(device).float()
            masks = masks.to(device).long()

            start_time = time.perf_counter()
            outputs = model(data)
            end_time = time.perf_counter()
            total_inference_time += end_time - start_time
            loss = criterion(outputs, masks)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

    avg_test_loss = test_loss / len(dataloader)
    test_accuracy = total_correct / total_pixels * 100.0

    # Flatten the lists of predictions and masks
    all_preds = np.concatenate(all_preds).flatten()
    all_masks = np.concatenate(all_masks).flatten()

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(
        all_masks, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_masks, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_masks, all_preds, average="weighted", zero_division=0)

    # Calculate IoU
    # iou = compute_iou(all_preds, all_masks, [1, 2])
    weighted_iou = jaccard_score(
        all_masks, all_preds, average="weighted", zero_division=0
    )

    # Compute per-class IoU
    iou = jaccard_score(
        all_masks, all_preds, average=None, zero_division=0
    )

    print(f"Test Loss: {avg_test_loss:.8f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {precision:.8f}, Recall: {recall:.8f}, F1 Score: {f1:.8f}")
    print(f"Mean IoU: {weighted_iou}")
    print(f"Per-class IoU: {compute_iou(all_preds, all_masks, num_classes)}")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")

    return avg_test_loss, test_accuracy, precision, recall, f1, iou

def main():
    # Take in command line arguments
    args = parse_args()

    # Flags from args
    raw_scale = args.raw_scale
    min_max = args.min_max
    stack_modalities = args.stack_modalities
    debug = args.debug

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------------->
    # DATASET PREPARATION
    # ----------------------------------------------------------------------->

    data_dir = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')
    mask_dir = Path('/home/connor/Thesis/updated_masks').glob('*.npz')

    if args.hist_shift:
        # Bool to track whether to compute enhanced param.
        compute_enhanced = True
        aop_mode = 2

    if args.aop_rotate:
        compute_enhanced = True
        aop_mode = 1

    if args.debug:
        debug = True

    # Create a dataset
    # Dataset shape: [batch_size, modalities, H, W]
    # Mask shape: [batch_size, H, W] (class indices)
    # valid_pixels shape: [batch_size, H, W] (boolean mask)
    dataset = MultiModalASLDataset(
        data_dir,
        mask_dir,
        modalities=tuple(args.modalities),
        aop_mode= aop_mode,
        compute_enhanced=compute_enhanced,
        raw_scale=raw_scale,
        min_max=min_max,
        debug=debug,
        stack_modalities=stack_modalities
    )

    # Get first sample of dataset
    # x, mask, valid = dataset[0] # x is [modalities, H, W]

    # Define dataset paritions
    num_total = len(dataset)
    num_train = int(0.7 * num_total)
    num_val = int(0.15 * num_total)
    num_test = num_total - num_train - num_val

    # Randomly split dataset into training and validation sets
    train_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test])

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load batches one at a time
    batch = next(iter(train_loader))

    # FOR DEBUG PRINT STATEMENTS
    if args.stack_modalities:
        data, masks, _ = batch
    else:
        data_dict, masks = batch
        data = torch.cat([data_dict[i] for i in args.modalities], dim=1)

    print(f"[DEBUG] Sample input shape: {data.shape}")
    print(f"[DEBUG] Sample mask shape: {masks.shape}")

    # ----------------------------------------------------------------------->
    # INITIALIZE MODEL + TRAINING
    # ----------------------------------------------------------------------->

    model_name = f"{args.model_name}-model-polar-mam_batchsize-{args.batch_size}_epochs-{args.epochs}"

    # Choose model
    model = UltraLight_VM_UNet(
            num_classes=10,
            input_channels=len(args.modalities),
            c_list=[8, 16, 24, 32, 48, 64],
            #c_list=[16, 24, 32, 48, 64, 96],
            split_att="fc",
            bridge=True,
        ).to(device)

    # If the model exists, load it; otherwise, train it
    if not Path(f"{model_name}.pt").exists():
        print(f"{model_name}.pt does not exist! Training...")        

        print(f"Training on device: {device}")

        # Train the model
        train_model(args, model, train_loader, device, val_loader)

        # Save the trained model
        torch.save(model.state_dict(), f"{model_name}.pt")

    else:
        print(f"Loading existing model: {model_name}_best_accuracy_model.pt")
        
        # Output model info
        summary(model, input_size=tuple(data.shape))
        model.load_state_dict(torch.load(f"{model_name}_best_accuracy_model.pt"))
        model.eval()
        
        test_model(model, test_loader, device, args.num_classes)

if __name__ == "__main__":
    main()
