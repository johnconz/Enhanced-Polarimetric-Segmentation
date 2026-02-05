# ----------------------------------------------------------------------->
# Author: Connor Prikkel
# Applied Sensing Lab, University of Dayton
# 9/6/2025
# ----------------------------------------------------------------------->

#!/usr/bin/env python3

# --- Handle Imports ---
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
import matplotlib.pyplot as plt
from clearml import Task, Logger
import seaborn as sns
import random
import helper_functions as hf

import torch.nn.functional as F

from dataset import MultiModalASLDataset 
from UltraLight_VM_UNet import UltraLight_VM_UNet
#from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

# For evaluation metrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassConfusionMatrix,
)

# For LaTeX text in plots
plt.rcParams['text.usetex'] = True

# Initialize constants and random seed
RANDOM_SEED = 42
CLASS_NAMES = ["Background", "Circular Panels", "Cones", "Cylinders", "Pyramids", "Square Panels", "Vehicles", "Tables", "Cases", "Tents"]
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

tqdm = partial(std_tqdm, dynamic_ncols=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Mamba on ASL data.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--modalities", nargs="+", default=["s0", "dolp", "aop"],
                        help="List of modalities to use; " \
                        "options: s0, s1, s2, dolp, aop, enhanced_s0, shape_enhancement, shape_contrast_enhancement")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--logger", action="store_true", help="Use ClearML for logging experiment results.")
    parser.add_argument("--task", type=str, default="test", help="ClearML task name.")
    
    # --- For dataset specifications ---
    parser.add_argument("--raw-scale", action="store_true", help="Apply min-max scaling on raw intensity data.")
    parser.add_argument("--min-max", action="store_true", help="Apply min-max normalization to s0.")
    parser.add_argument("--aop-rotate", action="store_true", help="Apply aop rotations to s0.")
    parser.add_argument("--hist-shift", action="store_true", help="Apply histogram shifting to s0.")
    parser.add_argument("--stack-modalities", action="store_true", help="Stack output modalities -> represent as a tensor.")
    parser.add_argument("--debug", action="store_true", help="Add debug print statements to see intermediate output.")
    parser.add_argument("--visualize", action="store_true", help="Visualize cutmix applied to training masks and predicted/actual test masks.")
    parser.add_argument("--title", type=str, default="Experiment", help="Title for logging and figure naming.")
    return parser.parse_args()

def save_and_log_vector_plot(
    logger,
    title: str,
    series: str,
    xaxis: str,
    yaxis: str,
    values,
    labels,
    save_dir="figures"
):
    """
    Save a barplot of a vector metric as PNG and log it to ClearML.
    """
    #Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Plot locally with seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=labels, y=values.squeeze(), ax=ax, palette="viridis")
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save locally
    save_path = Path(save_dir) / f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved vector figure to {save_path}")

    # Log to ClearML
    logger.report_vector(
        title=title,
        series=series,
        xaxis=xaxis,
        yaxis=yaxis,
        values=values,
        labels=labels
    )
    logger.report_matplotlib_figure(title=title, series=f"{series} (PNG)", figure=fig)

    plt.close(fig)

# For dynamic class weighting
def compute_augmented_class_weights(dataset, num_classes=10, num_samples=5000):
    """
    Estimate class weights from dataset with augmentation (e.g., CutMix).
    Uses median frequency balancing.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        _, mask, _ = dataset[idx]  # sample with augmentation
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum().item()

    median_freq = torch.median(class_counts[class_counts > 0])
    weights = median_freq / (class_counts + 1e-6)
    weights = weights / weights.mean()  # normalize so mean = 1
    return weights


# --- Training and Testing Functions ---
def train_model(args, model, dataloader, device, val_dataloader=None, model_name="model", logger: Logger=None):
    
    # Activate cutmix for training ONLY
    if hasattr(dataloader.dataset, 'dataset'):  # if Subset (created from random_split)
        dataloader.dataset.dataset.cutmix_active = True
    else:
        dataloader.dataset.cutmix_active = True
    
    # USED FIXED WEIGHTS TO SAVE TIME FOR NOW
    print("[INFO] Using fixed precomputed class weights")
        # After median frequency balancing on full training set
        #class_weights = torch.tensor([
        #    0.0009364112047478557, 0.3661355674266815, 3.2632901668548584,
        #    2.6224074363708496, 2.659174680709839, 0.1016852855682373,
        #    0.01724742352962494, 0.6431813836097717, 0.3034757375717163,
        #    0.022466091439127922
        #], dtype=torch.float32)

    
    # After CutMix median frequency balancing
    class_weights = torch.tensor([0.0010026864474639297, 0.3847227096557617, 3.213834524154663,
        2.6751484870910645, 2.551990032196045, 0.10843880474567413, 
        0.019780773669481277, 0.6874544620513916, 0.33227434754371643, 
        0.02535284124314785], 
    dtype=torch.float32)

    print(f"[INFO] Using class weights: {class_weights.tolist()}")

    # Since using manual seed, can use fixed class weights determined from full training set
    # FOR INVERSE FREQ
    #class_weights = torch.tensor([9.364e-05, 0.037, 0.326, 0.262, 0.266, 0.010, 0.002, 0.064, 0.030, 0.002], dtype=torch.float32)

    # Pass class weights to penalize mistakes on small classes more
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_miou = 0.0  # track best mIoU instead of accuracy
    best_val_loss = float("inf")
    early_stop_counter = 0
    early_stop_patience = 3  # stop if no improvement in 3 validation checks
    early_stop_delta = 0.002 # minimum change to qualify as improvement

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_pixels = 0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{args.epochs}") as t:
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

                t.set_postfix(loss=avg_loss, accuracy=train_accuracy)
                t.update(1)

        # Log training metrics
        if logger:
            logger.report_scalar("train", "loss", iteration=epoch, value=avg_loss)
            logger.report_scalar("train", "accuracy", iteration=epoch, value=train_accuracy)

        epoch_avg_loss = running_loss / len(dataloader)

        # --- Validation Every Few Epochs ---
        if val_dataloader and epoch % 5 == 0:

            model.eval()
            val_loss = 0.0
            val_total_pixels = 0
            val_total_correct = 0

            # Initialize foreground metrics (ignore background class 0)
            val_iou_metric = MulticlassJaccardIndex(num_classes=args.num_classes, average="macro", ignore_index=0).to(device)
            val_f1_metric = MulticlassF1Score(num_classes=args.num_classes, average="macro", ignore_index=0).to(device)

            with torch.no_grad():
                for batch in val_dataloader:
                    if args.stack_modalities:
                        data, masks, _ = batch
                    else:
                        modalities_dict, masks, _ = batch
                        data = torch.cat([modalities_dict[i] for i in args.modalities], dim=1)

                    data = data.to(device, non_blocking=True).float()
                    masks = masks.to(device, non_blocking=True).long()

                    outputs = model(data)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    val_iou_metric.update(preds, masks)
                    val_f1_metric.update(preds, masks)

                    val_total_correct += (preds == masks).sum().item()
                    val_total_pixels += torch.numel(masks)

                    # Visualize predicted vs actual masks for first sample of batch of first epoch
                    if args.visualize and batch == 0 and epoch == 0:
                         pred_mask = preds[0]
                         true_mask = masks[0]
                         hf.visualize_masks(pred_mask, true_mask, title="Predicted vs. Actual Masks")

            avg_val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_total_correct / val_total_pixels * 100.0
            val_miou = val_iou_metric.compute().item()
            val_f1 = val_f1_metric.compute().item()

            if logger:
                logger.report_scalar("val", "loss", iteration=epoch, value=avg_val_loss)
                logger.report_scalar("val", "accuracy", iteration=epoch, value=val_accuracy)
                logger.report_scalar("val", "foreground_mIoU", iteration=epoch, value=val_miou)
                logger.report_scalar("val", "foreground_F1", iteration=epoch, value=val_f1)

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {epoch_avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                f"Val mIoU (no BG): {val_miou:.4f}, Val F1 (no BG): {val_f1:.4f}"
            )

            # --- Model Selection Based on Foreground mIoU ---
            if val_miou > best_val_miou + early_stop_delta:
                best_val_miou = val_miou
                early_stop_counter = 0  # reset counter
                torch.save(model.state_dict(), f"{model_name}-best-miou-model.pt")
                print(f"Saved new best model (foreground mIoU: {best_val_miou:.4f})")
            else:
                early_stop_counter += 1
                print(f"No significant improvement in mIoU. Early stop counter: {early_stop_counter}/{early_stop_patience}")

            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break


            # Still save based on lowest loss for reference
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{model_name}-best-loss-model.pt")

        else:
            print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")


# NOTE: Not used in favor of torchmetrics
# def compute_iou(preds, targets, num_classes):
#     iou = []
#     for cls in range(num_classes):
#         intersection = ((preds == cls) & (targets == cls)).sum().item()
#         union = ((preds == cls) | (targets == cls)).sum().item()
#         iou.append(intersection / union if union > 0 else 0)
#     return iou


def test_model(args, model, dataloader, device, logger: Logger=None, class_names=CLASS_NAMES):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()  # Use the same loss function as during training
    test_loss = 0.0
    #total_correct = 0
    #total_pixels = 0
    total_inference_time = 0.0

    #all_preds = []
    #all_masks = []
    
    # --- Use torchmetrics to compute metrics incrementally ---
    # Per-class metrics
    acc_metric = MulticlassAccuracy(num_classes=args.num_classes, average='micro').to(device)
    prec_metric = MulticlassPrecision(num_classes=args.num_classes, average=None).to(device) # per-class
    rec_metric = MulticlassRecall(num_classes=args.num_classes, average=None).to(device) # per-class
    f1_metric = MulticlassF1Score(num_classes=args.num_classes, average=None).to(device) # per-class
    iou_metric = MulticlassJaccardIndex(num_classes=args.num_classes, average=None).to(device) # per-class
    cm_metric = MulticlassConfusionMatrix(num_classes=args.num_classes, normalize="true").to(device) # columns sum to 1

    # Overall metrics
    mean_prec_metric = MulticlassPrecision(num_classes=args.num_classes, average='macro').to(device)
    mean_rec_metric = MulticlassRecall(num_classes=args.num_classes, average='macro').to(device)
    mean_f1_metric = MulticlassF1Score(num_classes=args.num_classes, average='macro').to(device)
    mean_iou_metric = MulticlassJaccardIndex(num_classes=args.num_classes, average='macro').to(device)

    with torch.no_grad():  # Disable gradient calculation for testing
        #for images, onehot_masks in dataloader:
        for batch in dataloader:
            if args.stack_modalities:
                data, masks, _ = batch
            else:
                modalities_dict, masks, _ = batch
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
            
            # NOTE: Storing all predictions leads to out-of-memory error
            # (Commented code is for manual metric computation)
            #all_preds.append(preds.cpu().numpy())
            #all_masks.append(masks.cpu().numpy())

            # Update torchmetrics
            acc_metric.update(preds, masks)
            prec_metric.update(preds, masks)
            rec_metric.update(preds, masks)
            f1_metric.update(preds, masks)
            iou_metric.update(preds, masks)
            cm_metric.update(preds, masks)

            mean_prec_metric.update(preds, masks)
            mean_rec_metric.update(preds, masks)
            mean_f1_metric.update(preds, masks)
            mean_iou_metric.update(preds, masks)

            #total_correct += (preds == masks).sum().item()
            #total_pixels += torch.numel(masks)

    #avg_test_loss = test_loss / len(dataloader)
    #test_accuracy = total_correct / total_pixels * 100.0

    # Flatten the lists of predictions and masks
    #all_preds = np.concatenate(all_preds).flatten()
    #all_masks = np.concatenate(all_masks).flatten()

    # Calculate Precision, Recall, and F1 Score
    #precision = precision_score(
    #    all_masks, all_preds, average="weighted", zero_division=0
    #)
    #recall = recall_score(all_masks, all_preds, average="weighted", zero_division=0)
    #f1 = f1_score(all_masks, all_preds, average="weighted", zero_division=0)

    # Calculate IoU
    # iou = compute_iou(all_preds, all_masks, [1, 2])
    #weighted_iou = jaccard_score(
    #    all_masks, all_preds, average="weighted", zero_division=0
    #)

    # Compute per-class IoU
    #iou = jaccard_score(
    #    all_masks, all_preds, average=None, zero_division=0
    #)

    # Finalize torchmetrics
    avg_test_loss = test_loss / len(dataloader)
    test_accuracy = acc_metric.compute().item() * 100.0
    precision = prec_metric.compute().cpu().numpy()
    recall = rec_metric.compute().cpu().numpy()
    f1 = f1_metric.compute().cpu().numpy()
    iou = iou_metric.compute().cpu().numpy()
    cm = cm_metric.compute().cpu().numpy()

    mean_precision = mean_prec_metric.compute().item()
    mean_recall = mean_rec_metric.compute().item()
    mean_f1 = mean_f1_metric.compute().item()
    mean_iou = mean_iou_metric.compute().item()

    # Print + return metrics
    print(f"Test Loss: {avg_test_loss:.8f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Mean Precision: {mean_precision:.8f}, Mean Recall: {mean_recall:.8f}, Mean F1 Score: {mean_f1:.8f}, Mean IoU: {mean_iou:.8f}")
    print(f"Per-class Precision: {precision}")
    print(f"Per-class Recall: {recall}")
    print(f"Per-class F1 Score: {f1}")
    print(f"Per-class IoU: {iou}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    #print(f"Per-class IoU: {compute_iou(all_preds, all_masks, args.num_classes)}")

    # Change to 2D array with one column per class
    f1 = f1.reshape(-1, 1)
    iou = iou.reshape(-1, 1)
    precision = precision.reshape(-1, 1)
    recall = recall.reshape(-1, 1)

    # Report final metrics
    if logger:
        logger.report_single_value(name="Test Loss", value=avg_test_loss)
        logger.report_single_value(name="Test Accuracy", value=test_accuracy)
        logger.report_single_value(name="Mean Precision", value=mean_precision)
        logger.report_single_value(name="Mean Recall", value=mean_recall)
        logger.report_single_value(name="Mean IoU", value=mean_iou)
        logger.report_single_value(name="Mean F1", value=mean_f1)
        save_and_log_vector_plot(logger, title=f"{args.title} Per-class F1", series="Evaluation Metrics", xaxis="Class", yaxis="F1", values=f1, labels=CLASS_NAMES)
        save_and_log_vector_plot(logger, title=f"{args.title} Per-class IoU", series="Evaluation Metrics", xaxis="Class", yaxis="IoU", values=iou, labels=CLASS_NAMES)
        save_and_log_vector_plot(logger, title=f"{args.title} Per-class Precision", series="Evaluation Metrics", xaxis="Class", yaxis="Precision", values=precision, labels=CLASS_NAMES)
        save_and_log_vector_plot(logger, title=f"{args.title} Per-class Recall", series="Evaluation Metrics", xaxis="Class", yaxis="Recall", values=recall, labels=CLASS_NAMES)


        # Confusion matrix as heatmap
        if class_names is None:
            class_names = [str(i) for i in range(args.num_classes)]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{args.title} Confusion Matrix")

        # Save as figure too
        plt.savefig(f"{args.title}_confusion_matrix.png", dpi=300, bbox_inches="tight")
        logger.report_matplotlib_figure(title="Test Confusion Matrix", series="Confusion Matrix", figure=fig)
        plt.close(fig)


    #return avg_test_loss, test_accuracy, precision, recall, f1, iou

    return {
        "loss": avg_test_loss,
        "accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "mean_iou": mean_iou,
        "confusion_matrix": cm,
    }

def main():
    # Take in command line arguments
    args = parse_args()

    # Flags from args
    raw_scale = args.raw_scale
    min_max = args.min_max
    stack_modalities = args.stack_modalities
    debug = args.debug

    if args.logger:
        # Initialize ClearML task
        task = Task.init(project_name="Thesis Experiments", task_name=args.task, auto_connect_frameworks=False)
        logger = task.get_logger()

    # Initialize placeholders for enhanced parameters
    aop_mode = 0 # 0 = none, 1 = rotate, 2 = hist shift
    compute_enhanced = False

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------------->
    # DATASET PREPARATION
    # ----------------------------------------------------------------------->

    data_dir = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')
    mask_dir = Path('/home/connor/Thesis/updated_masks').glob('*.npz')

    if args.hist_shift:
        compute_enhanced = True
        aop_mode = 2

    if args.aop_rotate:
        compute_enhanced = True
        aop_mode = 1

    if args.debug:
        debug = True

    # -------------------------
    # Build three datasets
    # -------------------------
    from dataset import CutMixSegmentation

    base_kwargs = dict(
        modalities=tuple(args.modalities),
        aop_mode=aop_mode,
        compute_enhanced=compute_enhanced,
        raw_scale=raw_scale,
        min_max=min_max,
        debug=debug,
        stack_modalities=stack_modalities,
        enable_disk_cache=False,
        enable_ram_cache=False,
    )

    full_dataset = MultiModalASLDataset(
        data_dir, mask_dir,
        cutmix_aug=None,  # initially disabled
        cutmix_active=False,
        visualize_cutmix=args.visualize,
        **base_kwargs
    )

    # Training dataset with CutMix
    rare_classes = [2, 3, 4]  # cones, cylinders, pyramids (adjust as needed)
    cutmix_aug = CutMixSegmentation(dataset=full_dataset, probability=0.5, rare_classes=rare_classes)

    full_dataset.cutmix_aug = cutmix_aug  # enable CutMix augmentation


    # -------------------------
    # Split indices consistently
    # -------------------------
    num_total = len(full_dataset)
    num_train = int(0.7 * num_total)
    num_val = int(0.15 * num_total)
    num_test = num_total - num_train - num_val

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [num_train, num_val, num_test],
    )

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    #val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)


    # Get sample batch for shape debugging
    batch = next(iter(train_loader))
    if args.stack_modalities:
        data, masks, _ = batch
    else:
        data_dict, masks, _ = batch
        data = torch.cat([data_dict[i] for i in args.modalities], dim=1)

    if args.debug:
        print(f"[DEBUG] Sample input shape: {data.shape}")
        print(f"[DEBUG] Sample mask shape: {masks.shape}")

    # ----------------------------------------------------------------------->
    # INITIALIZE MODEL + TRAINING
    # ----------------------------------------------------------------------->

    model_name = f"{args.model_name}-model-batchsize-{args.batch_size}-epochs-{args.epochs}"

    # Choose model
    model = UltraLight_VM_UNet(
            num_classes=10,
            input_channels=len(args.modalities),
            c_list=[8, 16, 24, 32, 48, 64],
            #c_list=[16, 24, 32, 48, 64, 96],
            split_att="fc",
            bridge=True,
        ).to(device)

    # If the model exists, load it; otherwise, train and load it
    if not Path(f"{model_name}.pt").exists():
        print(f"{model_name}.pt does not exist! Training...")        

        print(f"Training on device: {device}")

        # --- Enable CutMix for training, disable for validation ---
        train_loader.dataset.dataset.cutmix_active = True
        val_loader.dataset.dataset.cutmix_active = False

        # Train the model
        train_model(args, model, train_loader, device, val_loader, model_name=model_name, logger=logger)

        # Save the trained model
        torch.save(model.state_dict(), f"{model_name}.pt")

        print(f"Loading existing model: {model_name}-best-miou-model.pt")

        # --- Disable CutMix for testing ---
        test_loader.dataset.dataset.cutmix_active = False
        
        # Output model info
        summary(model, input_size=tuple(data.shape))
        model.load_state_dict(torch.load(f"{model_name}-best-miou-model.pt"))
        model.eval()
        
        test_model(args, model, test_loader, device, logger=logger)

    # Just test the model if alr exists
    else:
        print(f"Loading existing model: {model_name}-best-miou-model.pt")

        # --- Disable CutMix for testing ---
        test_loader.dataset.dataset.cutmix_active = False
        
        # Output model info
        summary(model, input_size=tuple(data.shape))
        model.load_state_dict(torch.load(f"{model_name}-best-miou-model.pt"))
        model.eval()
        
        test_model(args, model, test_loader, device, logger=logger)

if __name__ == "__main__":
    #print(torch.cuda.is_available())
    main()
