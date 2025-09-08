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

from dataset import MultiModalASLDataset 
from UltraLight_VM_UNet import UltraLight_VM_UNet 
import torch.nn.functional as F

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


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, args):
    model.train()
    running_loss, total_correct, total_pixels = 0.0, 0, 0

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as t:
        for batch in dataloader:
            if args.stack_modalities:
                images, masks, valid = batch
            else:
                modalities_dict, masks, valid = batch
                images = torch.cat([modalities_dict[m] for m in args.modalities], dim=1)

            images, masks = images.to(device), masks.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

            avg_loss = running_loss / (t.n + 1)
            train_acc = total_correct / total_pixels * 100.0
            t.set_postfix(loss=avg_loss, accuracy=train_acc)
            t.update(1)

    return running_loss / len(dataloader), total_correct / total_pixels

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, epoch, args):
    model.eval()
    running_loss, total_correct, total_pixels = 0.0, 0, 0

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs} [Val]") as t:
        for batch in dataloader:
            if args.stack_modalities:
                images, masks, valid = batch
            else:
                modalities_dict, masks, valid = batch
                images = torch.cat([modalities_dict[m] for m in args.modalities], dim=1)

            images, masks = images.to(device), masks.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

            avg_loss = running_loss / (t.n + 1)
            val_acc = total_correct / total_pixels * 100.0
            t.set_postfix(loss=avg_loss, accuracy=val_acc)
            t.update(1)

    return running_loss / len(dataloader), total_correct / total_pixels


# ---------------- Training & Evaluation ----------------
def train_model(args, model, dataloader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(args.epochs):
        model.train()
        running_loss, total_correct, total_pixels = 0.0, 0, 0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}") as t:
            for batch in dataloader:
                if args.stack_modalities:
                    images, masks, valid = batch
                else:
                    modalities_dict, masks, valid = batch
                    images = torch.cat([modalities_dict[m] for m in args.modalities], dim=1)

                images, masks = images.to(device), masks.to(device)
                masks = masks.long()  # CrossEntropyLoss expects class indices

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == masks).sum().item()
                total_pixels += torch.numel(masks)

                avg_loss = running_loss / (t.n + 1)
                train_acc = total_correct / total_pixels * 100.0
                t.set_postfix(loss=avg_loss, accuracy=train_acc)
                t.update(1)

def main():
    # Take in command line arguments
    args = parse_args()
    raw_scale = False
    min_max = False
    debug = False
    stack_modalities = False

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------------->
    # DATASET PREPARATION
    # ----------------------------------------------------------------------->

    data_dir = Path('/home/connor/MATLAB/data').glob('*.asl.hdr')
    mask_dir = Path('/home/connor/Thesis/updated_masks').glob('*.npz')

    if args.raw_scale:
        raw_scale = True

    if args.min_max:
        min_max = True

    if args.stack_modalities:
        stack_modalities = True

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
    data_iter = iter(train_loader)
    data, masks, _ = next(data_iter)

    print(f"[DEBUG] Sample input shape: {data.shape}")
    print(f"[DEBUG] Sample mask shape: {masks.shape}")

    # ----------------------------------------------------------------------->
    # INITIALIZE MODEL + TRAINING
    # ----------------------------------------------------------------------->
    global model_name

    model_name = f"{args.model_name}-model-polar-mam_batchsize-{args.batch_size}_epochs-{args.epochs}"

    if not Path(f"{model_name}.pt").exists():
        print(f"{model_name}.pt does not exist! Training...")

        model = UltraLight_VM_UNet(
            num_classes=10,
            input_channels=len(args.modalities),
            c_list=[8, 16, 24, 32, 48, 64],
            #c_list=[16, 24, 32, 48, 64, 96],
            split_att="fc",
            bridge=True,
        ).to(device)

        print(f"Training on device: {device}")

        # train_model(args, model, train_loader, device, test_loader, local_rank)
        train_model(args, model, train_loader, device, test_loader)

        if True:  # local_rank == 0:
            torch.save(model.state_dict(), f"{model_name}.pt")

    else:
        model = UltraLight_VM_UNet(
            num_classes=10,
            input_channels=len(args.modalities),
            c_list=[8, 16, 24, 32, 48, 64],
            split_att="fc",
            bridge=True,
        ).to(device)
        #TODO: figure out model size
        #summary(
        #    model,
        #   input_size=(data.shape)
        #)
        model.load_state_dict(torch.load(f"{model_name}_best_accuracy_model.pt"))
        model.eval()
        test_model(model, test_loader, device, args.num_classes)


    summary(model, input_size=tuple(sample_images.shape))

if __name__ == "__main__":
    main()
