#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm as std_tqdm
from functools import partial
from torchinfo import summary

from dataset import MultiModalASLDataset   # <-- import your dataset
from UltraLight_VM_UNet import UltraLight_VM_UNet       # your model
import torch.nn.functional as F

tqdm = partial(std_tqdm, dynamic_ncols=True)#!/usr/bin/env python3

def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Mamba on ASL data.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--asl-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--stack-modalities", action="store_true")
    parser.add_argument("--modalities", nargs="+", default=["s0", "dolp", "aop"])
    parser.add_argument("--model-name", type=str, default="asl-vm-unet")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of dataset for validation (default=0.2)")
    
    # --- For dataset specifications ---
    parser.add_argument("--s0", action="store_true", help="Compute s0.")
    parser.add_argument("--s1", action="store_true", help="Compute s1.")
    parser.add_argument("--s2", action="store_true", help="Compute s2.")
    parser.add_argument("--dolp", action="store_true", help="Compute DoLP.")
    parser.add_argument("--aop", action="store_true", help="Compute AoP.")
    parser.add_argument("--raw_scale", action="store_true", help="Apply min-max scaling on raw intensity data.")
    parser.add_argument("--min_max", action="store_true", help="Apply min-max normalization to s0.")
    parser.add_argument("--aop_rotate", action="store_true", help="Apply aop rotations to s0.")
    parser.add_argument("--enhanced_mixtures", action="store_true", help="Save 'mixtures' relating to enhanced s0.")
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

if __name__ == "__main__":
    # Read input from user + initialize placeholder vars
    args = parse_args()
    raw_scale = False
    min_max = False
    debug = False
    stack_modalities = False

    # Initialize list of target modalities
    modalities = []

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
        modalities.append("enhanced_s0")

    if args.aop_rotate:
        compute_enhanced = True
        aop_mode = 1
        modalities.append("enhanced_s0")

    if args.s0:
        modalities.append("s0")
    if args.s1:
        modalities.append("s1")
    if args.s2:
        modalities.append('s2')
    if args.dolp:
        modalities.append("dolp")
    if args.aop:
        modalities.append("aop")
    if args.enhanced_mixtures:
        modalities.append("s0e1")
        modalities.append("s0e2")

    if args.debug:
        debug = True

    # Create a dataset
    dataset = MultiModalASLDataset(
        data_dir,
        mask_dir,
        modalities=modalities,
        aop_mode= aop_mode,
        compute_enhanced=compute_enhanced,
        raw_scale=raw_scale,
        min_max=min_max,
        debug=debug,
        stack_modalities=stack_modalities
    )

    # Get first sample of dataset
    x, mask, valid = dataset[0] # x is [modalities, H, W]

    asl_files = list(Path(args.asl_dir).glob("*.asl.hdr"))
    mask_files = list(Path(args.mask_dir).glob("*.npz"))

    dataset = MultiModalASLDataset(
        asl_files,
        mask_files,
        modalities=tuple(args.modalities),
        stack_modalities=args.stack_modalities,
        debug=args.debug,
    )

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    sample_batch = next(iter(train_loader))
    if args.stack_modalities:
        sample_images, sample_masks, _ = sample_batch
    else:
        sample_dict, sample_masks, _ = sample_batch
        sample_images = torch.cat([sample_dict[m] for m in args.modalities], dim=1)

    print(f"[DEBUG] Sample input shape: {sample_images.shape}")
    print(f"[DEBUG] Sample mask shape: {sample_masks.shape}")

    model = UltraLight_VM_UNet(
        num_classes=args.num_classes,
        input_channels=sample_images.shape[1],
        c_list=[8, 16, 24, 32, 48, 64],
        split_att="fc",
        bridge=True,
    ).to(device)

    summary(model, input_size=tuple(sample_images.shape))

    print(f"Training on {device} with {len(dataset)} samples...")
    train_model(args, model, train_loader, device)

    torch.save(model.state_dict(), f"{args.model_name}.pt")
    print(f"Model saved to {args.model_name}.pt")
