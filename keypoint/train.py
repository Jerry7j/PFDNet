"""Train a PFDNet detector for the hip, knee, or ankle."""

import argparse
from pathlib import Path


def run_epoch(model, loader, device, optimizer=None):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(images), targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.shape[0]
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=("hip", "knee", "ankle"))
    parser.add_argument("--data-root", type=Path, help="Root of the regional sample and label directories")
    parser.add_argument("--images-dir", type=Path, help="Explicit training image directory")
    parser.add_argument("--labels-dir", type=Path, help="Explicit training label directory")
    parser.add_argument("--val-images-dir", type=Path, help="Explicit validation image directory")
    parser.add_argument("--val-labels-dir", type=Path, help="Explicit validation label directory")
    parser.add_argument("--point-count", type=int, help="Override the default number of landmarks for the region")
    parser.add_argument("--output", required=True, type=Path, help="Output model state dictionary")
    parser.add_argument("--backbone-weights", type=Path, help="Optional local Res2Net state dictionary")
    parser.add_argument("--epochs", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--channels", type=int, default=128, help="Decoder width")
    parser.add_argument("--device", help="PyTorch device, defaults to CUDA if available")
    parser.add_argument("--validate", action="store_true", help="Use the optional _val folders")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.channels) <= 0 or args.workers < 0:
        parser.error("epochs, batch size and channels must be positive; workers must be nonnegative")
    if args.point_count is not None and args.point_count <= 0:
        parser.error("--point-count must be positive")
    if (args.images_dir is None) != (args.labels_dir is None):
        parser.error("--images-dir and --labels-dir must be provided together")
    if (args.val_images_dir is None) != (args.val_labels_dir is None):
        parser.error("--val-images-dir and --val-labels-dir must be provided together")
    if args.data_root is None and args.images_dir is None:
        parser.error("provide --data-root or both --images-dir and --labels-dir")
    if args.validate and args.data_root is None and args.val_images_dir is None:
        parser.error("--validate needs --data-root or both validation directories")

    import torch
    from torch.utils.data import DataLoader

    from .data.mat_dataset import LandmarkDataset, POINT_COUNTS
    from .models.pfdnet import PFDNet

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    point_count = args.point_count or POINT_COUNTS[args.region]
    train = LandmarkDataset(args.data_root, args.region, images_dir=args.images_dir,
                            labels_dir=args.labels_dir, point_count=point_count)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = None
    if args.validate or args.val_images_dir is not None:
        val = LandmarkDataset(args.data_root, args.region, "val", images_dir=args.val_images_dir,
                              labels_dir=args.val_labels_dir, point_count=point_count)
        val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)

    model = PFDNet(channel=args.channels, num_keypoints=point_count)
    if args.backbone_weights:
        state = torch.load(args.backbone_weights, map_location="cpu", weights_only=True)
        model.resnet.load_state_dict(state)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, train_loader, device, optimizer)
        report = f"Epoch {epoch}/{args.epochs} train MSE {loss:.6f}"
        if val_loader is not None:
            report += f" val MSE {run_epoch(model, val_loader, device):.6f}"
        print(report, flush=True)
        torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
