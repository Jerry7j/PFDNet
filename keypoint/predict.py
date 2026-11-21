"""Predict landmark coordinates and optionally save overlays and pixel errors."""

import argparse
import json
from pathlib import Path

IMAGE_SUFFIXES = {".mat", ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=("hip", "knee", "ankle"))
    parser.add_argument("--input", required=True, type=Path, help="One image or a directory")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Prediction JSON")
    parser.add_argument("--labels-dir", type=Path, help="Optional directory of matching MAT or TXT labels")
    parser.add_argument("--point-count", type=int, help="Override the default number of landmarks for the region")
    parser.add_argument("--overlay-dir", type=Path, help="Optional directory for PNG overlays")
    parser.add_argument("--heatmap-dir", type=Path, help="Optional directory for per-landmark PNG heatmaps")
    parser.add_argument("--device", help="PyTorch device, defaults to CUDA if available")
    args = parser.parse_args()
    if args.point_count is not None and args.point_count <= 0:
        parser.error("--point-count must be positive")
    inputs = sorted(args.input.iterdir()) if args.input.is_dir() else [args.input]
    inputs = [p for p in inputs if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not inputs:
        parser.error("input contains no supported images")

    import cv2
    import numpy as np
    import torch

    from .data.mat_dataset import POINT_COUNTS, image_tensor, label_path_for, read_image, read_points
    from .models.pfdnet import PFDNet

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    channels = state["agg1.conv5.weight"].shape[1] // 3
    output_count = state["agg1.conv5.weight"].shape[0]
    point_count = args.point_count or POINT_COUNTS[args.region]
    # Older hip scripts used only the first of three output heatmaps.
    legacy_hip = args.point_count is None and args.region == "hip" and output_count == 3
    if output_count != point_count and not legacy_hip:
        parser.error(f"checkpoint has {output_count} outputs, expected {point_count} for {args.region}. "
                     "Set --point-count if training used a different landmark count.")
    model = PFDNet(channel=channels, num_keypoints=output_count)
    model.load_state_dict(state)
    model.to(device).eval()
    for directory in (args.overlay_dir, args.heatmap_dir):
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    def write_png(destination, image):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode: {destination}")
        destination.write_bytes(encoded.tobytes())

    records = []
    with torch.inference_mode():
        for path in inputs:
            rgb = read_image(path, args.region)
            heatmaps = model(image_tensor(rgb).unsqueeze(0).to(device))[0, :point_count]
            height, width = heatmaps.shape[-2:]
            indices = heatmaps.flatten(1).argmax(1).cpu().numpy()
            # Convert heatmap (column, row) indices back to source pixels.
            points = np.column_stack((indices % width * (rgb.shape[1] / width),
                                      indices // width * (rgb.shape[0] / height)))
            record = {"image": path.name, "points": points.tolist()}
            truth = None
            if args.labels_dir:
                label = label_path_for(path, args.labels_dir, args.region)
                truth = read_points(label, args.region, point_count)
                record["pixel_errors"] = np.linalg.norm(points - truth, axis=1).tolist()
            records.append(record)

            if args.overlay_dir:
                canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for point in points.round().astype(int):
                    cv2.circle(canvas, tuple(point), 3, (0, 0, 255), -1)
                if truth is not None:
                    for point in truth.round().astype(int):
                        cv2.circle(canvas, tuple(point), 3, (0, 255, 0), -1)
                write_png(args.overlay_dir / f"{path.name}.png", canvas)
            if args.heatmap_dir:
                for index, heatmap in enumerate(heatmaps.cpu().numpy()):
                    image = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    write_png(args.heatmap_dir / f"{path.name}.{index}.png", image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
