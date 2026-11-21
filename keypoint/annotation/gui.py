"""Annotate landmarks in a folder of images with OpenCV."""

import argparse
from ast import literal_eval
from pathlib import Path

WINDOW_NAME = "Key Point Label Tool"
IMAGE_TYPES = {".bmp", ".jpg", ".jpeg", ".png"}
ANNOTATION_DEFAULT = {"Object": (255, 0, 0)}
ARROWS = {2490368: "up", 2621440: "down", 2424832: "left", 2555904: "right",
          65362: "up", 65364: "down", 65361: "left", 65363: "right"}


def run(directory):
    """Open the OpenCV landmark editor for supported images in a directory.

    Resume at the first image without an annotation. Save named pixel
    coordinates to a TXT file beside each image when leaving it.
    """
    import cv2
    import numpy as np

    images = sorted(path for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_TYPES)
    if not images:
        raise ValueError(f"No supported images in {directory}")

    color_file = directory / "label.labels"
    colors = ANNOTATION_DEFAULT
    if color_file.is_file():
        colors = dict(literal_eval(line) for line in color_file.read_text(encoding="utf-8-sig").splitlines()
                      if line.strip()) or ANNOTATION_DEFAULT
    labels = list(colors)
    active_label = 0
    points = []

    def click(event, x, y, flags, userdata):
        """Add the active label on a left click or undo the last point on a right click."""
        if event == cv2.EVENT_LBUTTONDOWN and y < image.shape[0]:
            points.append((labels[active_label], (x, y)))
        elif event == cv2.EVENT_RBUTTONUP and points:
            points.pop()

    def label_path(image_path):
        """Return the path to the TXT annotation beside the given image."""
        return image_path.with_suffix(".txt")

    # Resume at the first image without a sidecar annotation.
    index = next((i for i, path in enumerate(images) if not label_path(path).exists()), 0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            path = images[index]
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image: {path}")
            sidecar = label_path(path)
            points = []
            if sidecar.is_file():
                points = [literal_eval(line) for line in sidecar.read_text(encoding="utf-8-sig").splitlines()
                          if line.strip()]
            cv2.setMouseCallback(WINDOW_NAME, click)

            while True:
                canvas = cv2.copyMakeBorder(image, 0, 22, 0, 0, cv2.BORDER_CONSTANT, value=(192, 192, 192))
                for label, point in points:
                    color = colors.get(label, (192, 192, 192))
                    cv2.circle(canvas, point, 2, color, -1)
                    if label:
                        cv2.putText(canvas, label, point, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
                status = f"{index + 1}/{len(images)} | {path.name} | {labels[active_label]}"
                cv2.putText(canvas, status, (3, image.shape[0] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 1)
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKeyEx(35)
                action = ARROWS.get(key)
                if action == "up":
                    active_label = max(0, active_label - 1)
                elif action == "down":
                    active_label = min(len(labels) - 1, active_label + 1)
                elif (action == "left" and index > 0) or (action == "right" and index + 1 < len(images)):
                    index += -1 if action == "left" else 1
                    break
                if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

            # Save edits before advancing or closing the window.
            if points:
                sidecar.write_text("".join(f"{point!r}\n" for point in points), encoding="utf-8")
            elif sidecar.exists():
                sidecar.unlink()
            if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return
    finally:
        cv2.destroyAllWindows()


def main():
    """Read the image directory from the command line or a folder picker and launch the editor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="?", type=Path, help="Image directory, or omit for folder picker")
    args = parser.parse_args()
    directory = args.images
    if directory is None:
        from tkinter.filedialog import askdirectory

        selected = askdirectory(title="Select an image directory")
        if not selected:
            return
        directory = Path(selected)
    run(directory)


if __name__ == "__main__":
    main()
