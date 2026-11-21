"""Convert text landmarks to LabelMe JSON or update its Fujisawa point."""

import argparse
import json
from ast import literal_eval
from math import isfinite
from pathlib import Path

LABELS = ("Rotate_center", "Surgical_point", "Fujisawa_point")


def read_text_points(path):
    """Read a UTF-8 file of (label, (x, y)) tuples in file order.

    Skip blank lines and discard the labels. Return a list of [x, y] float
    pairs. Invalid or nonfinite coordinates raise ValueError with the line number.
    """
    points = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            _, (x, y) = literal_eval(line)
            point = [float(x), float(y)]
            if not all(map(isfinite, point)):
                raise ValueError("non-finite coordinate")
        except (SyntaxError, ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{path}:{number}: expected (label, (x, y))") from error
        points.append(point)
    return points


def main():
    """Write a LabelMe JSON document from the command-line inputs.

    Use three text rows in LABELS order for a new document or one row to
    replace the Fujisawa point in existing shapes. Reference the image by path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--existing", type=Path,
                        help="Retain existing shapes and replace Fujisawa_point from one text row")
    args = parser.parse_args()
    points = read_text_points(args.points)
    if args.existing:
        document = json.loads(args.existing.read_text(encoding="utf-8-sig"))
        shapes = document["shapes"]
        matches = [shape for shape in shapes if shape["label"] == LABELS[2]]
        if len(matches) != 1 or len(points) != 1:
            parser.error("updating requires one Fujisawa_point shape and one text row")
        matches[0]["points"] = points
    else:
        if len(points) != len(LABELS):
            parser.error("expected three text rows in this order: " + ", ".join(LABELS))
        shapes = [{"label": label, "points": [point], "group_id": None,
                   "shape_type": "point", "flags": {}} for label, point in zip(LABELS, points)]

    from PIL import Image

    with Image.open(args.image) as image:
        width, height = image.size
    # LabelMe loads the external image via imagePath. Do not embed its bytes.
    data = {"version": "5.0.1", "flags": {}, "shapes": shapes,
            "imagePath": args.image.name, "imageData": None, "imageHeight": height, "imageWidth": width}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Relate the input image to the output JSON directory, even if they differ.
    from os.path import relpath

    try:
        data["imagePath"] = Path(relpath(args.image, args.output.parent)).as_posix()
    except ValueError:  # Windows paths on different drives cannot be made relative.
        data["imagePath"] = args.image.resolve().as_posix()
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
