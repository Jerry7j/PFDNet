"""Measure original or corrected landmarks and render research simulations."""

import argparse
import json
from math import isfinite
from pathlib import Path

from .geometry import fujisawa_angle, measure_angles, miniaci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for name in ("measure", "fujisawa", "miniaci"):
        commands.add_parser(name).add_argument("--points", required=True, type=Path)
    simulation = commands.choices["miniaci"]
    simulation.add_argument("--method", choices=("HTO", "DFO"), required=True)
    simulation.add_argument("--mode", choices=("MOpen", "MClose", "LOpen", "LClose"), required=True)
    simulation.add_argument("--image", type=Path, help="Optional single-leg crop")
    simulation.add_argument("--output", type=Path, help="Optional simulation PNG")
    ruler = commands.add_parser("ruler")
    ruler.add_argument("--image", required=True, type=Path)
    ruler.add_argument("--roi", nargs=4, required=True, type=int, metavar=("X", "Y", "W", "H"))
    ruler.add_argument("--span", type=float, help="Known segment length in the desired physical unit")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "miniaci" and bool(args.image) != bool(args.output):
        parser.error("--image and --output must be provided together")

    def load_image(path, cv2, np):
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            parser.error(f"Could not decode image: {path}")
        return image

    if args.command == "ruler":
        import cv2
        import numpy as np

        from .simulation import ruler_segments

        if args.span is not None and (not isfinite(args.span) or args.span <= 0):
            parser.error("span must be positive and finite")
        segments = ruler_segments(load_image(args.image, cv2, np), args.roi)
        result = {"segments": segments}
        if args.span:
            result["pixels_per_unit"] = [float(np.linalg.norm(np.subtract(b, a)) / args.span)
                                         for a, b in segments]
    else:
        data = json.loads(args.points.read_text(encoding="utf-8-sig"))
        hip, ankle = data["hip"], data["ankle"]
        if args.command == "measure":
            result = measure_angles(hip, data["knee_upper"], data["knee_lower"], ankle)
        elif args.command == "fujisawa":
            result = {"angle_degrees": fujisawa_angle(hip, data["target"], ankle)}
        else:
            adjusted, angle = miniaci(hip, data["target"], ankle, data["hinge"], args.method, args.mode)
            result = {"adjusted_center": adjusted, "angle_degrees": angle}
            if args.image:
                import cv2
                import numpy as np

                from .simulation import simulate

                canvas = simulate(load_image(args.image, cv2, np), data["hinge"], data["cut_end"],
                                  angle, args.method, args.mode)
                ok, encoded = cv2.imencode(".png", canvas)
                if not ok:
                    parser.error("Could not encode simulation image")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(encoded.tobytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
