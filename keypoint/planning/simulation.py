"""Split and rotate regions of a radiograph for research visualization."""

import cv2
import numpy as np


def simulate(image, hinge, cut_end, angle, method, mode):
    """Rotate one side of the cut on a single-leg crop in its pixel coordinates."""
    if method not in ("HTO", "DFO") or mode not in ("MOpen", "MClose", "LOpen", "LClose"):
        raise ValueError("Unknown method or mode")
    if hinge == cut_end:
        raise ValueError("The cut needs two distinct points")
    height, width = image.shape[:2]
    y, x = np.ogrid[:height, :width]
    dx, dy = cut_end[0] - hinge[0], cut_end[1] - hinge[1]
    # Order the line from left to right so its positive half is below the cut.
    direction = 1 if dx > 0 or (dx == 0 and dy > 0) else -1
    below = direction * (dx * (y - hinge[1]) - dy * (x - hinge[0])) >= 0
    moving = below if method == "HTO" else ~below
    signed_angle = angle if mode in ("MOpen", "LClose") else -angle
    if method == "DFO":
        signed_angle = -signed_angle
    transform = cv2.getRotationMatrix2D(tuple(map(float, hinge)), -signed_angle, 1.0)

    moving_mask = moving.astype(np.uint8)
    rotated_mask = cv2.warpAffine(moving_mask, transform, (width, height),
                                  flags=cv2.INTER_NEAREST).astype(bool)
    # Warp the marked part and its mask separately, preserving true black pixels.
    moving_image = np.zeros_like(image)
    moving_image[moving] = image[moving]
    rotated = cv2.warpAffine(moving_image, transform, (width, height))
    result = np.zeros_like(image)
    if method == "HTO":
        result[rotated_mask] = rotated[rotated_mask]
        result[~moving] = image[~moving]
    else:
        result[~moving] = image[~moving]
        result[rotated_mask] = rotated[rotated_mask]
    return result


def ruler_segments(image, roi, min_width=20):
    """Return bright ruler line candidates in full-image pixel coordinates."""
    x, y, w, h = roi
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError("Ruler ROI must be inside the image")
    crop = image[y:y + h, x:x + w]
    bright = np.where(np.all(crop > 246, axis=2), 255, 0).astype(np.uint8)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, 0.8).detect(bright)[0]
    if lines is None:
        return []
    return [((float(x1) + x, float(y1) + y), (float(x2) + x, float(y2) + y))
            for x1, y1, x2, y2 in lines.reshape(-1, 4) if abs(x2 - x1) > min_width]
