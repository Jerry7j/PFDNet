"""Landmark angle measurements and Fujisawa and Miniaci geometry."""

from math import atan2, degrees, dist, isclose, sqrt


def line_angle(a, b, c, d):
    """Legacy 0–180 degree convention, including the original point ordering."""
    if a == b or c == d:
        raise ValueError("Angle segments must have distinct endpoints")
    first = atan2(a[1] - b[1], a[0] - b[0])
    second = atan2(c[1] - d[1], c[0] - d[0])
    difference = abs(degrees(first - second))
    return min(difference, 360.0 - difference) % 180.0


def measure_angles(hip, knee_upper, knee_lower, ankle):
    """Each joint line is [center, first endpoint, second endpoint]."""
    upper_center, upper_a, upper_b = knee_upper
    lower_center, lower_a, lower_b = knee_lower
    ankle_center, ankle_a, ankle_b = ankle
    return {
        "mLDFA": line_angle(hip, upper_center, upper_a, upper_b),
        "JLCA": line_angle(upper_a, upper_b, lower_a, lower_b),
        "MPTA": line_angle(lower_a, lower_b, lower_center, ankle_center),
        "LDTA": line_angle(ankle_a, ankle_b, lower_center, ankle_center),
    }


def fujisawa_angle(hip, target, ankle):
    """Return the angle between the two segments through the target."""
    return line_angle(hip, target, target, ankle)


def miniaci(hip, target, ankle, hinge, method, mode):
    """Intersect the target axis with the circle centered at the hinge."""
    if method == "HTO":
        anchor, moving = hip, ankle
        use_smaller_x = mode in ("MOpen", "LClose")
    elif method == "DFO":
        anchor, moving = ankle, hip
        use_smaller_x = mode in ("MOpen", "LClose")
    else:
        raise ValueError("Method must be HTO or DFO")
    if mode not in ("MOpen", "MClose", "LOpen", "LClose"):
        raise ValueError("Unknown correction mode")

    length = dist(anchor, target)
    radius = dist(hinge, moving)
    if length == 0 or radius == 0:
        raise ValueError("The target axis and hinge radius must be nonzero")
    ux = (target[0] - anchor[0]) / length
    uy = (target[1] - anchor[1]) / length
    hx, hy = hinge[0] - anchor[0], hinge[1] - anchor[1]
    projection = hx * ux + hy * uy
    perpendicular = hx * uy - hy * ux
    offset_squared = radius ** 2 - perpendicular ** 2
    if offset_squared < 0 and not isclose(radius, abs(perpendicular)):
        raise ValueError("Target axis does not intersect the hinge circle")
    offset = sqrt(max(0.0, offset_squared))
    candidates = [(anchor[0] + t * ux, anchor[1] + t * uy)
                  for t in (projection - offset, projection + offset)]
    # Keep the branch used by each legacy method. A vertical axis ties on x.
    if abs(ux) < 1e-12:
        adjusted = min(candidates, key=lambda p: dist(p, moving))
    else:
        adjusted = (min if use_smaller_x else max)(candidates, key=lambda p: p[0])
    angle = line_angle(hinge, adjusted, hinge, moving)
    return adjusted, angle
