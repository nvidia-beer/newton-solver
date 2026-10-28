"""Visualize the terminal velocity-fit example's measured optimization history."""

import numpy as np
from PIL import Image, ImageDraw


def record_terminal(name, arguments, frames, width, height):
    if name != "diffsim_optimize":
        raise ValueError(f"No terminal visualization for {name}")
    from newton_solver.examples.example_diffsim_optimize import create_parser, optimize_velocity

    args = create_parser().parse_args(arguments)
    result = optimize_velocity(iterations=args.iterations, device=args.device)
    if not np.isfinite(result["final_loss_m2"]) or result["final_loss_m2"] >= result["initial_loss_m2"]:
        raise RuntimeError("The velocity fit did not reduce its loss")
    losses = np.asarray(result["history_m2"])
    for frame in range(frames):
        count = min(len(losses), 1 + frame * len(losses) // max(1, frames * 3 // 4))
        picture = Image.new("RGB", (width, height), (245, 245, 242))
        draw = ImageDraw.Draw(picture)
        left, right, top, bottom = 80, width - 60, 90, height - 110
        draw.text((left, 55), "Initial velocity optimization | terminal position error", fill=(30, 35, 42))
        draw.line((left, top, left, bottom, right, bottom), fill=(75, 80, 87), width=2)
        log_values = np.log10(np.maximum(losses, 1e-12))
        low, high = -12, max(0, float(log_values.max()))
        points = [
            (
                left + i * (right - left) / (len(losses) - 1),
                bottom - (log_values[i] - low) / (high - low) * (bottom - top),
            )
            for i in range(count)
        ]
        if len(points) > 1:
            draw.line(points, fill=(32, 116, 168), width=4)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(32, 116, 168))
        draw.text(
            (left, bottom + 20), f"Gradient updates: {count - 1}   Loss: {losses[count - 1]:.3g} m^2", fill=(30, 35, 42)
        )
        draw.text(
            (left, bottom + 45),
            "Vertical axis: log10 squared terminal-position error; horizontal axis: update",
            fill=(50, 55, 62),
        )
        draw.text(
            (left, height - 35), "Actual Warp autodiff results, plotted as an optimization animation", fill=(65, 70, 77)
        )
        yield np.asarray(picture)
