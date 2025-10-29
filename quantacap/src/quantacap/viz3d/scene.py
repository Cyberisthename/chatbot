"""Render field series using optional plotting backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from quantacap.utils.optional_import import optional_import

from .fieldmap import FieldFrame


def export_scene(
    frames: Iterable[FieldFrame],
    *,
    out_prefix: str,
    video_format: str = "auto",
) -> dict:
    frames = list(frames)
    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(f"{out_prefix}_meta.json")
    meta = {
        "frames": len(frames),
        "format": None,
        "images": [],
    }

    mpl = None
    writer_cls = None
    if video_format in ("auto", "gif", "mp4"):
        try:
            mpl = optional_import("matplotlib", pip_name="matplotlib", purpose="render 3D field maps")
            mpl.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib import animation  # type: ignore

            writer_cls = animation.PillowWriter if (video_format in ("auto", "gif")) else animation.FFMpegWriter
        except Exception:  # pragma: no cover - optional dependency
            mpl = None
            writer_cls = None

    images = []
    if mpl is None:
        for frame in frames[:3]:
            np.save(Path(f"{out_prefix}_frame{frame.step}.npy"), frame.field)
            images.append(f"{out_prefix}_frame{frame.step}.npy")
        meta["format"] = "npy"
        meta["images"] = images
    else:  # pragma: no cover - requires matplotlib
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        writer_instance = writer_cls(fps=5) if writer_cls else None
        animation_suffix = "gif" if writer_cls is None or writer_cls.__name__.lower().startswith("pillow") else "mp4"
        animation_path = Path(f"{out_prefix}.{animation_suffix}")
        if writer_instance:
            with writer_instance.saving(fig, str(animation_path), dpi=100):
                for frame in frames:
                    ax.clear()
                    slice_idx = frame.field.shape[2] // 2
                    plane = frame.field[:, :, slice_idx]
                    ax.imshow(plane, cmap="viridis")
                    ax.set_title(f"t={frame.step}")
                    writer_instance.grab_frame()
        else:
            for frame in frames[:3]:
                slice_idx = frame.field.shape[2] // 2
                plt.imshow(frame.field[:, :, slice_idx], cmap="viridis")
                image_path = Path(f"{out_prefix}_frame{frame.step}.png")
                plt.savefig(image_path)
                images.append(str(image_path))
        meta["format"] = animation_path.suffix.lstrip(".") if writer_instance else "png"
        meta["images"] = images if images else [str(animation_path)]
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return meta
