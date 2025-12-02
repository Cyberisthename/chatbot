from __future__ import annotations

from pathlib import Path

import pytest

from quantacap.viz3d.fieldmap import build_field_series
from quantacap.viz3d.scene import export_scene


@pytest.mark.parametrize("field", ["amplitude", "phase", "entropy"])
def test_build_field_series(field: str, tmp_path: Path) -> None:
    frames = list(build_field_series(source="adapter:missing", field=field, grid=(4, 4, 4), steps=3))
    assert len(frames) == 3
    meta = export_scene(frames, out_prefix=str(tmp_path / "compmap"), video_format="auto")
    assert meta["frames"] == 3
    assert Path(meta["images"][0]).exists()
