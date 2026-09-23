import json
from pathlib import Path


def save_aligned(fig, axes, base: str | Path, panel_ids: list[str]) -> None:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    boxes = [axis.get_position() for axis in axes]
    rows = {}
    for panel, box in zip(panel_ids, boxes, strict=True):
        key = round(box.y0, 3)
        rows.setdefault(key, []).append((panel, box))
    for members in rows.values():
        heights = [box.height for _, box in members]
        if max(heights) - min(heights) > 1.5 / 72 / fig.get_figheight():
            raise ValueError("Panel alignment exceeds 1.5 pt")
    payload = {panel: {"x0": box.x0, "y0": box.y0, "width": box.width, "height": box.height} for panel, box in zip(panel_ids, boxes, strict=True)}
    (base.parent / f"{base.name}.alignment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for suffix in ("svg", "pdf", "png", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(base.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", **options)
