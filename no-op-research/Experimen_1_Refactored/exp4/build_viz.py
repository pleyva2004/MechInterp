"""Embed a viz-data JSON file into the exp4 viewer template.

Usage:
    uv run --offline python -m exp4.build_viz --data viz_data.json --out viz.html

The data is re-serialised compactly (no whitespace) and substituted for
``__VIZ_DATA__`` in ``viz_template.html``. ``</`` is escaped as ``<\\/`` so the
JSON cannot close its ``<script>`` tag early.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("viz_template.html")
PLACEHOLDER = "__VIZ_DATA__"


def build(data_path: Path, out_path: Path, template_path: Path = TEMPLATE, standalone: bool = False,
          keep: list[str] | None = None) -> int:
    """Write the page; return its size in bytes. standalone: wrap a fragment template (written for a host that adds
    the document skeleton, like claude.ai artifacts) into a complete document, e.g. for GitHub Pages. keep: the only
    per-dataset arrays to embed (e.g. the slider needs just 'country' and 'control')."""
    template = template_path.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"template must contain {PLACEHOLDER} exactly once")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if keep:
        data["data"] = {k: {a: v[a] for a in keep} for k, v in data["data"].items()}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace(PLACEHOLDER, payload)
    if standalone and not html.lstrip().lower().startswith("<!doctype"):
        # head and body tags are optional in HTML: the fragment's meta/title/style land in head, <main> opens body
        html = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + html.replace('<meta charset="utf-8">\n', "", 1) + "\n</html>\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return len(html.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="viz_data.json")
    parser.add_argument("--out", type=Path, required=True, help="output .html")
    parser.add_argument("--template", type=Path, default=TEMPLATE, help="page template (default: viz_template.html)")
    parser.add_argument("--standalone", action="store_true", help="wrap a fragment template into a full document")
    parser.add_argument("--keep", default=None, help="comma-separated per-dataset arrays to embed (default: all)")
    args = parser.parse_args()
    size = build(args.data, args.out, args.template, args.standalone, args.keep.split(",") if args.keep else None)
    print(f"wrote {args.out} ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
