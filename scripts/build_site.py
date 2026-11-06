"""Build the GitHub Pages site (docs/index.html) from site/template.html.

    python scripts/make_figures.py --out results/figures.json   # needs the pilot data
    python scripts/build_site.py                                  # needs only the JSON

results/figures.json holds a few example frames from ARMBench (CC BY 4.0,
Amazon Robotics), credited on the page.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Can a world model tell when a robot pick goes wrong? Online failure monitors on Amazon's ARMBench robot videos.">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=ROOT / "site/template.html")
    p.add_argument("--figures", default=ROOT / "results/figures.json")
    p.add_argument("--out", default=ROOT / "docs/index.html")
    a = p.parse_args()
    body = Path(a.template).read_text().replace("__FIGDATA__", Path(a.figures).read_text())
    # the template starts with <title>/<meta>/<link>/<style>; everything before the first <div> belongs in <head>
    cut = body.index('<div class="wrap">')
    html = HEAD + body[:cut] + "</head>\n<body>\n" + body[cut:] + "\n</body>\n</html>\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    (out.parent / ".nojekyll").touch()  # serve docs/ as plain files
    print("wrote", out, f"{len(html) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
