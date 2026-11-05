"""Render results JSON files as a markdown table.

    python scripts/report.py results/pilot_baselines.json [results/vjepa2.json ...]
"""
import json
import sys


def fmt(x, pct=False):
    if x is None or x != x:
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.3f}"


def main() -> None:
    rows = []
    for path in sys.argv[1:]:
        for name, r in json.load(open(path))["results"].items():
            on = {o["target_fa"]: o for o in r["online"]}
            o5 = on.get(0.05, r["online"][0])
            ci = r["clip"].get("auroc_ci95")
            rows.append([
                name,
                fmt(r["clip"]["auroc"]) + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""),
                fmt(r["clip"]["ap"]), fmt(r["clip"]["auroc_open"]), fmt(r["clip"]["auroc_deconstruction"]),
                fmt(r["frame"]["frame_auroc"]),
                fmt(o5["fa_test"], True), fmt(o5["detect_rate"], True), fmt(o5["early_rate"], True),
                fmt(o5["within_1s"], True),
                "-" if o5["median_latency_s"] != o5["median_latency_s"] else f"{o5['median_latency_s']:.2f}",
                fmt(o5["clip_precision"], True),
            ])
    head = ["monitor", "clip AUROC [95% CI]", "clip AP", "AUROC open", "AUROC decon.", "frame AUROC",
            "test FA @5%", "detected", "alarm before onset", "alarm <=1 s after onset", "median latency (s)", "precision"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
