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

    for path in sys.argv[1:]:
        d = json.load(open(path))
        R = d["results"]
        if not any("pre_onset" in r for r in R.values()):
            continue
        print("\nControls (5% target false-alarm rate):\n")
        head = ["monitor", "caught (calibrated)", "test FA", "caught at exactly 5% (test threshold)",
                "AUROC book", "AUROC box", "pre-onset AUROC [95% CI]", "early alarms: observed / chance (p)"]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for name, r in R.items():
            o5 = next(o for o in r["online"] if o["target_fa"] == 0.05)
            po, ew = r["pre_onset"], r["early_warning_null_5"]
            print("| " + " | ".join([
                name, fmt(o5["detect_rate"], True), fmt(o5["fa_test"], True), fmt(o5["oracle_detect_rate"], True),
                fmt(r["clip"].get("auroc_item_book")), fmt(r["clip"].get("auroc_item_box")),
                f"{po['auroc']:.3f} [{po['ci95'][0]:.2f}, {po['ci95'][1]:.2f}]",
                f"{ew['observed']}/{ew['n']} vs {ew['expected']:.1f} (p={ew['p_one_sided']:.2f})",
            ]) + " |")
        c = d.get("controls", {})
        if c.get("paired_auroc"):
            print("\nPaired clip-AUROC differences (bootstrap, same clips):\n")
            print("| comparison | difference | 95% CI | p |")
            print("|---|---|---|---|")
            for k, v in c["paired_auroc"].items():
                print(f"| {k} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p']:.3f} |")
        if c.get("clip_max_corr_with_velocity"):
            print("\nCorrelation of clip-max score with the velocity (motion) score, test clips:\n")
            print("| monitor | r |")
            print("|---|---|")
            for k, v in c["clip_max_corr_with_velocity"].items():
                print(f"| {k} | {v:.2f} |")


if __name__ == "__main__":
    main()
