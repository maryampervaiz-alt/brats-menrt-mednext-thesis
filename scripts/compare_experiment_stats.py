from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired statistical comparison between two CV experiments")
    p.add_argument("--a-fold-csv", type=str, required=True)
    p.add_argument("--b-fold-csv", type=str, required=True)
    p.add_argument("--metric", type=str, default="val_dice")
    p.add_argument("--alternative", type=str, default="two-sided", choices=["two-sided", "greater", "less"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a = pd.read_csv(Path(args.a_fold_csv)).sort_values("fold")
    b = pd.read_csv(Path(args.b_fold_csv)).sort_values("fold")

    if args.metric not in a.columns or args.metric not in b.columns:
        raise KeyError(f"Metric `{args.metric}` missing in one of the files.")
    if len(a) != len(b):
        raise ValueError("Fold counts do not match.")

    x = a[args.metric].to_numpy()
    y = b[args.metric].to_numpy()

    stat, pval = wilcoxon(x, y, alternative=args.alternative, zero_method="wilcox")
    print(f"Metric: {args.metric}")
    print(f"A mean±std: {x.mean():.4f} ± {x.std(ddof=1):.4f}")
    print(f"B mean±std: {y.mean():.4f} ± {y.std(ddof=1):.4f}")
    print(f"Wilcoxon statistic: {stat:.4f}")
    print(f"p-value ({args.alternative}): {pval:.6f}")


if __name__ == "__main__":
    main()

