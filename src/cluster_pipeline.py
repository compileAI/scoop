"""cluster_pipeline.py

A *single-file* production version of the research notebook that

1. streams / slides a JSON article dump through the clustering pipeline;
2. logs a concise set of *unsupervised* quality & performance metrics in
   real-time (because we do **not** expect gold labels in production); and
3. can be run from the command-line *or* imported as a module.

The heavylifting is still done by the original `simulate()` & helper
functions — they are included verbatim at the bottom (search for
"==== CORE PIPELINE ==== " if you want to dive into the maths).

Usage (example)
---------------
```bash
python cluster_pipeline.py \
    --data scraped-articles/preprocessed.json \
    --window-size 14 --slide-size 7 --num-windows 10 \
    --min-articles 5 --T 4 \
    --keyword-score tfidf \
    --time-aware --theme-aware \
    --log-file cluster.log
```

The script prints coloured *human-readable* logs to `stdout` **and** writes a
CSV-ready one-liner to the optional `--log-file`.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from clustering_core import simulate

import warnings
warnings.filterwarnings("ignore")

###############################################################################
#                              -- CLI & LOGGING --                           #
###############################################################################

FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=FMT, datefmt=DATEFMT)
LOGGER = logging.getLogger("cluster-pipeline")

# helper: Gini coefficient on an array of positive numbers

def gini(x: np.ndarray | list[float]) -> float:
    """Return Gini coefficient in [0, 1]. 0 = equal; 1 = all mass in one bin."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.isclose(arr.sum(), 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    # normalised by mean
    g = (n + 1.0 - 2 * (cum / cum[-1]).sum()) / n
    return float(np.round(g, 3))

# helper: metrics for monitoring live performance

def compute_live_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Return unsupervised quality metrics for *one* full run."""

    # drop the special outlier label (-1) when counting clusters
    clustered = df[df["cluster"] >= 0]

    # ––– size-based stats –––––––––––––––––––––––––––––––––––––––––––
    sizes = clustered.groupby("cluster").size().values
    g_size = gini(sizes)
    outlier_pct = round(1 - len(clustered) / len(df), 3) if len(df) else 0.0

    # ––– coherence proxy ––––––––––––––––––––––––––––––––––––––––––––
    mean_sim = clustered.groupby("cluster")["sim"].mean().mean()

    return {
        "clusters":      int(clustered["cluster"].nunique()),
        "mean_sim":      round(mean_sim, 3),
        "gini_size":     g_size,
        "outlier_pct":   outlier_pct,
    }

def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run `simulate()` once and return the merged *metrics* dictionary."""

    LOGGER.info("Starting pipeline – %s", cfg)
    t0 = time.time()

    # ------------------------ heavy work -----------------------------
    all_window, *_ = simulate(
        cfg["data"],
        cfg["window_size"], cfg["slide_size"], cfg["num_windows"],
        cfg["min_articles"], cfg["N"], cfg["T"],
        cfg["keyword_score"], False,  # verbose=False inside simulate
        False,                        # story_label (no gold)
        time_aware=cfg["time_aware"],
        theme_aware=cfg["theme_aware"],
    )

    wall_s = round(time.time() - t0, 1)
    live = compute_live_metrics(all_window)

    LOGGER.info("Finished in %ss – clusters=%d  mean_sim=%.3f  "
                "gini=%.3f  outliers=%.1f%%",
                wall_s, live["clusters"], live["mean_sim"],
                live["gini_size"], live["outlier_pct"] * 100)

    return {**cfg, **live, "wall_s": wall_s}

def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Streaming news clustering")

    p.add_argument("--data", required=True,
                   help="Path to pre-processed JSON article dump")

    # core hyper-parameters ---------------------------------------------------
    p.add_argument("--window-size", type=int, default=14)
    p.add_argument("--slide-size",  type=int, default=7)
    p.add_argument("--num-windows", type=int, default=10)
    p.add_argument("--min-articles", type=int, default=5)
    p.add_argument("--T", type=int, default=4,
                   help="Temperature for soft assignment (higher = sharper)")
    p.add_argument("--N", type=int, default=10,
                   help="Top-N tokens used in theme scoring")

    p.add_argument("--keyword-score", choices=["tfidf", "bm25"],
                   default="tfidf")

    p.add_argument("--time-aware",  action="store_true",
                   help="Use exponential time decay for centres & TF")
    p.add_argument("--theme-aware", action="store_true",
                   help="Weight sentences by cluster keywords")

    # logging ---------------------------------------------------------------
    p.add_argument("--log-file", default=None,
                   help="Append CSV metrics to this file as we go")

    args = p.parse_args()

    return {
        "data": pathlib.Path(args.data).expanduser().as_posix(),
        "window_size": args.window_size,
        "slide_size":  args.slide_size,
        "num_windows": args.num_windows,
        "min_articles": args.min_articles,
        "T": args.T,
        "N": args.N,
        "keyword_score": args.keyword_score,
        "time_aware": args.time_aware,
        "theme_aware": args.theme_aware,
        "log_file": args.log_file,
    }

def main():
    cfg = parse_args()
    metrics = run_pipeline(cfg)

    # ---------------------------------------------------------------- logging
    if cfg["log_file"]:
        log_path = pathlib.Path(cfg["log_file"]).expanduser()
        is_new = not log_path.exists()
        df_line = pd.DataFrame([metrics])
        df_line.to_csv(log_path, mode="a", header=is_new, index=False)
        LOGGER.info("Appended metrics to %s", log_path)


if __name__ == "__main__":
    main()
