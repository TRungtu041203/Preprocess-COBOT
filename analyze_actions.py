#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------- utils ----------------

RIGHT_HAND = list(range(0, 21))
LEFT_HAND  = list(range(21, 42))

def to_LJ3(arr: np.ndarray) -> np.ndarray:
    # fitted (3, L, 48, 1) -> (L, 48, 3)
    if arr.ndim == 4 and arr.shape[0] == 3 and arr.shape[-1] == 1:
        return np.transpose(arr, (1, 2, 0, 3))[..., 0].astype(np.float32)
    # raw (L, 48, 3)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.float32, copy=False)
    # batch (N, 3, L, 48, 1) -> pick first
    if arr.ndim == 5 and arr.shape[1] == 3 and arr.shape[-1] == 1:
        return np.transpose(arr[0], (1, 2, 0, 3))[..., 0].astype(np.float32)
    raise ValueError(f"Unexpected shape {arr.shape}")

def joint_valid_mask(data: np.ndarray, treat_zero_as_missing: bool, zero_eps: float) -> np.ndarray:
    """
    Returns M: (L, 48) True if joint is valid.
    Missing if any coord is non-finite; optionally also if all |coord|<=zero_eps.
    """
    M = np.isfinite(data).all(axis=2)
    if treat_zero_as_missing:
        M &= ~(np.abs(data) <= zero_eps).all(axis=2)
    return M

def _run_lengths(x: np.ndarray) -> list:
    """Return lengths of contiguous runs where x == True."""
    if x.size == 0:
        return []
    lens = []
    t = 0
    T = x.shape[0]
    while t < T:
        if x[t]:
            s = t
            while t < T and x[t]:
                t += 1
            lens.append(t - s)
        else:
            t += 1
    return lens

# --------------- scanning ----------------

def scan_actions(actions_root: Path, treat_zero_as_missing: bool, zero_eps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      per-sample dataframe
      gap-length dataframe with columns: action, sample, hand, kind ('all','any'), gap_len
    """
    rows = []
    gap_rows = []
    action_dirs = sorted([p for p in actions_root.iterdir() if p.is_dir()])
    for adir in tqdm(action_dirs, desc="Scanning actions"):
        for npy in sorted(adir.glob("*.npy")):
            try:
                arr = np.load(npy)
                data = to_LJ3(arr)  # (L, 48, 3)
            except Exception as e:
                print(f"Skip {npy}: {e}")
                continue

            L = int(data.shape[0])

            # ----- missing masks -----
            M = joint_valid_mask(data, treat_zero_as_missing, zero_eps)  # (L,48)

            # “whole-hand missing”: no joint observed in that hand at that frame
            R_all_miss = ~M[:, RIGHT_HAND].any(axis=1)
            L_all_miss = ~M[:, LEFT_HAND].any(axis=1)

            # “any-missing”: at least one joint missing in that hand at that frame
            R_any_miss = ~M[:, RIGHT_HAND].all(axis=1)
            L_any_miss = ~M[:, LEFT_HAND].all(axis=1)

            # gap lengths (contiguous True runs)
            gaps_R_all = _run_lengths(R_all_miss)
            gaps_L_all = _run_lengths(L_all_miss)
            gaps_R_any = _run_lengths(R_any_miss)
            gaps_L_any = _run_lengths(L_any_miss)
            
            # Debug: Show gap counts for this sample
            total_gaps = len(gaps_R_all) + len(gaps_L_all) + len(gaps_R_any) + len(gaps_L_any)
            if total_gaps > 0:
                print(f"    📁 {npy.stem}: {total_gaps} gaps (R-all:{len(gaps_R_all)}, L-all:{len(gaps_L_all)}, R-any:{len(gaps_R_any)}, L-any:{len(gaps_L_any)})")

            # aggregate per-sample stats
            missing_total = int((~M).sum())
            total_elems   = M.size
            missing_pct   = (missing_total / total_elems) if total_elems else 0.0
            per_frame_missing = (~M).reshape(L, -1).sum(axis=1)
            avg_missing_per_frame = float(per_frame_missing.mean()) if L > 0 else 0.0

            rows.append({
                "action": adir.name,
                "sample": npy.stem,
                "length": L,
                "missing_total": missing_total,
                "missing_pct": missing_pct,
                "avg_missing_per_frame": avg_missing_per_frame,
                "num_gaps_R_all": len(gaps_R_all),
                "num_gaps_L_all": len(gaps_L_all),
                "num_gaps_R_any": len(gaps_R_any),
                "num_gaps_L_any": len(gaps_L_any),
                "p90_R_all": np.percentile(gaps_R_all, 90) if gaps_R_all else 0,
                "p90_L_all": np.percentile(gaps_L_all, 90) if gaps_L_all else 0,
                "p90_R_any": np.percentile(gaps_R_any, 90) if gaps_R_any else 0,
                "p90_L_any": np.percentile(gaps_L_any, 90) if gaps_L_any else 0,
            })

            # expand into gap-rows for plotting
            for g in gaps_R_all:
                gap_rows.append({"action": adir.name, "sample": npy.stem, "hand": "R", "kind": "all", "gap_len": g})
            for g in gaps_L_all:
                gap_rows.append({"action": adir.name, "sample": npy.stem, "hand": "L", "kind": "all", "gap_len": g})
            for g in gaps_R_any:
                gap_rows.append({"action": adir.name, "sample": npy.stem, "hand": "R", "kind": "any", "gap_len": g})
            for g in gaps_L_any:
                gap_rows.append({"action": adir.name, "sample": npy.stem, "hand": "L", "kind": "any", "gap_len": g})

    return pd.DataFrame(rows), pd.DataFrame(gap_rows)

# --------------- existing plots ---------------

def plot_samples_per_action(df: pd.DataFrame, out_path: Path):
    cnt = df.groupby("action")["sample"].count().sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=cnt.index, y=cnt.values, color="#4C78A8")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Num samples")
    plt.title("Samples per action")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_length_box(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="action", y="length", color="#72B7B2")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Length (frames)")
    plt.title("Action length distribution (boxplot)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_avg_missing_per_frame(df: pd.DataFrame, out_path: Path):
    m = df.groupby("action")["avg_missing_per_frame"].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=m.index, y=m.values, color="#E45756")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Avg missing elements per frame")
    plt.title("Average missing per frame by action")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_missing_pct(df: pd.DataFrame, out_path: Path):
    m = df.groupby("action")["missing_pct"].mean().sort_values(ascending=False) * 100.0
    plt.figure(figsize=(12, 5))
    sns.barplot(x=m.index, y=m.values, color="#F58518")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Missing (%)")
    plt.title("Average percentage missing by action")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --------------- NEW: gap-length visuals ---------------

def _compute_bins(gaps: list[int]) -> np.ndarray:
    if not gaps:
        return np.arange(1, 6)
    gmax = max(gaps)
    gmax = int(min(max(6, gmax), 120))  # cap for readability
    return np.arange(1, gmax + 1)

def plot_gap_hist_and_cdf(gaps_df: pd.DataFrame, kind: str, out_prefix: Path, title_suffix: str):
    """
    kind: 'all' (whole-hand missing) or 'any' (some joint missing) per hand
    """
    sub = gaps_df[gaps_df["kind"] == kind]
    if sub.empty:
        print(f"⚠️  No gaps found for kind='{kind}' - skipping plots")
        return
    
    print(f"📊 Creating gap plots for kind='{kind}' with {len(sub)} gaps")

    # overall + per hand
    for label, dfk in [("both_hands", sub), ("right", sub[sub["hand"]=="R"]), ("left", sub[sub["hand"]=="L"])]:
        if dfk.empty:
            continue
        gaps = dfk["gap_len"].astype(int).tolist()
        bins = _compute_bins(gaps)

        # Histogram
        plt.figure(figsize=(10, 4))
        sns.histplot(gaps, bins=bins, color="#4C78A8")
        plt.xlabel("Gap length (frames)")
        plt.ylabel("Count")
        plt.title(f"Gap length histogram ({kind}) — {label} {title_suffix}")
        plt.tight_layout()
        plt.savefig(out_prefix.parent / f"{out_prefix.stem}_{kind}_{label}_hist.png")
        plt.close()

        # CDF
        arr = np.sort(np.array(gaps))
        y = np.arange(1, len(arr)+1) / max(1, len(arr))
        p90 = np.percentile(arr, 90) if len(arr) else 0
        p95 = np.percentile(arr, 95) if len(arr) else 0

        plt.figure(figsize=(10, 4))
        plt.plot(arr, y, lw=2)
        if len(arr):
            plt.axvline(p90, color="#E45756", ls="--", label=f"P90 ≈ {int(round(p90))}")
            plt.axvline(p95, color="#72B7B2", ls="--", label=f"P95 ≈ {int(round(p95))}")
        plt.ylim(0, 1)
        plt.xlabel("Gap length (frames)")
        plt.ylabel("CDF")
        plt.title(f"Gap length CDF ({kind}) — {label} {title_suffix}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix.parent / f"{out_prefix.stem}_{kind}_{label}_cdf.png")
        plt.close()

def plot_gap_bins_per_action(gaps_df: pd.DataFrame, kind: str, out_path: Path):
    """
    Stacked bars by action over bins: 1, 2, 3–5, 6–8, 9–12, 13+
    """
    sub = gaps_df[gaps_df["kind"] == kind]
    if sub.empty:
        print(f"⚠️  No gaps found for kind='{kind}' - skipping bin plots")
        return
    
    print(f"📊 Creating gap bin plots for kind='{kind}' with {len(sub)} gaps")

    def bin_name(g):
        if g == 1: return "1"
        if g == 2: return "2"
        if 3 <= g <= 5: return "3–5"
        if 6 <= g <= 8: return "6–8"
        if 9 <= g <= 12: return "9–12"
        return "13+"

    tmp = sub.copy()
    tmp["bin"] = tmp["gap_len"].astype(int).map(bin_name)

    tab = (tmp.groupby(["action", "bin"])["gap_len"]
           .count().unstack(fill_value=0))
    tab = tab.reindex(columns=["1","2","3–5","6–8","9–12","13+"], fill_value=0)

    # normalize rows to percentages
    tab_pct = tab.div(tab.sum(axis=1).replace(0,1), axis=0) * 100.0
    tab_pct = tab_pct.sort_index()

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(tab_pct))
    palette = ["#4C78A8","#72B7B2","#F58518","#E45756","#54A24B","#B279A2"]
    for i, col in enumerate(tab_pct.columns):
        plt.bar(tab_pct.index, tab_pct[col], bottom=bottom, label=col, color=palette[i % len(palette)])
        bottom += tab_pct[col].values
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Gap distribution (%)")
    plt.title(f"Gap length distribution by action ({kind})")
    plt.legend(title="Gap bin", ncols=3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --------------- summaries ----------------

def write_summary(df: pd.DataFrame, out_csv: Path):
    if df.empty:
        pd.DataFrame(columns=[
            "action", "num_samples", "length_mean", "length_median",
            "missing_total_mean", "missing_pct_mean", "avg_missing_per_frame_mean",
            "p90_R_all_mean","p90_L_all_mean","p90_R_any_mean","p90_L_any_mean"
        ]).to_csv(out_csv, index=False)
        return

    g = df.groupby("action")
    summary = pd.DataFrame({
        "num_samples": g["sample"].count(),
        "length_mean": g["length"].mean(),
        "length_median": g["length"].median(),
        "missing_total_mean": g["missing_total"].mean(),
        "missing_pct_mean": g["missing_pct"].mean() * 100.0,
        "avg_missing_per_frame_mean": g["avg_missing_per_frame"].mean(),
        "p90_R_all_mean": g["p90_R_all"].mean(),
        "p90_L_all_mean": g["p90_L_all"].mean(),
        "p90_R_any_mean": g["p90_R_any"].mean(),
        "p90_L_any_mean": g["p90_L_any"].mean(),
    }).sort_values("num_samples", ascending=False).reset_index()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

# --------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions_root", type=Path, required=True, help="Path to actions_raw root")
    ap.add_argument("--out_dir", type=Path, default=Path("actions_report"))
    ap.add_argument("--treat_zero_as_missing", action="store_true",
                    help="Count values with |value|<=zero_eps as missing (in addition to NaNs)")
    ap.add_argument("--zero_eps", type=float, default=0.0,
                    help="Threshold for zero-missing if enabled")
    ap.add_argument("--suggest_percentile", type=float, default=90.0,
                    help="Percentile of gap length to suggest as max_gap (e.g., 90 or 95)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Scanning actions for gaps and missing data...")
    df, gaps = scan_actions(args.actions_root, args.treat_zero_as_missing, args.zero_eps)
    
    print(f"✅ Found {len(df)} samples with stats")
    print(f"✅ Found {len(gaps)} gap records")
    
    if df.empty:
        print("❌ No sample data found!")
        return
    
    if gaps.empty:
        print("❌ No gaps found! Check your data and mask settings.")
        return
    
    # Save data
    df.to_csv(args.out_dir / "per_sample_stats.csv", index=False)
    gaps.to_csv(args.out_dir / "gap_lengths.csv", index=False)
    print("💾 Data saved to CSV files")

    write_summary(df, args.out_dir / "summary_by_action.csv")
    if not df.empty:
        plot_samples_per_action(df, args.out_dir / "samples_per_action.png")
        plot_length_box(df, args.out_dir / "length_boxplot.png")
        plot_avg_missing_per_frame(df, args.out_dir / "avg_missing_per_frame.png")
        plot_missing_pct(df, args.out_dir / "missing_pct.png")

    # New: gap visuals
    if not gaps.empty:
        plot_gap_hist_and_cdf(gaps, kind="all", out_prefix=args.out_dir / "gaplen", title_suffix="(whole-hand missing)")
        plot_gap_hist_and_cdf(gaps, kind="any", out_prefix=args.out_dir / "gaplen", title_suffix="(any joint missing)")
        plot_gap_bins_per_action(gaps, kind="all", out_path=args.out_dir / "gaplen_by_action_all.png")
        plot_gap_bins_per_action(gaps, kind="any", out_path=args.out_dir / "gaplen_by_action_any.png")

        # Suggested max_gap from chosen percentile on whole-hand gaps, per hand
        g_all_R = gaps[(gaps["kind"]=="all") & (gaps["hand"]=="R")]["gap_len"].values
        g_all_L = gaps[(gaps["kind"]=="all") & (gaps["hand"]=="L")]["gap_len"].values

        def pctl(x, p):
            return int(np.percentile(x, args.suggest_percentile)) if x.size else 0

        sug_R = pctl(g_all_R, args.suggest_percentile)
        sug_L = pctl(g_all_L, args.suggest_percentile)
        # conservative choice: max across hands, clamp to a reasonable ceiling
        suggested = int(np.clip(max(sug_R, sug_L), 1, 12))

        print("\n=== GAP LENGTH SUGGESTION ===")
        print(f"Percentile used: P{int(args.suggest_percentile)} of whole-hand gaps")
        print(f"Right hand: P{int(args.suggest_percentile)} = {sug_R} frames")
        print(f"Left  hand: P{int(args.suggest_percentile)} = {sug_L} frames")
        print(f"--> Suggested max_gap for Step 1: {suggested} frames (use 3–7 typically; adjust to your FPS/motion)")
        
        # Additional gap statistics
        print(f"\n=== DETAILED GAP STATISTICS ===")
        print(f"Total gaps found: {len(gaps):,}")
        if not gaps.empty:
            print(f"Gap length range: {gaps['gap_len'].min()} - {gaps['gap_len'].max()} frames")
            print(f"Mean gap length: {gaps['gap_len'].mean():.2f} frames")
            print(f"90th percentile: {gaps['gap_len'].quantile(0.9):.1f} frames")
            print(f"95th percentile: {gaps['gap_len'].quantile(0.95):.1f} frames")

if __name__ == "__main__":
    sns.set_context("talk")
    sns.set_style("whitegrid")
    main()
