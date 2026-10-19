"""
Fall detection model trained on the SisFall dataset.

Usage (Colab or local):
    python train_fall_model.py /path/to/SisFall_dataset

Expected folder layout:
    SisFall_dataset/SA01/D01_SA01_R01.txt, F01_SA01_R01.txt, ...  (SA = young, SE = elderly)

Design choices (made so the model can later be compared on the shoe):
  * MMA8451Q accelerometer (+-8 g) and ITG3200 gyro clipped to +-500 deg/s,
    i.e. the same ranges as the MPU6050 setup in the shoe firmware.
  * Only ORIENTATION-INDEPENDENT magnitudes are used (|accel| in g, |gyro| in deg/s),
    because the dataset sensor is on the waist and yours is on the foot.
  * Resampled 200 Hz -> 50 Hz, 3 s windows.
  * Evaluation is split BY SUBJECT, never by random rows.
"""
import io
import os
import re
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.stats import skew, kurtosis
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay,
)

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/content/SisFall_dataset"
OUT_DIR = "."

FS_RAW, FS = 200, 50
DEC = FS_RAW // FS                 # 4
WIN = 3 * FS                       # 150 samples = 3 s
ADL_STEP = FS                      # regular ADL windows every 1 s
# Where the impact peak sits inside a window (0.75 s ... 2.25 s, every 0.25 s)
PEAK_OFFSETS = np.arange(int(0.75 * FS), int(2.25 * FS) + 1, FS // 4)

NAME_RE = re.compile(r"^([DF]\d{2})_(S[AE]\d{2})_R\d{2}\.txt$")


# ---------------------------------------------------------------- loading
def load_file(path):
    """Return (accel_magnitude_g, gyro_magnitude_dps) at 50 Hz, or None if unreadable."""
    try:
        with open(path) as f:
            txt = f.read().replace(";", "")
        raw = np.loadtxt(io.StringIO(txt), delimiter=",", ndmin=2)
        if raw.shape[1] < 9:
            return None
    except Exception:
        return None

    # [(2*range)/(2^resolution)] * bits   (from the SisFall readme)
    acc = raw[:, 6:9] * (2 * 8 / 2 ** 14)              # MMA8451Q, +-8 g, 14 bit
    gyr = raw[:, 3:6] * (2 * 2000 / 2 ** 16)           # ITG3200, +-2000 deg/s, 16 bit
    gyr = np.clip(gyr, -500, 500)                      # match shoe MPU6050 range

    a = np.linalg.norm(acc, axis=1)
    g = np.linalg.norm(gyr, axis=1)
    if len(a) < DEC * (WIN + 1):
        return None
    a = decimate(a, DEC, zero_phase=True).astype(np.float32)
    g = decimate(g, DEC, zero_phase=True).astype(np.float32)
    return a, g


# ---------------------------------------------------------------- features
def window_features(A, G):
    """A, G: (n_windows, WIN) arrays -> DataFrame of features."""
    f = {}
    for name, X in (("a", A), ("g", G)):
        f[f"{name}_mean"] = X.mean(1)
        f[f"{name}_std"] = X.std(1)
        f[f"{name}_min"] = X.min(1)
        f[f"{name}_max"] = X.max(1)
        f[f"{name}_rms"] = np.sqrt((X ** 2).mean(1))
        f[f"{name}_p25"], f[f"{name}_p75"] = np.percentile(X, [25, 75], axis=1)
        f[f"{name}_skew"] = skew(X, axis=1)
        f[f"{name}_kurt"] = kurtosis(X, axis=1)
        f[f"{name}_maxjerk"] = np.abs(np.diff(X, axis=1)).max(1) * FS

    # Fall pattern: free-fall dip -> impact -> stillness
    pk = A.argmax(1)
    idx = np.arange(A.shape[1])[None, :]
    settle = idx > (pk[:, None] + FS // 5)             # ignore 0.2 s of impact ringing
    before = np.where(idx < pk[:, None], A, np.nan)
    after = np.where(settle, A, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["pre_min"] = np.nanmin(before, axis=1)
        f["post_std"] = np.nanstd(after, axis=1)
        f["post_dev"] = np.nanmean(np.abs(after - 1.0), axis=1)
        f["post_gyro"] = np.nanmean(np.where(settle, G, np.nan), axis=1)
    f["freefall_frac"] = (A < 0.6).mean(1)
    f["impact_frac"] = (A > 2.0).mean(1)

    df = pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)
    df["pre_min"] = df["pre_min"].fillna(1.0)
    return df.fillna(0.0).astype(np.float32)


# ---------------------------------------------------------------- dataset
def build_dataset():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True))
    A_all, G_all, y_all, subj_all, act_all = [], [], [], [], []
    skipped = 0
    for path in files:
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        act, subj = m.groups()
        out = load_file(path)
        if out is None:
            skipped += 1
            continue
        a, g = out
        last = len(a) - WIN
        pk = int(a.argmax())
        peak_starts = pk - PEAK_OFFSETS
        peak_starts = peak_starts[(peak_starts >= 0) & (peak_starts <= last)]

        if act.startswith("F"):
            # Falls: ONLY windows that contain the impact (the rest of the file is
            # ambiguous: pre-fall activity / lying afterwards).
            starts, label = peak_starts, 1
        else:
            # ADLs: regular sliding windows + windows aligned on the ADL's own peak
            # (hard negatives: sitting down hard, jumping...). This stops the model
            # from using "peak in the middle of the window" as a shortcut.
            regular = np.arange(0, last + 1, ADL_STEP)
            starts, label = np.unique(np.concatenate([regular, peak_starts])), 0
        if len(starts) == 0:
            continue

        A_all.append(sliding_window_view(a, WIN)[starts])
        G_all.append(sliding_window_view(g, WIN)[starts])
        y_all += [label] * len(starts)
        subj_all += [subj] * len(starts)
        act_all += [act] * len(starts)

    if not A_all:
        sys.exit(f"No SisFall files found under {DATA_DIR}")
    print(f"Loaded {len(files)} files ({skipped} unreadable skipped)")
    A = np.concatenate(A_all)
    G = np.concatenate(G_all)
    return (A, G, np.array(y_all), np.array(subj_all), np.array(act_all))


def make_model():
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )


def pick_threshold(y_true, proba, min_recall=0.95):
    """Highest-precision threshold that still keeps recall >= min_recall."""
    p, r, t = precision_recall_curve(y_true, proba)
    ok = np.where(r[:-1] >= min_recall)[0]
    if len(ok) == 0:
        return 0.5
    return float(t[ok[np.argmax(p[:-1][ok])]])


def main():
    A, G, y, subj, act = build_dataset()
    print(f"Windows: {len(y)}  | falls: {y.sum()}  | ADL: {(y == 0).sum()}  "
          f"| subjects: {len(set(subj))}")
    print("Sanity check, median |accel| over ADL windows (should be ~1 g): "
          f"{np.median(A[y == 0].mean(1)):.2f}")

    X = window_features(A, G)
    del A, G

    # ---- 1) hold out whole subjects for the final test -----------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups=subj))
    Xtr, ytr, gtr = X.iloc[tr], y[tr], subj[tr]
    Xte, yte = X.iloc[te], y[te]
    print(f"Train subjects: {len(set(gtr))}  | Test subjects: {len(set(subj[te]))} "
          f"(test falls: {yte.sum()})")

    # ---- 2) subject-wise CV on the training subjects only ---------------------
    oof = np.zeros(len(ytr))
    rows = []
    for k, (a_i, b_i) in enumerate(GroupKFold(n_splits=5).split(Xtr, ytr, gtr)):
        m = make_model().fit(Xtr.iloc[a_i], ytr[a_i])
        pr = m.predict_proba(Xtr.iloc[b_i])[:, 1]
        oof[b_i] = pr
        pred = pr >= 0.5
        rows.append((precision_score(ytr[b_i], pred, zero_division=0),
                     recall_score(ytr[b_i], pred, zero_division=0),
                     f1_score(ytr[b_i], pred, zero_division=0)))
        print(f"  fold {k + 1}: precision={rows[-1][0]:.3f} recall={rows[-1][1]:.3f} f1={rows[-1][2]:.3f}")
    cv = np.array(rows)
    print("CV (subject-wise) mean+-std  precision={:.3f}+-{:.3f}  recall={:.3f}+-{:.3f}  f1={:.3f}+-{:.3f}"
          .format(cv[:, 0].mean(), cv[:, 0].std(), cv[:, 1].mean(), cv[:, 1].std(),
                  cv[:, 2].mean(), cv[:, 2].std()))

    # Threshold is chosen from out-of-fold predictions, NOT from the test subjects
    thr = pick_threshold(ytr, oof, min_recall=0.95)
    print(f"Chosen decision threshold (recall>=0.95 on CV): {thr:.3f}")

    # ---- 3) final evaluation on unseen subjects ------------------------------
    model = make_model().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = proba >= thr
    print("\n=== HELD-OUT SUBJECTS (random forest) ===")
    print(classification_report(yte, pred, target_names=["ADL", "FALL"], digits=3, zero_division=0))
    print("Average precision (PR-AUC):", round(average_precision_score(yte, proba), 3))
    cm = confusion_matrix(yte, pred)
    print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

    base = (Xte["a_max"] > 3.0).to_numpy()
    print("\n=== BASELINE: 'peak acceleration > 3 g' only ===")
    print(classification_report(yte, base, target_names=["ADL", "FALL"], digits=3, zero_division=0))

    # Which ADL types cause false alarms?
    fp_acts = pd.Series(act[te][(pred == 1) & (yte == 0)]).value_counts().head(8)
    if len(fp_acts):
        print("ADL types producing most false alarms (see SisFall readme for names):")
        print(fp_acts.to_string())

    # ---- plots ----------------------------------------------------------------
    ConfusionMatrixDisplay(cm, display_labels=["ADL", "FALL"]).plot(cmap="Blues")
    plt.title("Held-out subjects")
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(15)
    imp.plot.barh(figsize=(6, 5), title="Top feature importances")
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 4) final model on all data, saved with everything needed to reuse it --
    final = make_model().fit(X, y)
    joblib.dump({"model": final, "features": list(X.columns), "fs": FS, "win": WIN,
                 "threshold": thr}, os.path.join(OUT_DIR, "fall_rf_sisfall.joblib"))
    print("\nSaved fall_rf_sisfall.joblib, confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()
