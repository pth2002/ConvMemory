import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURES = ["top1_zscore", "top2_margin", "score_std"]


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows, features):
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)


def labels(rows, target):
    return np.asarray([float(row[target]) for row in rows], dtype=np.float32)


def expected_calibration_error(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows = []
    for idx in range(bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - accuracy)
        rows.append(
            {
                "bin": idx,
                "low": lo,
                "high": hi,
                "count": int(mask.sum()),
                "mean_confidence": confidence,
                "empirical_success": accuracy,
                "abs_gap": abs(confidence - accuracy),
            }
        )
    return float(ece), rows


def safe_auc(y_true, y_prob):
    if len(set(np.asarray(y_true, dtype=int))) < 2:
        return ""
    return float(roc_auc_score(y_true, y_prob))


def evaluate(name, y_true, y_prob):
    eps = 1e-6
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    ece, bins = expected_calibration_error(y_true, clipped)
    return {
        "model": name,
        "questions": int(len(y_true)),
        "brier": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped)),
        "auc": safe_auc(y_true, clipped),
        "ece_10": ece,
        "mean_probability": float(np.mean(clipped)),
        "empirical_success": float(np.mean(y_true)),
    }, bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="results/v042/error_calibration_mpnet/cases.csv")
    parser.add_argument("--target", choices=["conv_hit_at_10", "raw_hit_at_10"], default="conv_hit_at_10")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[31, 47])
    parser.add_argument("--out", default="results/v046/confidence_calibration_mpnet")
    args = parser.parse_args()

    rows = read_rows(args.cases)
    if args.target not in rows[0]:
        raise ValueError(f"Target column not found: {args.target}")

    train_seed_set = {str(seed) for seed in args.train_seeds}
    test_seed_set = {str(seed) for seed in args.test_seeds}
    train_rows = [row for row in rows if str(row["seed"]) in train_seed_set]
    test_rows = [row for row in rows if str(row["seed"]) in test_seed_set]
    if not train_rows or not test_rows:
        raise ValueError("Train/test seed split produced no rows.")

    x_train = matrix(train_rows, FEATURES)
    y_train = labels(train_rows, args.target)
    x_test = matrix(test_rows, FEATURES)
    y_test = labels(test_rows, args.target)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(x_train_scaled, y_train)
    logistic_prob = logistic.predict_proba(x_test_scaled)[:, 1]

    raw_margin = x_test[:, FEATURES.index("top2_margin")]
    raw_margin_norm = (raw_margin - raw_margin.min()) / (raw_margin.max() - raw_margin.min() + 1e-8)

    train_margin = x_train[:, FEATURES.index("top2_margin")]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(train_margin, y_train)
    isotonic_prob = isotonic.predict(raw_margin)

    baseline_prob = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=np.float32)

    summary = []
    calibration_rows = []
    for name, prob in [
        ("train_rate_baseline", baseline_prob),
        ("top2_margin_minmax", raw_margin_norm),
        ("platt_logistic_features", logistic_prob),
        ("isotonic_top2_margin", isotonic_prob),
    ]:
        row, bins = evaluate(name, y_test, prob)
        summary.append(row)
        for item in bins:
            calibration_rows.append({"model": name, **item})

    out_dir = Path(args.out)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "calibration_bins.csv", calibration_rows)

    report = ["# v0.46 Confidence Calibration", ""]
    report.append(f"Target: `{args.target}`")
    report.append(f"Train seeds: `{', '.join(map(str, args.train_seeds))}`")
    report.append(f"Test seeds: `{', '.join(map(str, args.test_seeds))}`")
    report.append("")
    report.append("| Model | Brier | Log loss | AUC | ECE@10 | Empirical success | Mean probability |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(summary, key=lambda item: item["brier"]):
        auc = row["auc"] if row["auc"] != "" else "n/a"
        report.append(
            f"| `{row['model']}` | {row['brier']:.4f} | {row['log_loss']:.4f} | "
            f"{auc if isinstance(auc, str) else f'{auc:.4f}'} | {row['ece_10']:.4f} | "
            f"{row['empirical_success']:.4f} | {row['mean_probability']:.4f} |"
        )
    report.append("")
    report.append(
        "This is a post-hoc confidence model for deciding whether a retrieved memory context is likely to contain evidence. "
        "It does not change ConvMemory rankings."
    )
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved calibration report to {out_dir}")


if __name__ == "__main__":
    main()
