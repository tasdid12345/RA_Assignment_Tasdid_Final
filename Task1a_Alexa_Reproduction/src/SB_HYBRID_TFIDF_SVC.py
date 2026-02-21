import os
import re
import glob
import time
import random
import requests
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
GITHUB_OWNER = "SmartHomePrivacyProject"
GITHUB_REPO = "VCFingerprinting"
GITHUB_BRANCH = "master"
GITHUB_FOLDER_PATH = "data/trace_csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

LOCAL_TRACE_DIR = os.path.join(SCRIPT_DIR, "trace_csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

TEST_SIZE = 0.2
RANDOM_STATE = 42

LABELS_100_XLSX = os.path.join(PROJECT_DIR, "amazon_echo_query_list_100.xlsx")
LABELS_100_XLSX_URL = (
    "https://raw.githubusercontent.com/SmartHomePrivacyProject/VCFingerprinting/master/data/"
    "amazon_echo_query_list_100.xlsx"
)

USE_ALLOWED_LABELS_XLSX = True
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()

# Token controls
USE_IAT = True

# 🔥 include BOTH raw + bin size tokens
USE_RAW_SIZE_TOKEN = True
USE_BIN_SIZE_TOKEN = True

# TF-IDF params
NGRAM_RANGE = (1, 4)
MIN_DF = 1
MAX_FEATURES = 250000

# SVC convergence
SVC_MAX_ITER = 100000

# Grid search
DO_C_GRID = True
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# Calibration for Top-k
CALIB_CV = 3
CALIB_METHOD = "sigmoid"


# =========================================================
# LABEL CANONICALIZATION MAPS
# =========================================================
ALIAS_MAP = {
    "how many seconds in a year": "how many seconds are in a year",
    "how many teaspoons in a table spoon": "how many teaspoons are in a tablespoon",
    "how many teaspoons in a tablespoon": "how many teaspoons are in a tablespoon",
    "play npr": "play npr 91 7 wvxu",
    "set a timer for 30 seconds": "set an timer for 30 seconds",
    "set a timer for thirty seconds": "set an timer for 30 seconds",
    "set volume to five": "set volume to 5",
    "what happened in the midterm election": "what happened in the midterm elections",
    "what is in the news": "whats in the news",
    "what the best comedy movie": "whats the best comedy movie",
    "what is the best comedy movie": "whats the best comedy movie",
    "what is the date tomorrow": "whats the date tomorrow",
    "whats the fourth book in narnia series": "whats the fourth book in the narnia series",
    "what is the scariest movie of all time": "whats the scariest movie of all time",
    "who plays wolverine in xmen": "who plays wolverine in x men",
}

WHATIS_TO_WHATS = {
    "what is in the news": "whats in the news",
    "what is the date tomorrow": "whats the date tomorrow",
    "what is the scariest movie of all time": "whats the scariest movie of all time",
    "what is the best comedy movie": "whats the best comedy movie",
}


# =========================================================
# GITHUB HELPERS
# =========================================================
def github_headers():
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "VCFingerprintingDownloader/1.0"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def list_github_folder_files(owner: str, repo: str, path: str, ref: str = "master"):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r = requests.get(url, headers=github_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(
            f"GitHub API error {r.status_code}. Response:\n{r.text}\n\n"
            f"Tip: if rate-limited (403), set GITHUB_TOKEN env var."
        )
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected API response type: {type(data)}. Response:\n{data}")
    return data


def download_with_retries(
    session: requests.Session,
    url: str,
    out_path: str,
    max_tries: int = 6,
    base_sleep: float = 0.6
):
    tmp_path = out_path + ".part"
    for attempt in range(1, max_tries + 1):
        try:
            with session.get(url, headers=github_headers(), stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, out_path)
            return True
        except Exception:
            sleep_s = (base_sleep * (2 ** (attempt - 1))) + random.uniform(0, 0.35)
            time.sleep(sleep_s)

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return False


def download_trace_csvs(throttle=0.03):
    os.makedirs(LOCAL_TRACE_DIR, exist_ok=True)
    items = list_github_folder_files(GITHUB_OWNER, GITHUB_REPO, GITHUB_FOLDER_PATH, ref=GITHUB_BRANCH)
    csv_items = [it for it in items if it.get("type") == "file" and it.get("name", "").endswith(".csv")]

    print("Downloading dataset from GitHub (Contents API)...")
    print(f"Found {len(csv_items)} CSV files")

    session = requests.Session()
    ok = skipped = failed = 0
    for idx, it in enumerate(csv_items, start=1):
        name = it["name"]
        url = it.get("download_url")
        if not url:
            failed += 1
            continue

        out_path = os.path.join(LOCAL_TRACE_DIR, name)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            continue

        success = download_with_retries(session, url, out_path)
        ok += int(success)
        failed += int(not success)

        time.sleep(throttle)
        if idx % 50 == 0:
            print(f"Progress: {idx}/{len(csv_items)} | ok={ok} skipped={skipped} failed={failed}")

    print(f"Download done. ok={ok}, skipped={skipped}, failed={failed}\n")
    local_count = len(glob.glob(os.path.join(LOCAL_TRACE_DIR, "*.csv")))
    print(f"Local CSV count now: {local_count}")


def ensure_file_downloaded(url: str, dst_path: str):
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)


# =========================================================
# LABEL PARSING
# =========================================================
def normalize_text_label(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\"'“”‘’]", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(what|where|who|how|when|why|that|there|here|it)\s+s\b", r"\1s", s)
    s = re.sub(r"\b(\w+)\s+n\s+t\b", r"\1nt", s)
    s = re.sub(r"\bwon\s+t\b", "wont", s)
    s = re.sub(r"\bcan\s+t\b", "cant", s)
    s = re.sub(r"\bdo\s+not\b", "dont", s)
    s = re.sub(r"\bx\s*men\b", "x men", s)
    s = re.sub(r"\btable\s+spoon\b", "tablespoon", s)
    return s


def strip_capture_suffix_by_tokens(stem: str) -> str:
    toks = stem.split("_")
    while toks:
        t = toks[-1].lower()
        if t in {"l", "capture", "cap", "pcap", "trace"}:
            toks.pop()
            continue
        if re.fullmatch(r"(capture|cap|pcap|trace)\d*", t):
            toks.pop()
            continue
        if re.fullmatch(r"\d+", t) or re.fullmatch(r"\d+s", t):
            toks.pop()
            continue
        break
    return "_".join(toks) if toks else stem


def canonicalize_label(label: str) -> str:
    label = normalize_text_label(label)
    if label in WHATIS_TO_WHATS:
        label = WHATIS_TO_WHATS[label]
    label = re.sub(r"\bthirty seconds\b", "30 seconds", label)
    label = re.sub(r"\bto five\b", "to 5", label)
    return ALIAS_MAP.get(label, label)


def infer_label_from_filename(fp: str) -> str:
    stem = os.path.splitext(os.path.basename(fp))[0].lower()
    stem = strip_capture_suffix_by_tokens(stem)
    return canonicalize_label(stem)


def load_allowed_labels_100(xlsx_path: str) -> set:
    if not (os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 0):
        print(f"[INFO] Missing label Excel. Downloading to: {xlsx_path}")
        ensure_file_downloaded(LABELS_100_XLSX_URL, xlsx_path)
    df = pd.read_excel(xlsx_path)
    labels = []
    for x in df.iloc[:, 0].tolist():
        if pd.isna(x):
            continue
        labels.append(canonicalize_label(x))
    return set(labels)


# =========================================================
# TRACE -> TEXT TOKENS
# =========================================================
def dir_to_sign(d):
    if pd.isna(d):
        return 1
    if isinstance(d, (int, float, np.integer, np.floating)):
        return -1 if int(d) < 0 else 1
    s = str(d).strip().lower()
    if s in {"-1", "in", "incoming", "inbound", "rx", "recv", "receive", "download"}:
        return -1
    return 1


def size_bin(s: float) -> int:
    edges = [0, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 3000]
    for i in range(len(edges) - 1):
        if edges[i] <= s < edges[i + 1]:
            return i
    return len(edges) - 2


def iat_bin(dt: float) -> str:
    if dt < 0.0005:
        return "I0"
    if dt < 0.001:
        return "I1"
    if dt < 0.002:
        return "I2"
    if dt < 0.005:
        return "I3"
    if dt < 0.01:
        return "I4"
    if dt < 0.02:
        return "I5"
    if dt < 0.05:
        return "I6"
    if dt < 0.1:
        return "I7"
    return "I8"


def trace_to_text(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    # drop unnamed index
    if df.columns.tolist() and df.columns.tolist()[0] in {"", "unnamed: 0"}:
        df = df.drop(columns=[df.columns.tolist()[0]])
        df.columns = [c.lower().strip() for c in df.columns]

    # unify column names
    if "size" not in df.columns:
        for alt in ["pkt_size", "packet_size", "length", "len"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "size"})
                break
    if "direction" not in df.columns:
        for alt in ["dir", "flow", "sign"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "direction"})
                break
    if "time" not in df.columns:
        for alt in ["ts", "timestamp", "t"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "time"})
                break

    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0.0)
    df["direction"] = df["direction"].apply(dir_to_sign)

    sizes = df["size"].to_numpy()
    times = df["time"].to_numpy()
    dirs = df["direction"].to_numpy()

    toks = []
    prev_t = None
    for s, t, d in zip(sizes, times, dirs):
        sign = "IN" if d < 0 else "OUT"
        s_int = int(round(float(s)))

        # ---- DUAL TOKENS (raw + bin) ----
        if USE_RAW_SIZE_TOKEN:
            toks.append(f"{sign}_{s_int}")
        if USE_BIN_SIZE_TOKEN:
            toks.append(f"{sign}_B{size_bin(float(s))}")
        # ---------------------------------

        if USE_IAT:
            if prev_t is None:
                toks.append("I_START")
            else:
                toks.append(iat_bin(float(t - prev_t)))
            prev_t = float(t)

    return " ".join(toks)


def plot_and_save_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_model(C_val: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\S+",
                ngram_range=NGRAM_RANGE,
                min_df=MIN_DF,
                max_features=MAX_FEATURES,
                sublinear_tf=True,
                norm="l2",
            )),
            ("svc", LinearSVC(C=C_val, max_iter=SVC_MAX_ITER)),
        ]
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    download_trace_csvs()

    csv_files = sorted(glob.glob(os.path.join(LOCAL_TRACE_DIR, "*.csv")))
    if len(csv_files) == 0:
        raise RuntimeError(f"No CSVs found in {LOCAL_TRACE_DIR}")

    y_all = [infer_label_from_filename(fp) for fp in csv_files]

    if USE_ALLOWED_LABELS_XLSX:
        allowed = load_allowed_labels_100(LABELS_100_XLSX)
        kept = [(fp, y) for fp, y in zip(csv_files, y_all) if y in allowed]
        csv_files = [fp for fp, _ in kept]
        y = np.array([yy for _, yy in kept])
    else:
        y = np.array(y_all)

    print(f"After filtering: Traces={len(csv_files)}, Unique labels={len(set(y))}")

    # Build docs
    X_text = [trace_to_text(fp) for fp in csv_files]

    # Split once
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # C selection by CV on training set
    best_C = C_GRID[0]
    best_mean = -1.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if DO_C_GRID:
        print("\n=== C GRID SEARCH (CV on train) ===")
        for C_val in C_GRID:
            model = build_model(C_val)
            scores = []
            for tr_idx, va_idx in skf.split(X_train, y_train):
                X_tr = [X_train[i] for i in tr_idx]
                y_tr = y_train[tr_idx]
                X_va = [X_train[i] for i in va_idx]
                y_va = y_train[va_idx]
                model.fit(X_tr, y_tr)
                pred = model.predict(X_va)
                scores.append(accuracy_score(y_va, pred))
            mean_acc = float(np.mean(scores))
            std_acc = float(np.std(scores))
            print(f"C={C_val:<5}  CV mean={mean_acc:.4f}  std={std_acc:.4f}")
            if mean_acc > best_mean:
                best_mean = mean_acc
                best_C = C_val
        print(f"\n[BEST] C={best_C} with CV mean={best_mean:.4f}\n")

    # Train final model with best C
    final_model = build_model(best_C)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("=== Held-out Test ===")
    print(f"Top-1: {acc*100:.2f}%")

    # Calibrate for Top-k
    calib = CalibratedClassifierCV(final_model, method=CALIB_METHOD, cv=CALIB_CV)
    calib.fit(X_train, y_train)
    proba = calib.predict_proba(X_test)
    label_order = calib.classes_

    top3 = top_k_accuracy_score(y_test, proba, k=3, labels=label_order)
    top5 = top_k_accuracy_score(y_test, proba, k=5, labels=label_order)
    print(f"Top-3: {top3*100:.2f}%")
    print(f"Top-5: {top5*100:.2f}%\n")

    # SAVE OUTPUT FILES
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    # confusion_matrix.csv
    cm_csv = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(cm_csv)

    # confusion_matrix.png
    cm_png = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_and_save_confusion_matrix(y_test, y_pred, labels=labels_sorted, out_path=cm_png)

    # classification_report.txt
    report_txt = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # metrics.txt
    metrics_txt = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write("Model: TFIDF DualTokens (raw+bin) + IAT + LinearSVC (calibrated)\n")
        f.write(f"Best C: {best_C}\n")
        f.write(f"Top-1 Accuracy: {acc*100:.2f}%\n")
        f.write(f"Top-3 Accuracy: {top3*100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5*100:.2f}%\n")
        f.write(f"Total Traces: {len(csv_files)}\n")
        f.write(f"Total Classes: {len(set(y))}\n")

    print("\nSaved outputs:")
    print(report_txt)
    print(metrics_txt)
    print(cm_csv)
    print(cm_png)

    print("\nDONE.")


if __name__ == "__main__":
    main()