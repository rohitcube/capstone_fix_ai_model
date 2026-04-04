"""preprocess.py

Feature extraction and data loading for IMU gesture recognition.
Matches the FPGA inference pipeline exactly (preprocess_receive_and_send.py).

Features: 7 stats x 6 channels x 2 IMUs = 84 features per window.
Window size: 100 timesteps (matching FPGA inference).
"""

import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


# ── Feature extraction (must match preprocess_receive_and_send.py exactly) ──

def extract_signal_features(signal_values):
    signal_values = np.asarray(signal_values, dtype=np.float32)
    mean_val   = np.mean(signal_values)
    std_val    = np.std(signal_values)
    rms_val    = np.sqrt(np.mean(np.square(signal_values)))
    max_val    = np.max(signal_values)
    min_val    = np.min(signal_values)
    median_val = np.median(signal_values)
    energy_val = np.sum(np.square(signal_values))
    return [mean_val, std_val, rms_val, max_val, min_val, median_val, energy_val]


def extract_features_for_one_imu(imu_window_2d):
    """imu_window_2d shape: (T, 6), columns: [ax, ay, az, gx, gy, gz]"""
    imu_window_2d = np.asarray(imu_window_2d, dtype=np.float32)
    assert imu_window_2d.ndim == 2 and imu_window_2d.shape[1] == 6, \
        f"Expected shape (T, 6), got {imu_window_2d.shape}"
    feature_list = []
    for col_idx in range(6):
        feature_list.extend(extract_signal_features(imu_window_2d[:, col_idx]))
    return feature_list


def build_feature_vector(imu0_window, imu1_window):
    """Combine both IMU windows into one 84-feature vector.
    imu0_window, imu1_window: shape (T, 6)
    """
    feats0 = extract_features_for_one_imu(imu0_window)
    feats1 = extract_features_for_one_imu(imu1_window)
    features = np.array(feats0 + feats1, dtype=np.float32)
    assert features.shape == (84,), f"Expected 84 features, got {features.shape}"
    return features


# ── Idle trimming ──
# Shared threshold — must match live_test.py MOTION_THRESHOLD
MOTION_ONSET_THRESHOLD = 1.0
ONSET_ROLLING = 10       # rolling window for std computation
ONSET_BUFFER = 5         # keep this many samples before detected onset

def trim_idle(imu_data):
    """Trim leading idle samples from IMU data (T, 6).
    Detects motion onset via rolling std of acceleration magnitude.
    Returns trimmed array. If no onset detected, returns original (no trim).
    """
    if len(imu_data) < ONSET_ROLLING:
        return imu_data

    # Acceleration magnitude from columns 0,1,2 (ax, ay, az)
    mag = np.sqrt(np.sum(imu_data[:, :3] ** 2, axis=1))

    # Rolling std
    for i in range(ONSET_ROLLING, len(mag)):
        window_std = np.std(mag[i - ONSET_ROLLING:i])
        if window_std > MOTION_ONSET_THRESHOLD:
            # Found onset — keep a small buffer before it
            start = max(0, i - ONSET_ROLLING - ONSET_BUFFER)
            return imu_data[start:]

    # No onset detected — return as-is
    return imu_data


# ── Windowing ──

def iter_windows(data, window_size=100, step=50, min_len=20):
    """Yield windows from a 2D array (T, C). Overlapping if step < window_size."""
    n = len(data)
    if n < min_len:
        return
    if n < window_size:
        yield data
        return
    for start in range(0, n - window_size + 1, step):
        yield data[start:start + window_size]


# ── CSV loading ──

def _load_old_format_csvs(csv_dir):
    """Load old-format CSVs (separate imu0/imu1 files per gesture).
    Files like: training_data_p1_g1_imu0.csv, gesture_data_p1_g2_imu1.csv
    """
    pattern = re.compile(r'(.+)_p(\d+)_g(\d+)_imu(\d+)\.csv')
    file_groups = defaultdict(dict)

    for fpath in glob.glob(os.path.join(csv_dir, '*_p*_g*_imu*.csv')):
        fname = os.path.basename(fpath)
        m = pattern.match(fname)
        if not m:
            continue
        prefix, person, gesture, imu = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        key = (prefix, person, gesture)
        file_groups[key][imu] = fpath

    recordings = []
    for (prefix, person, gesture), imu_files in file_groups.items():
        if 0 not in imu_files:
            print(f"  Skipping {prefix}_p{person}_g{gesture}: missing imu0")
            continue

        df0 = pd.read_csv(imu_files[0])
        signal_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

        if 1 in imu_files:
            df1 = pd.read_csv(imu_files[1])
        else:
            print(f"  Warning: {prefix}_p{person}_g{gesture} missing imu1, zero-filling")
            df1 = None

        # Group by sample_id within each file
        for sample_id, g0 in df0.groupby('sample_id'):
            g0 = g0.sort_values('timestep')
            imu0_data = g0[signal_cols].values

            if df1 is not None:
                g1 = df1[df1['sample_id'] == sample_id].sort_values('timestep')
                if len(g1) == 0:
                    imu1_data = np.zeros_like(imu0_data)
                else:
                    imu1_data = g1[signal_cols].values
            else:
                imu1_data = np.zeros_like(imu0_data)

            # Truncate to same length
            min_len_pair = min(len(imu0_data), len(imu1_data))
            recordings.append({
                'label': gesture,
                'imu0': imu0_data[:min_len_pair],
                'imu1': imu1_data[:min_len_pair],
                'source': f'{prefix}_p{person}_g{gesture}_s{sample_id}',
            })

    return recordings


def _load_new_format_csvs(csv_dir):
    """Load new-format CSVs (combined imu0+imu1 in one file).
    Files like: collected_p1_g3_s5.csv
    """
    recordings = []
    signal_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

    for fpath in glob.glob(os.path.join(csv_dir, 'collected_*.csv')):
        df = pd.read_csv(fpath)
        if 'imu_id' not in df.columns:
            continue

        for sample_id, g in df.groupby('sample_id'):
            gesture = int(g['gesture_id'].iloc[0])
            g0 = g[g['imu_id'] == 0].sort_values('timestep')
            g1 = g[g['imu_id'] == 1].sort_values('timestep')

            if len(g0) == 0:
                continue
            imu0_data = g0[signal_cols].values
            imu1_data = g1[signal_cols].values if len(g1) > 0 else np.zeros_like(imu0_data)

            min_len_pair = min(len(imu0_data), len(imu1_data))
            recordings.append({
                'label': gesture,
                'imu0': imu0_data[:min_len_pair],
                'imu1': imu1_data[:min_len_pair],
                'source': os.path.basename(fpath),
            })

    return recordings


def load_and_window_csvs(csv_dir, window_size=100, step=50, min_len=20):
    """Load all CSVs from csv_dir, extract windowed features.

    Returns DataFrame with columns: [label, f0, f1, ..., f83]
    """
    print(f"Loading data from: {csv_dir}")
    print(f"  Window size: {window_size}, Step: {step}")

    recordings = _load_old_format_csvs(csv_dir)
    recordings.extend(_load_new_format_csvs(csv_dir))

    print(f"  Found {len(recordings)} recordings")

    rows = []
    for rec in recordings:
        imu0 = rec['imu0']
        imu1 = rec['imu1']

        for w_idx, (w0, w1) in enumerate(zip(
            iter_windows(imu0, window_size, step, min_len),
            iter_windows(imu1, window_size, step, min_len),
        )):
            # Ensure same length
            wlen = min(len(w0), len(w1))
            features = build_feature_vector(w0[:wlen], w1[:wlen])
            row = {'label': rec['label']}
            for i, f in enumerate(features):
                row[f'f{i}'] = f
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        label_counts = df['label'].value_counts().sort_index()
        print(f"  Total windows: {len(df)}")
        print(f"  Per-class counts:")
        for label, count in label_counts.items():
            print(f"    gesture {label}: {count}")
    else:
        print("  WARNING: No windows extracted!")

    return df


if __name__ == "__main__":
    import sys
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data_v2")
    df = load_and_window_csvs(csv_dir, window_size=50, step=25)
    print(f"\nFeature matrix shape: {df.shape}")
    if len(df) > 0:
        print(df.head())
