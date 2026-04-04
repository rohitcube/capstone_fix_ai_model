"""live_test.py

Real-time gesture inference from MQTT stream using trained weights.
Pure numpy inference (no TensorFlow needed) -- matches FPGA pipeline exactly.

V0: Bare-bones continuous prediction. No thresholds, no cooldown.
    Predicts every time both buffers fill (every ~1 second).

Usage:
    python live_test.py
    python live_test.py --weights-dir weights_out
    python live_test.py --broker 192.168.1.100
"""

import argparse
import csv
import json
import os
import ssl
import sys
import time
import threading
from collections import defaultdict

import numpy as np
import paho.mqtt.client as mqtt

from preprocess import build_feature_vector

# ── Config ──
GESTURE_NAMES = [
    "no_gesture", "move_forward", "turn_left", "turn_right",
    "jump", "attack", "turn_180",
]

WINDOW_SIZE = 50
SLIDE_STEP = 10           # predict every 10 new samples (0.2s)
VOTE_WINDOW = 5          # require majority in last 5 predictions to emit
VOTE_THRESHOLD = 3        # need 3/5 agreeing predictions to emit
COOLDOWN_SECONDS = 1.8    # suppress after any gesture emit
MOTION_THRESHOLD = 0.15   # gyro_std below this = idle (idle max=0.055, gestures p5≈0.08+)

# Sensor-pattern filter: which gestures require leg movement
# move_forward(1), jump(4), turn_180(6) need both sensors
NEEDS_LEG = {1, 4, 6}
LEG_MOTION_THRESHOLD = 0.10  # higher than MOTION_THRESHOLD — leg must clearly be moving

DEVICE_IMU0 = "arm"
DEVICE_IMU1 = "leg"

PREDICT_TOPIC = "u96/out"
CA_CERT_PATH = r"C:\capstone_repo\cg4002-b01-capstone\mosquitto\mosquitto-certs\ca.crt"

# Debug — save each inference window to debug_windows/
DEBUG_SAVE_WINDOWS = True
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_windows")
if DEBUG_SAVE_WINDOWS:
    os.makedirs(DEBUG_DIR, exist_ok=True)


class NumpyMLP:
    """Pure numpy MLP inference matching FPGA pipeline."""

    def __init__(self, weights_dir):
        self.scaler_mean = np.load(os.path.join(weights_dir, "scaler_mean.npy"))
        self.scaler_scale = np.load(os.path.join(weights_dir, "scaler_scale.npy"))
        self.w1 = np.load(os.path.join(weights_dir, "dense1_weights.npy"))
        self.b1 = np.load(os.path.join(weights_dir, "dense1_bias.npy"))
        self.w2 = np.load(os.path.join(weights_dir, "dense2_weights.npy"))
        self.b2 = np.load(os.path.join(weights_dir, "dense2_bias.npy"))
        self.w3 = np.load(os.path.join(weights_dir, "output_weights.npy"))
        self.b3 = np.load(os.path.join(weights_dir, "output_bias.npy"))

        num_classes = self.b3.shape[0]
        self.idx_to_name = {i: GESTURE_NAMES[i] if i < len(GESTURE_NAMES)
                            else f"gesture_{i}" for i in range(num_classes)}
        print(f"Loaded model: 84 -> {self.w1.shape[1]} -> {self.w2.shape[1]} -> {num_classes}")

    def predict(self, features_84):
        x = (features_84 - self.scaler_mean) / self.scaler_scale
        x = np.maximum(0, x @ self.w1 + self.b1)
        x = np.maximum(0, x @ self.w2 + self.b2)
        logits = x @ self.w3 + self.b3
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        pred_class = int(np.argmax(logits))
        return pred_class, probs


class LiveInference:
    def __init__(self, model, window_size=50, mqtt_client=None):
        self.model = model
        self.window_size = window_size
        self.mqtt_client = mqtt_client
        self.buffers = defaultdict(list)
        self.lock = threading.Lock()
        self.prediction_count = 0
        self.emit_count = 0
        self.recent_preds = []  # last N (class, confidence) for voting
        self.last_emit_time = 0
        self.samples_since_predict = 0

    def push_sample(self, device_id, sample):
        with self.lock:
            self.buffers[device_id].append(sample)
            self.samples_since_predict += 1

            imu0_buf = self.buffers[DEVICE_IMU0]
            imu1_buf = self.buffers[DEVICE_IMU1]

            if (len(imu0_buf) >= self.window_size and
                len(imu1_buf) >= self.window_size and
                self.samples_since_predict >= SLIDE_STEP):

                self.samples_since_predict = 0

                # Take last window_size samples (sliding window)
                imu0_data = np.array(imu0_buf[-self.window_size:], dtype=np.float32)
                imu1_data = np.array(imu1_buf[-self.window_size:], dtype=np.float32)

                # Trim buffers to prevent unbounded growth (keep window_size)
                if len(imu0_buf) > self.window_size * 2:
                    self.buffers[DEVICE_IMU0] = imu0_buf[-self.window_size:]
                if len(imu1_buf) > self.window_size * 2:
                    self.buffers[DEVICE_IMU1] = imu1_buf[-self.window_size:]

                self._run_inference(imu0_data, imu1_data)

    def _run_inference(self, imu0_window, imu1_window):
        # Motion gate: skip if both sensors have low rotation
        arm_gyro_mag = np.sqrt(np.sum(imu0_window[:, 3:6] ** 2, axis=1))
        leg_gyro_mag = np.sqrt(np.sum(imu1_window[:, 3:6] ** 2, axis=1))
        arm_g = np.std(arm_gyro_mag)
        leg_g = np.std(leg_gyro_mag)
        if arm_g < MOTION_THRESHOLD and leg_g < MOTION_THRESHOLD:
            sys.stdout.write(f"\r  [IDLE] gyro arm={arm_g:.3f} leg={leg_g:.3f}   ")
            sys.stdout.flush()
            return

        features = build_feature_vector(imu0_window, imu1_window)
        pred_class, probs = self.model.predict(features)

        # Sensor-pattern filter: if leg is idle, block gestures that need leg
        if pred_class in NEEDS_LEG and leg_g < LEG_MOTION_THRESHOLD:
            orig_name = self.model.idx_to_name.get(pred_class, f"class_{pred_class}")
            # Leg not moving — pick best class that doesn't need leg
            sorted_classes = np.argsort(probs)[::-1]
            for cls in sorted_classes:
                if cls not in NEEDS_LEG or leg_g >= LEG_MOTION_THRESHOLD:
                    pred_class = int(cls)
                    break
            new_name = self.model.idx_to_name.get(pred_class, f"class_{pred_class}")
            sys.stdout.write(f"\r  [LEG FILTER] {orig_name}->{new_name} leg={leg_g:.3f} < thresh={LEG_MOTION_THRESHOLD}   ")
            sys.stdout.flush()

        self.prediction_count += 1
        name = self.model.idx_to_name.get(pred_class, f"class_{pred_class}")
        confidence = probs[pred_class] * 100

        # Save debug window
        if DEBUG_SAVE_WINDOWS:
            self._save_debug_window(imu0_window, imu1_window, name, confidence)

        # Confidence gate
        if probs[pred_class] < 0.09:
            return

        # Add to vote buffer
        self.recent_preds.append(pred_class)
        if len(self.recent_preds) > VOTE_WINDOW:
            self.recent_preds = self.recent_preds[-VOTE_WINDOW:]

        # Check for consensus
        if len(self.recent_preds) < VOTE_THRESHOLD:
            return

        # Count votes
        from collections import Counter as Ctr
        votes = Ctr(self.recent_preds)
        top_class, top_count = votes.most_common(1)[0]

        if top_count < VOTE_THRESHOLD:
            # No consensus — show what's being seen
            sys.stdout.write(f"\r  voting: {name}({confidence:.0f}%) [{top_count}/{VOTE_WINDOW}]   ")
            sys.stdout.flush()
            return

        # Consensus reached — check cooldown
        now = time.time()
        if now - self.last_emit_time < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - (now - self.last_emit_time)
            sys.stdout.write(f"\r  [COOLDOWN {remaining:.1f}s] {name}   ")
            sys.stdout.flush()
            return
        self.last_emit_time = now

        # Emit!
        self.emit_count += 1
        top_name = self.model.idx_to_name.get(top_class, f"class_{top_class}")
        top_conf = confidence  # use latest confidence

        print(f"\n{'='*50}")
        print(f"  >>> {top_name.upper()} ({top_count}/{VOTE_WINDOW} votes)")
        print(f"  Confidence: {top_conf:.1f}%")
        print(f"  Gyro: arm={arm_g:.3f} leg={leg_g:.3f}")
        print(f"  Emit #{self.emit_count}")
        print(f"{'='*50}")

        # Publish prediction to MQTT
        if self.mqtt_client and self.mqtt_client.is_connected():
            payload = json.dumps({
                "type": "gesture_prediction",
                "window_index": self.emit_count,
                "pred_class": int(top_class),
                "logits": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "timestamp": time.time(),
            })
            self.mqtt_client.publish(PREDICT_TOPIC, payload, qos=1)
            print(f"  [MQTT] Published to {PREDICT_TOPIC}: pred_class={top_class}")

        # Clear vote buffer after emit to prevent re-firing
        self.recent_preds.clear()

    def _save_debug_window(self, imu0_window, imu1_window, predicted, confidence):
        """Save the raw window data so we can compare with training data."""
        fname = f"window_{self.prediction_count}_{predicted}_{confidence:.0f}pct.csv"
        fpath = os.path.join(DEBUG_DIR, fname)
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "imu_id", "ax", "ay", "az", "gx", "gy", "gz"])
            for t, row in enumerate(imu0_window):
                writer.writerow([t, 0] + [f"{v:.5f}" for v in row])
            for t, row in enumerate(imu1_window):
                writer.writerow([t, 1] + [f"{v:.5f}" for v in row])

    def get_status(self):
        with self.lock:
            return {
                DEVICE_IMU0: len(self.buffers[DEVICE_IMU0]),
                DEVICE_IMU1: len(self.buffers[DEVICE_IMU1]),
            }


def main():
    parser = argparse.ArgumentParser(description="Live gesture inference")
    parser.add_argument("--weights-dir", default="weights_out_v2")
    parser.add_argument("--broker", default="172.20.10.2")
    parser.add_argument("--port", type=int, default=8883)
    parser.add_argument("--topic", default="firebeetle/raw")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    if not os.path.isdir(args.weights_dir):
        print(f"Weights directory not found: {args.weights_dir}")
        print("Run train.py first to generate weights.")
        sys.exit(1)

    model = NumpyMLP(args.weights_dir)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="live-tester")

    # TLS setup — skip cert verification for now
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)

    inference = LiveInference(model, args.window_size, mqtt_client=client)

    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"[MQTT] Connected (rc={rc}), subscribing to {args.topic}")
        client.subscribe(args.topic)

    def on_message(client, userdata, msg, properties=None):
        try:
            payload = msg.payload.decode(errors="ignore").strip()
            parts = payload.split(",")
            if len(parts) != 8:
                return
        except Exception:
            return

        device_id = parts[0]
        if device_id not in (DEVICE_IMU0, DEVICE_IMU1):
            return

        sample = [float(parts[i]) for i in range(2, 8)]
        inference.push_sample(device_id, sample)

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {args.broker}:{args.port} (TLS)...")
    try:
        client.connect(args.broker, args.port, keepalive=30)
    except ConnectionRefusedError:
        print("ERROR: Could not connect to MQTT broker.")
        sys.exit(1)

    client.loop_start()
    print("Listening for IMU data... (Ctrl+C to stop)\n")
    print(f"Expecting devices: {DEVICE_IMU0}, {DEVICE_IMU1}")
    print(f"Window: {args.window_size} samples, slide every {SLIDE_STEP}")
    print(f"Voting: {VOTE_THRESHOLD}/{VOTE_WINDOW} consensus required")
    print(f"Cooldown: {COOLDOWN_SECONDS}s (discrete gestures only)\n")

    try:
        while True:
            status = inference.get_status()
            sys.stdout.write(
                f"\r  {DEVICE_IMU0}={status[DEVICE_IMU0]:3d}/{args.window_size}  "
                f"{DEVICE_IMU1}={status[DEVICE_IMU1]:3d}/{args.window_size}  "
                f"Windows: {inference.prediction_count}  Emits: {inference.emit_count}   "
            )
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"Total predictions made: {inference.prediction_count}")


if __name__ == "__main__":
    main()
