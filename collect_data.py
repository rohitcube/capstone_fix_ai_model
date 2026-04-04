"""collect_data.py

MQTT data collection tool for recording labeled gesture samples.
Subscribes to the local Mosquitto broker, lets you record gestures
via keyboard, and saves CSVs compatible with the training pipeline.

Usage:
    python collect_data.py

Prerequisites:
    - Mosquitto broker running: mosquitto -c mosquitto.conf -v
    - FireBeetle(s) streaming IMU data to firebeetle/imu
    - pip install paho-mqtt
"""

import csv
import os
import ssl
import sys
import time
import threading
from collections import defaultdict

import paho.mqtt.client as mqtt

# ── Configuration ──
BROKER_HOST = "172.20.10.2"
BROKER_PORT = 8883
CA_CERT_PATH = r"C:\capstone_repo\cg4002-b01-capstone\mosquitto\mosquitto-certs\ca.crt"
MQTT_TOPIC = "firebeetle/imu"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "data_v2")
os.makedirs(SAVE_DIR, exist_ok=True)

PERSON_ID = 1
# 1 rohit
# 2 
# 3 pradeep
# 4 ansel
# 5 keerthaan

# Map device names to imu_id (must match DEVICE_NAME on each board)
DEVICE_TO_IMU = {
    "arm": 0,
    "leg": 1,
}

GESTURE_NAMES = {
    0: "no_gesture",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
    4: "jump",
    5: "attack",
    6: "turn_180", #to the left
}

# ── State ──
recording = False
current_gesture = None
current_sample_id = None
buffer = []  # list of dicts
device_seen = defaultdict(float)  # device_name -> last seen timestamp
sample_counts = defaultdict(int)  # gesture_id -> count of saved samples
next_sample_id = 0
lock = threading.Lock()


def _count_existing_samples():
    """Count already-saved samples per gesture in SAVE_DIR."""
    global next_sample_id
    max_id = -1
    for fname in os.listdir(SAVE_DIR):
        if fname.startswith("collected_") and fname.endswith(".csv"):
            parts = fname.replace(".csv", "").split("_")
            for p in parts:
                if p.startswith("g") and p[1:].isdigit():
                    g = int(p[1:])
                    sample_counts[g] += 1
                if p.startswith("s") and p[1:].isdigit():
                    max_id = max(max_id, int(p[1:]))
    next_sample_id = max_id + 1


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[MQTT] Connected (rc={reason_code}), subscribing to {MQTT_TOPIC}")
    result = client.subscribe(MQTT_TOPIC)
    print(f"[MQTT] Subscribe result: {result}")


def on_message(client, userdata, msg, properties=None):
    global buffer
    try:
        payload = msg.payload.decode(errors="ignore").strip()
        # CSV format: device_name,timestamp,ax,ay,az,gx,gy,gz
        parts = payload.split(",")
        if len(parts) != 8:
            print(f"[DEBUG] Unexpected field count ({len(parts)}): {payload[:80]}")
            return
    except Exception as e:
        print(f"[DEBUG] Parse error: {e}")
        return

    device_name = parts[0]
    device_seen[device_name] = time.time()

    row = {
        "device": device_name,
        "timestamp": int(parts[1]),
        "ax": float(parts[2]),
        "ay": float(parts[3]),
        "az": float(parts[4]),
        "gx": float(parts[5]),
        "gy": float(parts[6]),
        "gz": float(parts[7]),
    }

    with lock:
        if recording:
            buffer.append(row)


def save_recording(gesture_id, sample_id, data):
    """Save buffered data as a CSV, pairing both IMUs."""
    if not data:
        print("  No data to save!")
        return

    # Split by device
    by_device = defaultdict(list)
    for row in data:
        by_device[row["device"]].append(row)

    devices_found = list(by_device.keys())
    print(f"  Devices in recording: {devices_found}")
    for d in devices_found:
        print(f"    {d}: {len(by_device[d])} rows")

    # Trim both devices to the same number of rows
    if len(devices_found) == 2:
        min_len = min(len(by_device[d]) for d in devices_found)
        for d in devices_found:
            by_device[d] = by_device[d][:min_len]
        print(f"  Trimmed both to {min_len} rows")
    elif len(devices_found) == 1:
        print("  WARNING: Only 1 device detected. Is the other board connected?")

    fname = f"collected_p{PERSON_ID}_g{gesture_id}_s{sample_id}.csv"
    fpath = os.path.join(SAVE_DIR, fname)

    with open(fpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "gesture_id", "timestep", "sample_id",
                         "imu_id", "ax", "ay", "az", "gx", "gy", "gz"])

        for device_name, rows in sorted(by_device.items()):
            imu_id = DEVICE_TO_IMU.get(device_name, 0)
            for t, row in enumerate(rows):
                writer.writerow([
                    PERSON_ID, gesture_id, t, sample_id, imu_id,
                    f"{row['ax']:.5f}", f"{row['ay']:.5f}", f"{row['az']:.5f}",
                    f"{row['gx']:.5f}", f"{row['gy']:.5f}", f"{row['gz']:.5f}",
                ])

    print(f"  Saved to {fname}")
    sample_counts[gesture_id] += 1


def print_status():
    """Print current device status and sample counts."""
    now = time.time()
    print("\n" + "=" * 50)
    print("DEVICE STATUS:")
    if not device_seen:
        print("  No devices detected yet. Is the broker running?")
    for dev, last in sorted(device_seen.items()):
        age = now - last
        imu_label = f"imu_id={DEVICE_TO_IMU.get(dev, '?')}"
        status = "ACTIVE" if age < 2 else f"last seen {age:.0f}s ago"
        print(f"  {dev} ({imu_label}): {status}")

    print("\nSAMPLE COUNTS:")
    for g_id in sorted(GESTURE_NAMES.keys()):
        count = sample_counts.get(g_id, 0)
        name = GESTURE_NAMES[g_id]
        bar = "#" * count
        print(f"  {g_id}: {name:>12s}  [{count:3d}] {bar}")
    print("=" * 50)


def print_menu():
    print("\nGESTURES:")
    for g_id, name in sorted(GESTURE_NAMES.items()):
        print(f"  {g_id} = {name}")
    print("\nCOMMANDS:")
    print("  <number>  = select gesture, then Enter to start/stop recording")
    print("  s         = show status")
    print("  q         = quit")


def main():
    global recording, current_gesture, current_sample_id, buffer, next_sample_id

    _count_existing_samples()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="data-collector")
    client.tls_set(
        ca_certs=CA_CERT_PATH,
        certfile=None,
        keyfile=None,
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS_CLIENT,
    )
    client.tls_insecure_set(False)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = lambda c, u, m, rc, p=None: print(f"[MQTT] Subscribed successfully")
    client.on_disconnect = lambda c, u, d, rc, p=None: print(f"[MQTT] Disconnected: {rc}")

    print(f"Connecting to MQTT broker at {BROKER_HOST}:{BROKER_PORT} (TLS)...")
    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=30)
    except ConnectionRefusedError:
        print("ERROR: Could not connect to MQTT broker.")
        print("Make sure Mosquitto is running: mosquitto -c mosquitto.conf -v")
        sys.exit(1)

    client.loop_start()
    print("Connected! Waiting for device data...\n")
    time.sleep(2)  # let some packets arrive
    print_status()
    print_menu()

    try:
        while True:
            cmd = input("\n> ").strip().lower()

            if cmd == "q":
                break
            elif cmd == "s":
                print_status()
                continue
            elif cmd == "h":
                print_menu()
                continue

            # Try to parse as gesture number
            try:
                gesture_id = int(cmd)
            except ValueError:
                print("Unknown command. Type 'h' for help.")
                continue

            if gesture_id not in GESTURE_NAMES:
                print(f"Invalid gesture ID. Valid: {list(GESTURE_NAMES.keys())}")
                continue

            gesture_name = GESTURE_NAMES[gesture_id]
            print(f"\nSelected: {gesture_id} ({gesture_name})")
            print("Press ENTER to START recording, then ENTER again to STOP.")
            input("  >> Press ENTER to START...")

            # Start recording
            with lock:
                buffer = []
                recording = True
                current_gesture = gesture_id
                current_sample_id = next_sample_id

            print(f"  RECORDING gesture '{gesture_name}'... (press ENTER to stop)")
            input()

            # Stop recording
            with lock:
                recording = False
                saved_data = list(buffer)
                buffer = []

            print(f"  Stopped. Captured {len(saved_data)} rows.")
            save_recording(gesture_id, current_sample_id, saved_data)
            next_sample_id += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Disconnected from broker.")
        print_status()


if __name__ == "__main__":
    main()
