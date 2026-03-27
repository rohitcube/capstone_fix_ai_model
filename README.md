# FireBeetle MQTT Demo (PlatformIO)

## Architecture
FireBeetle (ESP32) -> phone hotspot Wi-Fi -> Mosquitto on laptop:1883 -> laptop subscriber.

## Files
- `platformio.ini`
- `src/main.cpp`
- `mosquitto.conf`

## Laptop Commands
```powershell
# Start broker with config
mosquitto -c .\mosquitto.conf -v

# Subscribe from laptop terminal
mosquitto_sub -h 127.0.0.1 -p 1883 -t firebeetle/test -v

# Optional local publish test
mosquitto_pub -h 127.0.0.1 -p 1883 -t firebeetle/test -m "hello from laptop"
```

## How To Use
1. Start phone hotspot.
2. Connect laptop and FireBeetle to hotspot.
3. Find laptop hotspot IP (`ipconfig`).
4. Update `MQTT_BROKER_IP` in `src/main.cpp`.
5. Flash board (`pio run -t upload`).
6. Open serial monitor (`pio device monitor`).
7. Run subscriber on laptop.
8. Verify incoming messages.

## Troubleshooting
- Wrong broker IP: Wi-Fi connects but MQTT fails repeatedly.
- Firewall blocking 1883: allow inbound TCP 1883 for Mosquitto.
- Mosquitto listening only on localhost: use provided `mosquitto.conf` listener.
- Hotspot client isolation: some hotspots block client-to-client traffic.
- Wi-Fi connected but MQTT failed: verify broker is running and IP/port is correct.
