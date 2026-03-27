#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>

// =========================
// User-editable settings
// =========================
const char* WIFI_SSID       = "YOUR_HOTSPOT_SSID";
const char* WIFI_PASSWORD   = "YOUR_HOTSPOT_PASSWORD";
const char* MQTT_BROKER_IP  = "192.168.43.100";   // Laptop hotspot IP
const uint16_t MQTT_PORT    = 1883;
const char* MQTT_TOPIC      = "firebeetle/test";
const char* DEVICE_NAME     = "firebeetle-esp32";

// =========================

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

unsigned long lastPublishMs = 0;
unsigned long lastMqttRetryMs = 0;
uint32_t counter = 0;

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.printf("[WiFi] Connecting to SSID: %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  Serial.println("[WiFi] Connected");
  Serial.print("[WiFi] Local IP: ");
  Serial.println(WiFi.localIP());
}

void connectMQTT() {
  if (mqttClient.connected()) return;

  unsigned long now = millis();
  if (now - lastMqttRetryMs < 2000) return; // simple retry spacing
  lastMqttRetryMs = now;

  String clientId = String(DEVICE_NAME) + "-" + String((uint32_t)(ESP.getEfuseMac() & 0xFFFFFFFF), HEX);

  Serial.printf("[MQTT] Connecting to %s:%u ...\n", MQTT_BROKER_IP, MQTT_PORT);
  if (mqttClient.connect(clientId.c_str())) {
    Serial.println("[MQTT] Connected");
  } else {
    Serial.printf("[MQTT] Connect failed, state=%d. Retrying...\n", mqttClient.state());
  }
}

// Placeholder for future sensor read
int readSensorPlaceholder() {
  return 0;
}

void publishMessage() {
  int rssi = WiFi.RSSI();
  String payload = "{\"device\":\"" + String(DEVICE_NAME) +
                   "\",\"counter\":" + String(counter) +
                   ",\"rssi\":" + String(rssi) + "}";

  bool ok = mqttClient.publish(MQTT_TOPIC, payload.c_str());
  if (ok) {
    Serial.printf("[MQTT] Published to %s: %s\n", MQTT_TOPIC, payload.c_str());
    counter++;
  } else {
    Serial.println("[MQTT] Publish failed");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== FireBeetle MQTT Demo ===");

  mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);

  connectWiFi();
  connectMQTT();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Disconnected. Reconnecting...");
    connectWiFi();
  }

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.connected()) {
      connectMQTT();
    } else {
      mqttClient.loop();

      unsigned long now = millis();
      if (now - lastPublishMs >= 1000) {
        lastPublishMs = now;
        publishMessage();
      }
    }
  }

  delay(10);
}
