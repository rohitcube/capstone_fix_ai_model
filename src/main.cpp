#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// =========================
// User-editable settings
// =========================
const char* WIFI_SSID       = "Rohit";
const char* WIFI_PASSWORD   = "rohit123";
const char* MQTT_BROKER_IP  = "172.20.10.4";    // Laptop IP on phone hotspot
const uint16_t MQTT_PORT    = 1883;
const char* MQTT_TOPIC      = "firebeetle/imu/raw";

// *** CHANGE THIS BEFORE FLASHING EACH BOARD ***
// Board 1: "arm_01"    Board 2: "leg_01"
const char* DEVICE_NAME     = "leg_01";

const int SAMPLE_RATE_MS = 20; // 50 Hz

// =========================

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
Adafruit_MPU6050 mpu;

unsigned long lastMqttRetryMs = 0;
unsigned long lastSampleMs = 0;
unsigned long seqNum = 0;

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
  if (now - lastMqttRetryMs < 2000) return;
  lastMqttRetryMs = now;

  String clientId = String(DEVICE_NAME) + "-" + String((uint32_t)(ESP.getEfuseMac() & 0xFFFFFFFF), HEX);

  Serial.printf("[MQTT] Connecting to %s:%u ...\n", MQTT_BROKER_IP, MQTT_PORT);
  if (mqttClient.connect(clientId.c_str())) {
    Serial.println("[MQTT] Connected");
  } else {
    Serial.printf("[MQTT] Connect failed, state=%d. Retrying...\n", mqttClient.state());
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.printf("=== FireBeetle IMU Streamer [%s] ===\n", DEVICE_NAME);

  // Init MPU6050
  if (!mpu.begin()) {
    Serial.println("[IMU] MPU6050 not found! Check wiring.");
    while (1) delay(100);
  }
  Serial.println("[IMU] MPU6050 ready");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
  mqttClient.setBufferSize(512);

  connectWiFi();
  connectMQTT();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  if (!mqttClient.connected()) {
    connectMQTT();
    return;
  }

  mqttClient.loop();

  unsigned long now = millis();
  if (now - lastSampleMs >= SAMPLE_RATE_MS) {
    lastSampleMs = now;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    char buf[400];
    snprintf(buf, sizeof(buf),
      "{\"type\":\"imu_data\",\"device_id\":\"%s\",\"sample_id\":0,\"recording\":false,"
      "\"timestamp\":%lu,\"seq\":%lu,"
      "\"payload\":{\"ax\":%.9f,\"ay\":%.9f,\"az\":%.9f,\"gx\":%.9f,\"gy\":%.9f,\"gz\":%.9f}}",
      DEVICE_NAME, now, seqNum++,
      a.acceleration.x, a.acceleration.y, a.acceleration.z,
      g.gyro.x, g.gyro.y, g.gyro.z);

    mqttClient.publish(MQTT_TOPIC, buf);
  }
}

