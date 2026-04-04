#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// =========================
// User-editable settings
// =========================
const char* WIFI_SSID       = "iPhone";
const char* WIFI_PASSWORD   = "Sasikala";
const char* MQTT_BROKER_IP  = "172.20.10.2";
const uint16_t MQTT_PORT    = 8883;
const char* MQTT_TOPIC      = "firebeetle/raw";

// *** CHANGE THIS BEFORE FLASHING EACH BOARD ***
// Board 1: "arm"    Board 2: "leg"
const char* DEVICE_NAME     = "arm"
;

const int SAMPLE_RATE_MS = 20; // 50 Hz

// =========================

// CA certificate for TLS verification against Mosquitto broker
const char* ca_cert = R"EOF(
-----BEGIN CERTIFICATE-----
MIIDUTCCAjmgAwIBAgIUEK5deI7+i0ekbJHefg16Ql8PNaAwDQYJKoZIhvcNAQEL
BQAwODELMAkGA1UEBhMCU0cxETAPBgNVBAoMCExvY2FsRGV2MRYwFAYDVQQDDA1N
eU1vc3F1aXR0b0NBMB4XDTI2MDEyNjA0MDkyOVoXDTM2MDEyNDA0MDkyOVowODEL
MAkGA1UEBhMCU0cxETAPBgNVBAoMCExvY2FsRGV2MRYwFAYDVQQDDA1NeU1vc3F1
aXR0b0NBMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuhDyLQ3iDYQV
v1Bsc5E7cHOIu19S5+CBbuFFnzG8gUPZYkN3AA6EQMTh45l41a6n3okW2xEdbO06
fIlsgSaqB6o6KkUFe6Sc7ZC6A8aDf7qkwYuXban3muqwOwNMp2QdITPaTo2brjh0
oZk4O/1Zxh2RjSdkfUI5JdPalcg9JdOcVouSMhkKdmX+tIwN67gP/ldtOg4JJWwn
qIhj8OSYIcXbxddsQp5bIQ8ep8ynMZhYdKFf7BxeS2J0XQv2mAaGjTL9S70KSeVg
cK5qrodG6WUlJYixcuuRjf7G66DlC24kq7QCQLYgvdJak5naCo28fo+n0Oncq/kl
dXnMWf/SBQIDAQABo1MwUTAdBgNVHQ4EFgQUdjf9GJZEG2XPWdfyPD303jov3Jow
HwYDVR0jBBgwFoAUdjf9GJZEG2XPWdfyPD303jov3JowDwYDVR0TAQH/BAUwAwEB
/zANBgkqhkiG9w0BAQsFAAOCAQEAeVVwGqTLPoCnK4w1N7Wah9XQfrox2D6pMATX
Ew2Yv8VYBIFvCxnB7YzBPiAedXZ9emIxkxZZWIk6ak3rmL8quAf5pHId0nmhPG4U
RwfGX9OIEowpLHR5M5LW9DwSuDxKph2WU5oIAwtuchwmvo6vzTKBvsPRiV85xEG7
BIM8/bY3rX1da977wtqbCqD7s28QDIk7vVgVz3+Gb0j/2Fc1QeBj/pCO8vWKpb+k
kByPMd0VcMRFIi4JAhuX0M8BRjMX6iKGQ0KhidX+eeN83BzTc54RWxZci6hDUfHK
jb42BVFBt2ENzXxUOJDhEmVps1ovr4VuLpMWayqcAMtQJn7/qA==
-----END CERTIFICATE-----
)EOF";

WiFiClientSecure wifiClient;
PubSubClient mqttClient(wifiClient);
Adafruit_MPU6050 mpu;

unsigned long lastMqttRetryMs = 0;
unsigned long lastSampleMs = 0;

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

  wifiClient.setInsecure();  // Skip cert verification for now

  mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
  mqttClient.setBufferSize(256);

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

    // Format: device_name,timestamp,ax,ay,az,gx,gy,gz
    char buf[150];
    snprintf(buf, sizeof(buf), "%s,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
             DEVICE_NAME, now,
             a.acceleration.x, a.acceleration.y, a.acceleration.z,
             g.gyro.x, g.gyro.y, g.gyro.z);

    mqttClient.publish(MQTT_TOPIC, buf);
  }
}
