#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <SCServo.h>

// ---- WiFi & UDP --------------------------------------------------------
const char*    WIFI_SSID  = "jingjun_ubuntu";
const char*    WIFI_PASS  = "12345678";
const uint16_t LOCAL_PORT = 5005;

WiFiUDP  udp;
char     packetBuf[512];
IPAddress pcIP;
uint16_t  pcPort  = 0;
bool      pcKnown = false;

// ---- Feetech servos ----------------------------------------------------
SMS_STS  feetech;
// Serial2 pins
#define SERVO_RX_PIN 16
#define SERVO_TX_PIN 17
#define SERVO_BAUD   1000000

const int NUM_SERVOS = 6;
u8  servoIDs[NUM_SERVOS] = {1, 2, 3, 4, 5, 6};

s16 cmdPos[NUM_SERVOS];
u16 cmdSpeed[NUM_SERVOS];
u8  cmdAcc[NUM_SERVOS];

const u16 DEFAULT_SPEED = 4000;   // steps/s
const u8  DEFAULT_ACC   = 250;     // acceleration

// ---- Absolute position -------------------------------------------------
// Firmware returns the full multi-turn position directly from ReadPos.
long absPos[NUM_SERVOS];

// ---- Feedback timer ----------------------------------------------------
unsigned long lastFbMs = 0;
const unsigned long FB_MS = 20;   // 50 Hz poll / feedback rate

// ========================================================================
// Helpers
// ========================================================================

void udpSend(const String& s) {
    if (!pcKnown) return;
    udp.beginPacket(pcIP, pcPort);
    udp.print(s);
    udp.endPacket();
}

void updateServo(int i) {
    int raw = feetech.ReadPos(servoIDs[i]);
    if (feetech.u8Error != 0) return;   // comm error, not a negative position
    absPos[i] = raw;
}

// ========================================================================
// Command handlers
// ========================================================================

// SET_MID – calibrate all servos: current physical position becomes 0
void doSetMid() {
    for (int i = 0; i < NUM_SERVOS; i++) {
        feetech.CalibrationOfs(servoIDs[i]);
        absPos[i] = 0;
    }
    udpSend("SET_MID_OK\n");
}

// R – return current tracked absolute positions
void doRead() {
    String msg = "POS";
    for (int i = 0; i < NUM_SERVOS; i++) {
        msg += ',';
        msg += absPos[i];
    }
    msg += '\n';
    udpSend(msg);
}

// P – position command
// Format: P p0 p1 p2 p3 p4 p5 p6 p7 V v0 v1 v2 v3 v4 v5 v6 v7 END
// V values of 0 mean "run at DEFAULT_SPEED (max)".
bool doPosition(const char* buf) {
    float p[8], v[8];

    int n = sscanf(buf,
        "P %f %f %f %f %f %f %f %f V %f %f %f %f %f %f %f %f",
        &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], &p[7],
        &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);

    if (n != 16) return false;

    for (int i = 0; i < NUM_SERVOS; i++) {
        long clamped = constrain((long)roundf(p[i]), -32768L, 32767L);
        cmdPos[i]   = (s16)clamped;
        int vi      = (int)roundf(v[i]);
        cmdSpeed[i] = (vi == 0) ? DEFAULT_SPEED : (u16)max(1, vi);
    }

    feetech.SyncWritePosEx(servoIDs, NUM_SERVOS, cmdPos, cmdSpeed, cmdAcc);
    return true;
}

// Dispatch one received UDP packet
void processPacket(int len) {
    if (len <= 0) return;
    packetBuf[len] = '\0';

    if (strncmp(packetBuf, "SET_MID", 7) == 0) {
        doSetMid();
    } else if (packetBuf[0] == 'R') {
        doRead();
    } else if (packetBuf[0] == 'P') {
        if (!doPosition(packetBuf))
            udpSend("ERR:bad_P_cmd\n");
    } else {
        udpSend("ERR:unknown_cmd\n");
    }
}

// ========================================================================
// Setup & Loop
// ========================================================================

void setup() {
    Serial.begin(115200);
    Serial2.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN);
    feetech.pSerial = &Serial2;

    for (int i = 0; i < NUM_SERVOS; i++) {
        cmdSpeed[i] = DEFAULT_SPEED;
        cmdAcc[i]   = DEFAULT_ACC;
    }

    // Connect WiFi
    Serial.print("Connecting to WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print('.'); }
    Serial.println("\nConnected! IP: " + WiFi.localIP().toString());

    udp.begin(LOCAL_PORT);
    Serial.printf("UDP listening on port %d\n", LOCAL_PORT);

    // Seed position tracking with current servo positions
    delay(200);
    for (int i = 0; i < NUM_SERVOS; i++) updateServo(i);
    Serial.println("Servo positions initialised.");

    // Enable torque one by one (SMS_STS_TORQUE_ENABLE = addr 40, value 1 = on)
    Serial.println("Enabling torque...");
    for (int i = 0; i < NUM_SERVOS; i++) {
        feetech.EnableTorque(servoIDs[i], 1);
        Serial.printf("  Servo %d: torque ON\n", servoIDs[i]);
        delay(300);
    }
    Serial.println("All servos torqued on.");
}

void loop() {
    // ---- Receive & dispatch UDP commands --------------------------------
    int pktSize = udp.parsePacket();
    if (pktSize > 0) {
        pcIP    = udp.remoteIP();
        pcPort  = udp.remotePort();
        pcKnown = true;
        int len = udp.read(packetBuf, sizeof(packetBuf) - 1);
        processPacket(len);
    }

    // ---- Periodic position tracking + feedback broadcast ---------------
    unsigned long now = millis();
    if (now - lastFbMs >= FB_MS) {
        lastFbMs = now;

        for (int i = 0; i < NUM_SERVOS; i++) updateServo(i);

        // Broadcast current absolute positions to PC
        if (pcKnown) {
            String fb = "FB";
            for (int i = 0; i < NUM_SERVOS; i++) { fb += ','; fb += absPos[i]; }
            fb += '\n';
            udpSend(fb);
        }
    }
}
