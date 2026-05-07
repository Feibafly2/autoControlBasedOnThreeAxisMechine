/*
  Arduino Uno R3 motion controller draft for Feibafly2 dual-Y three-axis slider.

  Pinout:
    X: PUL D2, DIR D3, ENA D4
    Y: PUL D5, DIR D6, ENA D7
    Z: PUL D8, DIR D9, ENA D10

  TB6600 enable is active-high in this build.

  Serial protocol, one command per line:
    STATUS
    HOME
    O
    X 10
    Y -5
    Z 2
    G X10 Y20 Z5 M1

  This firmware is a safe starting point, not a final motion controller.
  Verify limit switches and motor direction before running long moves.
*/

struct Axis {
  const char* name;
  uint8_t pul;
  uint8_t dir;
  uint8_t ena;
  float minMm;
  float maxMm;
  float positionMm;
};

const float STEPS_PER_MM = 200.0;
const unsigned int STEP_PULSE_US = 6;
const unsigned int STEP_INTERVAL_US = 700;
const bool ENABLE_ACTIVE_HIGH = true;

Axis axisX = {"X", 2, 3, 4, 0.0, 300.0, 0.0};
Axis axisY = {"Y", 5, 6, 7, 0.0, 300.0, 0.0};
Axis axisZ = {"Z", 8, 9, 10, 0.0, 130.0, 0.0};

void setupAxis(Axis& axis) {
  pinMode(axis.pul, OUTPUT);
  pinMode(axis.dir, OUTPUT);
  pinMode(axis.ena, OUTPUT);
  digitalWrite(axis.pul, LOW);
  digitalWrite(axis.dir, LOW);
  digitalWrite(axis.ena, ENABLE_ACTIVE_HIGH ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);
  setupAxis(axisX);
  setupAxis(axisY);
  setupAxis(axisZ);
  Serial.println("OK ARDUINO READY");
}

void pulseStep(const Axis& axis) {
  digitalWrite(axis.pul, HIGH);
  delayMicroseconds(STEP_PULSE_US);
  digitalWrite(axis.pul, LOW);
  delayMicroseconds(STEP_INTERVAL_US);
}

bool moveAxisTo(Axis& axis, float targetMm) {
  if (targetMm < axis.minMm || targetMm > axis.maxMm) {
    Serial.print("ERROR ");
    Serial.print(axis.name);
    Serial.println(" OUT_OF_RANGE");
    return false;
  }

  float delta = targetMm - axis.positionMm;
  bool positive = delta >= 0;
  digitalWrite(axis.dir, positive ? HIGH : LOW);

  long steps = lround(abs(delta) * STEPS_PER_MM);
  for (long i = 0; i < steps; i++) {
    pulseStep(axis);
  }

  axis.positionMm = targetMm;
  return true;
}

void printStatus() {
  Serial.print("OK STATUS X");
  Serial.print(axisX.positionMm, 2);
  Serial.print(" Y");
  Serial.print(axisY.positionMm, 2);
  Serial.print(" Z");
  Serial.println(axisZ.positionMm, 2);
}

float readAxisValue(const String& command, char axisName, float fallback) {
  int idx = command.indexOf(axisName);
  if (idx < 0) {
    idx = command.indexOf((char)tolower(axisName));
  }
  if (idx < 0) {
    return fallback;
  }
  int start = idx + 1;
  while (start < command.length() && command[start] == ' ') {
    start++;
  }
  int end = start;
  while (end < command.length() && (isDigit(command[end]) || command[end] == '-' || command[end] == '.')) {
    end++;
  }
  return command.substring(start, end).toFloat();
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "STATUS") {
    printStatus();
    return;
  }

  if (command == "HOME") {
    axisX.positionMm = 0.0;
    axisY.positionMm = 0.0;
    axisZ.positionMm = 0.0;
    Serial.println("OK HOME");
    return;
  }

  if (command == "O") {
    axisX.positionMm = 0.0;
    axisY.positionMm = 0.0;
    axisZ.positionMm = 0.0;
    Serial.println("OK ORIGIN");
    return;
  }

  if (command.startsWith("X ")) {
    if (moveAxisTo(axisX, axisX.positionMm + command.substring(2).toFloat())) Serial.println("OK MOVE X");
    return;
  }

  if (command.startsWith("Y ")) {
    if (moveAxisTo(axisY, axisY.positionMm + command.substring(2).toFloat())) Serial.println("OK MOVE Y");
    return;
  }

  if (command.startsWith("Z ")) {
    if (moveAxisTo(axisZ, axisZ.positionMm + command.substring(2).toFloat())) Serial.println("OK MOVE Z");
    return;
  }

  if (command.startsWith("G ")) {
    float x = readAxisValue(command, 'X', axisX.positionMm);
    float y = readAxisValue(command, 'Y', axisY.positionMm);
    float z = readAxisValue(command, 'Z', axisZ.positionMm);
    if (!moveAxisTo(axisZ, z)) return;
    if (!moveAxisTo(axisX, x)) return;
    if (!moveAxisTo(axisY, y)) return;
    Serial.println("OK G");
    return;
  }

  Serial.print("ERROR UNKNOWN_COMMAND ");
  Serial.println(command);
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  handleCommand(command);
}

