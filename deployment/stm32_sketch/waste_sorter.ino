/*
 * waste_sorter.ino
 * ================
 * Runs on the UNO Q's STM32U585 real-time co-processor.
 *
 * This sketch does NOT run the ML model -- the UNO Q's Qualcomm/Linux side
 * has 2GB of RAM and runs the full TensorFlow Lite classifier there
 * (see ../uno_q_linux/infer.py). This side just owns the physical servo
 * and exposes a single RPC method, set_servo_angle(), that the Linux side
 * calls once it has decided which bin an item belongs in.
 *
 * This split matches spec section 13 (decision layer / actuator
 * separation): change the servo wiring or add more bins here, without
 * touching any ML code.
 *
 * Wiring: servo signal pin -> D9 (or any PWM-capable pin on the UNO
 * headers), servo power from a separate 5V supply (not the board's 5V
 * rail, to avoid brownouts when the servo moves), grounds tied together.
 *
 * NOTE: the exact RPC registration macro/API (RPC_EXPOSE, Bridge.expose,
 * etc.) depends on the Arduino App Lab / Arduino_RPC library version on
 * your board -- check the App Lab examples that ship with your UNO Q for
 * the current syntax and adjust the two marked lines below.
 */

#include <Servo.h>
// #include <Arduino_RPC.h>  // <-- exact include name depends on App Lab version

Servo sortingServo;
const int SERVO_PIN = 9;

void setup() {
  Serial.begin(115200);
  sortingServo.attach(SERVO_PIN);
  sortingServo.write(90);  // neutral/park position at boot

  // Register the RPC method the Linux side calls.
  // ADAPT THIS LINE to your App Lab version's RPC registration API:
  //   RPC.bind("set_servo_angle", set_servo_angle);
  Serial.println("waste_sorter ready, waiting for RPC calls...");
}

void loop() {
  // Real-time side stays idle otherwise -- all decision logic lives on
  // the Linux/Python side. Add sensor polling (e.g. an IR break-beam to
  // detect when an item is in position) here if your mechanism needs it.
  delay(10);
}

// Called via RPC from infer.py's decision_layer.py whenever the Linux
// side has classified an item and decided where it should go.
void set_servo_angle(int angle) {
  angle = constrain(angle, 0, 180);
  sortingServo.write(angle);
  Serial.print("Servo moved to: ");
  Serial.println(angle);
}
