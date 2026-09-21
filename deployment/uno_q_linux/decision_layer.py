"""
decision_layer.py
==================
The clean interface between "what the model said" and "what the hardware
does" (spec section 13). Nothing in here knows about TensorFlow, and
nothing in the ML code knows about servos. If you change your sorting
mechanism (different servo angles, a conveyor belt, extra bins), you edit
ONLY this file -- never src/predict.py or the model.

The actuator commands are sent to the STM32 real-time co-processor over
Arduino's RPC bridge (Arduino App Lab's inter-processor API). The exact
RPC call names depend on the App Lab version you have installed --
check `arduino-app-lab --help` or the App Lab docs on your board once you
have it in hand; the structure below is the part that won't change.
"""
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    label: str
    confidence: float


# Maps a class name (as produced by the model) to a bin/servo position.
# Edit this table -- and only this table -- when you change the physical
# sorting mechanism.
CLASS_TO_BIN = {
    "Metal waste": {"servo_angle": 0, "bin_name": "metal_bin"},
    "Organic waste": {"servo_angle": 90, "bin_name": "organic_bin"},
    "Plastic waste": {"servo_angle": 180, "bin_name": "plastic_bin"},
}

UNKNOWN_SERVO_ANGLE = 90  # park position / reject chute for low-confidence items


class DecisionLayer:
    def __init__(self, rpc_client=None):
        """
        rpc_client: an object exposing a `.call(method, **kwargs)` method
        that talks to the STM32 side over Arduino App Lab's RPC bridge.
        Pass None to run in "dry run" mode (prints what it WOULD do --
        useful for testing the ML side before hardware is wired up).
        """
        self.rpc_client = rpc_client

    def handle(self, result: ClassificationResult):
        if result.label not in CLASS_TO_BIN:
            angle = UNKNOWN_SERVO_ANGLE
            bin_name = "reject_chute"
        else:
            angle = CLASS_TO_BIN[result.label]["servo_angle"]
            bin_name = CLASS_TO_BIN[result.label]["bin_name"]

        self._move_servo(angle, bin_name, result)

    def _move_servo(self, angle, bin_name, result: ClassificationResult):
        if self.rpc_client is None:
            print(f"[DRY RUN] Would move servo to {angle} deg -> {bin_name} "
                  f"(label={result.label}, confidence={result.confidence:.2f})")
            return

        # Real hardware call. The STM32 sketch (see stm32_sketch/) exposes
        # a "set_servo_angle" RPC method -- see stm32_sketch/waste_sorter.ino
        self.rpc_client.call("set_servo_angle", angle=angle)
        print(f"Moved servo to {angle} deg -> {bin_name}")
