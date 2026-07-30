"""
feetech_udp_driver.py
Driver for the ESP32-based Feetech servo controller communicating over UDP.

Protocol (all commands end with ' END'):
  Move:      "P p0 ... pN V v0 ... vN END"
             vi == 0  →  ESP32 uses its default max speed
  Calibrate: "SET_MID END"
  Read:      "R END"

Responses from ESP32:
  "FB,p0,...,pN\n"       periodic 50 Hz feedback
  "POS,p0,...,pN\n"      reply to R command
  "SET_MID_OK\n"         reply to SET_MID
  "ERR:...\n"            error string
"""

import socket
import threading
import time
from typing import List, Optional
from tkinter import *

class FeetechUDPDriver:
    NUM_MOTORS = 6
    # Current ESP32 firmware parses a legacy 8-slot P command even though only
    # the first NUM_MOTORS slots are used.
    COMMAND_MOTORS = 8

    def __init__(
        self,
        esp32_ip: str = '10.42.0.12',
        esp32_port: int = 5005,
        local_port: int = 5006,
        timeout: float = 2.0,
    ):
        """
        Args:
            esp32_ip:    IP address of the ESP32.
            esp32_port:  UDP port the ESP32 listens on (default 5005).
            local_port:  Local UDP port for receiving ESP32 replies / FB.
            timeout:     Seconds to wait for a reply before raising TimeoutError.
        """
        self.esp32_addr = (esp32_ip, esp32_port)
        self.timeout = timeout

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", local_port))
        self._sock.settimeout(0.05)  # short timeout for the recv loop

        # Latest absolute positions received from FB or POS packets
        self._positions: List[int] = [0] * self.NUM_MOTORS
        self._pos_lock = threading.Lock()

        # Event set whenever a POS reply arrives
        self._pos_event = threading.Event()
        self._pos_reply: List[int] = [0] * self.NUM_MOTORS

        # Event set whenever SET_MID_OK arrives
        self._mid_event = threading.Event()

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(self, msg: str):
        self._sock.sendto(msg.encode(), self.esp32_addr)

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break
            line = data.decode(errors="ignore").strip()
            self._parse(line)

    def _parse(self, line: str):
        if line.startswith("FB,"):
            parts = line[3:].split(",")
            if len(parts) == self.NUM_MOTORS:
                vals = [int(p) for p in parts]
                with self._pos_lock:
                    self._positions = vals
        elif line.startswith("POS,"):
            parts = line[4:].split(",")
            if len(parts) == self.NUM_MOTORS:
                self._pos_reply = [int(p) for p in parts]
                with self._pos_lock:
                    self._positions = self._pos_reply[:]
                self._pos_event.set()
        elif line == "SET_MID_OK":
            self._mid_event.set()
        elif line.startswith("ERR"):
            print(f"[FeetechUDP] ESP32 error: {line}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def move_all_motors_manual(
        self,
        motor_ids=None,
        position_span: int = 1000,
        speed: int = 0,
        command_interval_ms: int = 80,
    ) -> None:
        """Open a Tkinter slider panel for manual absolute-position control.

        ``motor_ids`` are zero-based indices in the UDP position array, not the
        physical Feetech bus IDs. Each slider starts at the current motor
        position and spans ``current +/- position_span`` steps.
        """
        if motor_ids is None:
            motor_ids = list(range(self.NUM_MOTORS))
        motor_ids = [int(motor_id) for motor_id in motor_ids]
        invalid_ids = [
            motor_id for motor_id in motor_ids
            if motor_id < 0 or motor_id >= self.NUM_MOTORS
        ]
        if invalid_ids:
            raise ValueError(
                f"motor_ids must be zero-based indices from 0 to {self.NUM_MOTORS - 1}; "
                f"invalid ids: {invalid_ids}"
            )

        cur_poses = self.read_positions()
        target_positions = cur_poses[:]
        pending_send = {"job": None}

        root = Tk()
        root.title("Fixed-end motor position controller")

        def send_positions():
            pending_send["job"] = None
            self.move_to_position(target_positions, [int(speed)] * self.NUM_MOTORS)

        def schedule_send():
            if pending_send["job"] is not None:
                root.after_cancel(pending_send["job"])
            pending_send["job"] = root.after(command_interval_ms, send_positions)

        def make_move_callback(motor_id: int, value_label):
            def move_to(position):
                target_positions[motor_id] = int(float(position))
                value_label.config(text=str(target_positions[motor_id]))
                schedule_send()
            return move_to

        for row, motor_id in enumerate(motor_ids):
            cur_pos = int(cur_poses[motor_id])
            frame = Frame(root)
            frame.pack(fill=X, padx=8, pady=4)
            value_label = Label(frame, text=str(cur_pos), width=8, anchor=E)
            value_label.pack(side=RIGHT, padx=(6, 0))
            scale_i = Scale(
                frame,
                from_=cur_pos - int(position_span),
                to=cur_pos + int(position_span),
                orient=HORIZONTAL,
                length=400,
                resolution=1,
                label=f"motor index {motor_id}",
            )
            scale_i.set(cur_pos)
            scale_i.config(command=make_move_callback(motor_id, value_label))
            scale_i.pack(side=LEFT, fill=X, expand=True)

        Button(root, text="Send now", command=send_positions).pack(fill=X, padx=8, pady=(8, 4))
        root.mainloop()

    def move_single_motor_abs(self, position: int, motor_id: int):
        """
        Move a single motor to an absolute position.

        Args:
            position: Target absolute position in steps (multi-turn, s16 range).
            motor_id: Zero-based motor index to move.
        """
        if not (0 <= motor_id < self.NUM_MOTORS):
            raise ValueError(f"motor_id must be between 0 and {self.NUM_MOTORS - 1}")
        cur_position = self.read_positions()
        cur_position[motor_id] = position
        self.move_to_position(cur_position)

    def move_to_position(
        self,
        positions: List[int],
        speeds: Optional[List[int]] = None,
    ):
        """
        Send a position command to all motors.

        Args:
            positions: Target absolute positions in steps (multi-turn, s16 range).
                       Must have NUM_MOTORS integers.
            speeds:    Target speeds in steps/s for each motor.
                       Use 0 (or omit) to run at the ESP32's default max speed.
                       Must have NUM_MOTORS integers if provided.

        Raises:
            ValueError: If list lengths are wrong.
        """
        if len(positions) != self.NUM_MOTORS:
            raise ValueError(f"positions must have {self.NUM_MOTORS} elements")
        if speeds is None:
            speeds = [0] * self.NUM_MOTORS
        if len(speeds) != self.NUM_MOTORS:
            raise ValueError(f"speeds must have {self.NUM_MOTORS} elements")

        command_positions = list(positions) + [0] * (self.COMMAND_MOTORS - self.NUM_MOTORS)
        command_speeds = list(speeds) + [0] * (self.COMMAND_MOTORS - self.NUM_MOTORS)
        p_str = " ".join(str(int(p)) for p in command_positions)
        v_str = " ".join(str(int(v)) for v in command_speeds)
        self._send(f"P {p_str} V {v_str} END")

    def read_positions(self) -> List[int]:
        """
        Request current absolute positions from the ESP32 and wait for the reply.

        Returns:
            List of NUM_MOTORS absolute positions in steps.

        Raises:
            TimeoutError: If no reply arrives within self.timeout seconds.
        """
        self._pos_event.clear()
        self._send("R END")
        if not self._pos_event.wait(timeout=self.timeout):
            raise TimeoutError("Timed out waiting for position reply from ESP32")
        return self._pos_reply[:]

    def get_latest_positions(self) -> List[int]:
        """
        Return the most recently received positions (from FB or POS packets)
        without sending a new request.
        """
        with self._pos_lock:
            return self._positions[:]

    def set_mid(self):
        """
        Calibrate all servos: set the current physical position as the midpoint
        (register value 2048).  Blocks until the ESP32 confirms or times out.

        Raises:
            TimeoutError: If no confirmation arrives within self.timeout seconds.
        """
        self._mid_event.clear()
        self._send("SET_MID END")
        if not self._mid_event.wait(timeout=self.timeout):
            raise TimeoutError("Timed out waiting for SET_MID_OK from ESP32")
        print("[FeetechUDP] All servos calibrated (mid set).")

    def close(self):
        """Stop the receive thread and close the socket."""
        self._running = False
        self._recv_thread.join(timeout=1.0)
        self._sock.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ----------------------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # esp_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.100"
    feetech_driver = FeetechUDPDriver()
    print("Reading current positions...")
    initial_pos = feetech_driver.read_positions()
    print("Positions:", initial_pos)
    time.sleep(0.1)

    # tp = initial_pos.copy()

    # tp[0] += 4000
    # feetech_driver.move_to_position(tp, [500 for _ in range(6)])

    # pos = [2048 for _ in range(6)]

    # feetech_driver.move_to_position(pos, [500 for _ in range(6)])
    # input("Press Enter to return to initial positions...")

    # feetech_driver.set_mid()
    # time.sleep(0.1)
    print("Reading current positions...")
    pos = feetech_driver.read_positions()
    print("Positions:", pos)
    # feetech_driver.move_to_position(initial_pos, [500 for _ in range(6)])
