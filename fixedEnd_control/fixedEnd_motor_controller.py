"""
feetech_udp_driver.py
Driver for the ESP32-based Feetech servo controller communicating over UDP.

Protocol (all commands end with ' END'):
  Move:      "P p0 p1 p2 p3 p4 p5 p6 p7 V v0 v1 v2 v3 v4 v5 v6 v7 END"
             vi == 0  →  ESP32 uses its default max speed
  Calibrate: "SET_MID END"
  Read:      "R END"

Responses from ESP32:
  "FB,p0,...,p7\n"       periodic 50 Hz feedback
  "POS,p0,...,p7\n"      reply to R command
  "SET_MID_OK\n"         reply to SET_MID
  "ERR:...\n"            error string
"""

import socket
import threading
import time
from typing import List, Optional


class FeetechUDPDriver:
    NUM_MOTORS = 6

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

    def move_to_position(
        self,
        positions: List[int],
        speeds: Optional[List[int]] = None,
    ):
        """
        Send a position command to all 8 motors.

        Args:
            positions: Target absolute positions in steps (multi-turn, s16 range).
                       Must be a list of exactly 8 integers.
            speeds:    Target speeds in steps/s for each motor.
                       Use 0 (or omit) to run at the ESP32's default max speed.
                       Must be a list of exactly 8 integers if provided.

        Raises:
            ValueError: If list lengths are wrong.
        """
        if len(positions) != self.NUM_MOTORS:
            raise ValueError(f"positions must have {self.NUM_MOTORS} elements")
        if speeds is None:
            speeds = [0] * self.NUM_MOTORS
        if len(speeds) != self.NUM_MOTORS:
            raise ValueError(f"speeds must have {self.NUM_MOTORS} elements")

        p_str = " ".join(str(int(p)) for p in positions)
        v_str = " ".join(str(int(v)) for v in speeds)
        self._send(f"P {p_str} V {v_str} END")

    def read_positions(self) -> List[int]:
        """
        Request current absolute positions from the ESP32 and wait for the reply.

        Returns:
            List of 8 absolute positions in steps.

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
    pos = feetech_driver.read_positions()
    print("Positions:", pos)
    time.sleep(0.1)

    # pos = [2048 for _ in range(6)]

    # feetech_driver.move_to_position(pos, [500 for _ in range(6)])


    # feetech_driver.set_mid()
    # time.sleep(0.1)
    print("Reading current positions...")
    pos = feetech_driver.read_positions()
    print("Positions:", pos)

