from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path


COMMANDS = [
    "STATUS",
    "HOME",
    "G X10 Y20 Z5 M1",
    "X 15",
    "Y -5",
    "Z 10",
    "STATUS",
]


def send_command(sock: socket.socket, command: str) -> str:
    sock.sendall((command + "\n").encode("utf-8"))
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(1024)
        if not chunk:
            raise RuntimeError("controller closed the connection")
        data += chunk
    return data.decode("utf-8", errors="replace").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the simulator and verify the command link.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    simulator = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "automation.simulator",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        deadline = time.time() + 5
        while True:
            try:
                sock = socket.create_connection((args.host, args.port), timeout=1)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.1)

        with sock:
            greeting = sock.recv(1024).decode("utf-8", errors="replace").strip()
            print(f"< {greeting}")
            for command in COMMANDS:
                response = send_command(sock, command)
                print(f"> {command}")
                print(f"< {response}")
                if not response.startswith("OK"):
                    raise RuntimeError(f"command failed: {command}: {response}")
        print("Simulation link check passed.")
        return 0
    finally:
        if args.keep_server:
            print("Simulator left running by request.")
        else:
            simulator.terminate()
            try:
                simulator.wait(timeout=3)
            except subprocess.TimeoutExpired:
                simulator.kill()


if __name__ == "__main__":
    raise SystemExit(main())
