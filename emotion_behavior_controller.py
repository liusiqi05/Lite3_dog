#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion behavior controller for DeepRobotics Lite3.

Run this file, type one emotion code, and the matching full sequence will run
before the next input is processed:

1 sad: spin, jump, backflip, yellow light, 1.mp3
2 happy: moonwalk, wave, red light, 2.mp3
3 excited: forward jump, run forward/back, red/yellow light switch, 3.mp3
4 fear: crawl, grip, blue blink, 4.mp3
"""

import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path


CTRL_IP = os.getenv("LITE3_CTRL_IP", "192.168.1.120")
CTRL_PORT = int(os.getenv("LITE3_CTRL_PORT", "43893"))
HEARTBEAT_INTERVAL = float(os.getenv("LITE3_HEARTBEAT_INTERVAL", "0.10"))

# UDP 监听端口（接收来自 camera_demo 的指令）
LISTEN_PORT = int(os.getenv("EMOTION_LISTEN_PORT", "9999"))

LIGHT_COMMAND_CODE = os.getenv("LITE3_LIGHT_CMD", "").strip() or None
LIGHT_SCRIPT = os.getenv("LITE3_LIGHT_SCRIPT", "").strip() or None


class Cmd:
    HEARTBEAT = 0x21040001

    STAND_LIE = 0x21010202
    RETURN_ZERO = 0x21010C05

    IN_PLACE_MODE = 0x21010D05
    MOVE_MODE = 0x21010D06

    LOW_SPEED = 0x21010300
    MEDIUM_SPEED = 0x21010307
    HIGH_SPEED = 0x21010303

    CRAWL_NORMAL = 0x21010406
    GRIP = 0x21010402

    TWIST_BODY = 0x21010204
    MOONWALK = 0x2101030C
    BACKFLIP = 0x21010502
    WAVE = 0x21010507
    FORWARD_JUMP = 0x2101050B
    TWIST_JUMP = 0x2101020D

    FORWARD_BACK = 0x21010130
    TURN = 0x21010135


class Lite3Commander:
    def __init__(self, ctrl_ip=CTRL_IP, ctrl_port=CTRL_PORT):
        self.ctrl_addr = (ctrl_ip, ctrl_port)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.lock = threading.Lock()
        self.running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="lite3_heartbeat", daemon=True
        )
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        while self.running:
            self.send_simple(Cmd.HEARTBEAT, quiet=True)
            time.sleep(HEARTBEAT_INTERVAL)

    def send_simple(self, code, param1=0, param2=0, quiet=False):
        payload = struct.pack("<3i", int(code), int(param1), int(param2))
        with self.lock:
            self.server.sendto(payload, self.ctrl_addr)
        if not quiet:
            print("send: code=0x%08X, param1=%s, param2=%s" % (code, param1, param2))

    def action(self, name, code, wait_s=2.0):
        print("action: %s" % name)
        self.send_simple(code)
        time.sleep(wait_s)

    def set_mode(self, name, code, wait_s=0.35):
        print("mode: %s" % name)
        self.send_simple(code)
        time.sleep(wait_s)

    def set_speed(self, name, code, wait_s=0.25):
        print("speed: %s" % name)
        self.send_simple(code)
        time.sleep(wait_s)

    def stand_up(self, wait_s=2.0):
        print("startup: stand up")
        self.send_simple(Cmd.STAND_LIE)
        time.sleep(wait_s)

    def return_zero(self, wait_s=1.0):
        print("return zero")
        self.send_simple(Cmd.RETURN_ZERO)
        time.sleep(wait_s)

    def hold_motion(self, name, code, value, duration_s, repeat_interval=0.20):
        print("motion: %s %.1fs" % (name, duration_s))
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.send_simple(code, value, 0)
            time.sleep(repeat_interval)
        self.send_simple(code, 0, 0)
        time.sleep(0.25)

    def stop_motion(self):
        self.send_simple(Cmd.FORWARD_BACK, 0, 0)
        self.send_simple(Cmd.TURN, 0, 0)

    def close(self):
        self.running = False
        if self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=1.0)
        try:
            self.stop_motion()
        finally:
            self.server.close()


class AudioPlayer:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir).resolve()

    def music_path(self, index):
        filename = "%s.mp3" % index
        candidates = [
            self.project_dir.parent / "music" / filename,
            self.project_dir.parent / "music" / filename,
            self.project_dir / "music" / filename,
            self.project_dir / "music" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def play(self, index, wait=False):
        path = self.music_path(index)
        if not path.exists():
            print("audio file not found: %s" % path)
            return None

        print("play audio: %s" % path)
        proc = self._start_player(path)
        if proc is not None and wait:
            proc.wait()
        return proc

    def wait(self, proc):
        if proc is not None:
            proc.wait()

    def _start_player(self, path):
        system = platform.system().lower()
        if system == "windows":
            os.startfile(str(path))
            return None

        if system == "darwin":
            return subprocess.Popen(
                ["afplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        players = [
            ("mpg123", ["mpg123", "-q", str(path)]),
            ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]),
            ("cvlc", ["cvlc", "--play-and-exit", "--quiet", str(path)]),
            ("mpv", ["mpv", "--no-video", "--really-quiet", str(path)]),
            ("mplayer", ["mplayer", "-really-quiet", str(path)]),
            ("xdg-open", ["xdg-open", str(path)]),
        ]
        for executable, command in players:
            if shutil.which(executable):
                return subprocess.Popen(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

        print("no audio player found; install mpg123 or ffmpeg/ffplay.")
        return None


class LightController:
    COLORS = {
        "off": (0, 0, 0),
        "red": (255, 0, 0),
        "yellow": (255, 200, 0),
        "blue": (0, 80, 255),
    }

    def __init__(self, commander):
        self.commander = commander
        self._warned = False

    def set_color(self, color, mode="solid"):
        rgb = self.COLORS[color]
        print("light: %s/%s" % (color, mode))

        if LIGHT_SCRIPT:
            subprocess.run([LIGHT_SCRIPT, color, mode], check=False)
            return

        if LIGHT_COMMAND_CODE:
            command_code = int(LIGHT_COMMAND_CODE, 0)
            rgb_value = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
            mode_value = 2 if mode == "blink" else 1
            if color == "off":
                mode_value = 0
            self.commander.send_simple(command_code, rgb_value, mode_value)
            return

        if not self._warned:
            print("notice: real light control is not configured.")
            print("        set LITE3_LIGHT_SCRIPT=/path/to/script if you have one.")
            print("        or set LITE3_LIGHT_CMD=0x........ if you know the LED command.")
            self._warned = True

    def blink(self, color, times=4, interval_s=0.35):
        for _ in range(times):
            self.set_color(color, "blink")
            time.sleep(interval_s)
            self.set_color("off", "solid")
            time.sleep(interval_s)

    def alternate(self, colors, stop_event, interval_s=0.35):
        index = 0
        while not stop_event.is_set():
            self.set_color(colors[index % len(colors)], "solid")
            index += 1
            stop_event.wait(interval_s)


class EmotionBehaviorRunner:
    def __init__(self, commander, lights, audio):
        self.commander = commander
        self.lights = lights
        self.audio = audio
        self.sequence_lock = threading.Lock()

    def run(self, state_code):
        with self.sequence_lock:
            self.commander.stop_motion()
            time.sleep(0.3)
            if state_code == "1":
                self.sad()
            elif state_code == "2":
                self.happy()
            elif state_code == "3":
                self.excited()
            elif state_code == "4":
                self.fear()  # 改为 fear
            else:
                print("invalid code; enter 1, 2, 3, 4, or q to quit.")

    def sad(self):
        print("")
        print("emotion 1 sad: start")
        self.lights.set_color("yellow")
        music = self.audio.play(1)
        self.commander.set_mode("move mode", Cmd.MOVE_MODE)
        self.commander.set_speed("low speed", Cmd.LOW_SPEED)
        self.commander.hold_motion("spin", Cmd.TURN, 32000, 2.8)
        self.commander.return_zero()
        self.commander.set_mode("in-place mode", Cmd.IN_PLACE_MODE)
        self.commander.action("jump (twist jump)", Cmd.TWIST_JUMP, wait_s=2.2)
        self.commander.return_zero()
        self.commander.action("backflip", Cmd.BACKFLIP, wait_s=3.2)
        self.commander.return_zero()
        self.audio.wait(music)
        print("emotion 1 sad: done")
        print("")

    def excited(self):
        print("")
        print("emotion 2 excited: start")
        self.lights.set_color("red")
        music = self.audio.play(3)
        self.commander.set_mode("in-place mode", Cmd.IN_PLACE_MODE)
        self.commander.set_speed("medium speed", Cmd.MEDIUM_SPEED)
        self.commander.action("moonwalk", Cmd.MOONWALK, wait_s=3.0)
        # moonwalk 是持续步态，需要先切回正常步态再归零
        self.commander.set_speed("medium speed", Cmd.MEDIUM_SPEED)
        self.commander.return_zero()
        self.commander.action("twist body", Cmd.TWIST_BODY, wait_s=10.0)
        self.commander.return_zero()
        self.commander.action("wave", Cmd.WAVE, wait_s=3.0)
        self.commander.return_zero()
        self.audio.wait(music)
        print("emotion 2 excited: done")
        print("")

    def happy(self):
        print("")
        print("emotion 3 happy: start")
        stop_light = threading.Event()
        light_thread = threading.Thread(
            target=self.lights.alternate,
            args=(("red", "yellow"), stop_light),
            name="excited_light_switch",
            daemon=True,
        )
        light_thread.start()
        music = self.audio.play(2)
        try:
            self.commander.set_mode("in-place mode", Cmd.IN_PLACE_MODE)
            self.commander.set_speed("high speed", Cmd.HIGH_SPEED)
            self.commander.action("forward jump", Cmd.FORWARD_JUMP, wait_s=2.2)
            self.commander.return_zero()
            self.commander.set_mode("move mode", Cmd.MOVE_MODE)
            for cycle in range(2):
                print("run forward/back cycle %d" % (cycle + 1))
                self.commander.hold_motion("run forward", Cmd.FORWARD_BACK, 30000, 1.8)
                self.commander.return_zero()
                self.commander.hold_motion("run backward", Cmd.FORWARD_BACK, -30000, 1.8)
                self.commander.return_zero()
            self.commander.stop_motion()
            self.audio.wait(music)
        finally:
            stop_light.set()
            light_thread.join(timeout=1.0)
        print("emotion 3 happy: done")
        print("")

    def fear(self):  # 原 alert 改为 fear
        print("")
        print("emotion 4 fear: start")
        self.commander.set_mode("in-place mode", Cmd.IN_PLACE_MODE)
        self.commander.set_speed("normal/crawl gait", Cmd.CRAWL_NORMAL)
        self.commander.action("twist jump", Cmd.TWIST_JUMP, wait_s=2.2)
        self.commander.return_zero()
        self.commander.action("crawl", Cmd.CRAWL_NORMAL, wait_s=1.5)
        self.commander.return_zero()
        self.commander.action("grip", Cmd.GRIP, wait_s=1.5)
        self.commander.return_zero()
        self.lights.blink("blue", times=6, interval_s=0.35)
        self.audio.play(4, wait=True)
        print("emotion 4 fear: done")
        print("")


class UDPListener:
    """监听来自 camera_demo 的 UDP 指令"""
    def __init__(self, runner, listen_port=LISTEN_PORT):
        self.runner = runner
        self.listen_port = listen_port
        self.running = True
        self.sock = None
        self.thread = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.listen_port))
        self.sock.settimeout(0.5)
        self.thread = threading.Thread(target=self._listen_loop, name="udp_listener", daemon=True)
        self.thread.start()
        print(f"UDP 监听已启动，端口 {self.listen_port}，等待情绪指令...")

    def _listen_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                msg = data.decode().strip()
                if msg in ('1', '2', '3', '4'):
                    print(f"\n[UDP] 收到来自 {addr[0]}:{addr[1]} 的指令: {msg}")
                    # 在新线程中执行，避免阻塞监听
                    t = threading.Thread(target=self.runner.run, args=(msg,), daemon=True)
                    t.start()
                else:
                    print(f"[UDP] 收到未知指令: {msg}")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[UDP] 错误: {e}")

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        if self.thread:
            self.thread.join(timeout=1.0)


def main():
    project_dir = Path(__file__).resolve().parent
    commander = Lite3Commander()
    lights = LightController(commander)
    audio = AudioPlayer(project_dir)
    runner = EmotionBehaviorRunner(commander, lights, audio)

    # 启动 UDP 监听（接收来自 camera_demo 的指令）
    listener = UDPListener(runner)
    listener.start()

    print("=" * 60)
    print("情绪行为控制器已启动")
    print(f"机器人控制地址: {CTRL_IP}:{CTRL_PORT}")
    print(f"UDP 监听端口: {LISTEN_PORT} (等待 camera_demo 发送指令)")
    print("")
    print("本地手动控制: 1=sad, 2=happy, 3=excited, 4=fear, q=quit")
    print("=" * 60)

    commander.stand_up()

    try:
        while True:
            state = input("state code: ").strip()
            if state.lower() in ("q", "quit", "exit"):
                break
            runner.run(state)
    except KeyboardInterrupt:
        print("")
        print("Ctrl+C received; exiting.")
    finally:
        listener.stop()
        commander.close()


if __name__ == "__main__":
    sys.exit(main())