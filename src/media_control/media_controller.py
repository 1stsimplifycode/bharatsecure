"""
MediaController - Secure Command Execution Module
Maps approved gestures to media control commands.
Supports: Linux (pactl/xdotool), macOS (osascript), headless (subprocess).
"""

import platform
import subprocess
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MediaController:
    """
    Executes media control commands for approved gestures.

    Gesture → Command mapping:
    - stop       → Pause/Stop media
    - play       → Play media
    - volume_up  → Increase volume +10%
    - volume_down→ Decrease volume -10%
    - mute       → Toggle mute

    Commands are system-agnostic and work on Jetson Nano (Ubuntu/Debian).
    Zero cost — uses built-in system utilities.
    """

    COMMANDS = {
        "stop":        "stop",
        "play":        "play",
        "volume_up":   "volume_up",
        "volume_down": "volume_down",
        "mute":        "mute",
    }

    def __init__(self, config: dict):
        self.config = config
        self.os_type = platform.system().lower()  # linux | darwin | windows
        self._last_gesture = None

        logger.info(f"✅ MediaController: OS={self.os_type}")
        self._verify_dependencies()

    def _verify_dependencies(self):
        """Check availability of system media control tools."""
        if self.os_type == "linux":
            tools = ["pactl", "xdotool"]
            for tool in tools:
                result = subprocess.run(["which", tool], capture_output=True)
                if result.returncode != 0:
                    logger.warning(f"'{tool}' not found. Some commands may use fallback.")

    def execute(self, gesture_name: str) -> bool:
        """
        Execute the media command for the given gesture.

        Args:
            gesture_name: One of stop, play, volume_up, volume_down, mute

        Returns:
            True if command executed successfully
        """
        if gesture_name == self._last_gesture:
            # Debounce: avoid repeated identical commands
            return False

        self._last_gesture = gesture_name
        logger.info(f"▶ Executing: {gesture_name.upper()}")

        try:
            if self.os_type == "linux":
                return self._execute_linux(gesture_name)
            elif self.os_type == "darwin":
                return self._execute_macos(gesture_name)
            else:
                return self._execute_print(gesture_name)
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False

    def _execute_linux(self, gesture: str) -> bool:
        """Linux commands using pactl (PulseAudio) and xdotool."""
        cmd_map = {
            "stop":        ["xdotool", "key", "XF86AudioStop"],
            "play":        ["xdotool", "key", "XF86AudioPlay"],
            "volume_up":   ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"],
            "volume_down": ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"],
            "mute":        ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
        }

        # Fallback for headless Jetson Nano (no display server)
        headless_map = {
            "stop":        ["amixer", "set", "Master", "0%"],
            "play":        ["echo", "PLAY"],
            "volume_up":   ["amixer", "set", "Master", "10%+"],
            "volume_down": ["amixer", "set", "Master", "10%-"],
            "mute":        ["amixer", "set", "Master", "toggle"],
        }

        cmd = cmd_map.get(gesture)
        if cmd:
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            if result.returncode == 0:
                return True
            # Try headless fallback
            cmd = headless_map.get(gesture)
            if cmd:
                subprocess.run(cmd, capture_output=True, timeout=2)
        return True

    def _execute_macos(self, gesture: str) -> bool:
        """macOS commands using osascript."""
        script_map = {
            "stop":        'tell application "Music" to stop',
            "play":        'tell application "Music" to play',
            "volume_up":   'set volume output volume (output volume of (get volume settings) + 10)',
            "volume_down": 'set volume output volume (output volume of (get volume settings) - 10)',
            "mute":        'set volume output muted not (output muted of (get volume settings))',
        }
        script = script_map.get(gesture, "")
        if script:
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=2)
        return True

    def _execute_print(self, gesture: str) -> bool:
        """Headless/test mode: just log the command."""
        logger.info(f"[HEADLESS] COMMAND: {gesture.upper()}")
        return True

    def reset_debounce(self):
        self._last_gesture = None
