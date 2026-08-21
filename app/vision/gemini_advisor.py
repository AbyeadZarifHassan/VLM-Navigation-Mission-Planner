"""Gemini advisor for intelligent obstacle avoidance decisions."""

import os
import json
import math


class GeminiAdvisor:
    """Sends telemetry + depth data to Gemini and parses avoidance commands."""

    MODEL = "gemini-2.0-flash"

    SYSTEM_PROMPT = (
        "You are an autonomous drone obstacle avoidance system. "
        "You receive current flight telemetry and depth sensor readings. "
        "An obstacle has been detected ahead. Your job is to decide the best "
        "avoidance maneuver so the drone can safely navigate around the obstacle.\n\n"
        "Respond ONLY with a JSON object — no markdown, no extra text:\n"
        '{"vx": float, "vy": float, "vz": float, "duration": float, "reasoning": "brief explanation"}\n\n'
        "Velocity is in NED body frame:\n"
        "- vx: forward(+) / backward(-), range -1.0 to 1.0 m/s\n"
        "- vy: right(+) / left(-), range -1.0 to 1.0 m/s\n"
        "- vz: down(+) / up(-), range -1.0 to 1.0 m/s\n"
        "- duration: how long to execute in seconds, range 1.0 to 5.0\n\n"
        "Guidelines:\n"
        "- If left distance > right distance, move left (vy negative).\n"
        "- If right distance > left distance, move right (vy positive).\n"
        "- If both sides are blocked, move backward (vx negative).\n"
        "- Prefer lateral moves over altitude changes when possible.\n"
        "- If altitude is low (< 2m), do NOT move down (vz must be <= 0).\n"
        "- Keep velocities conservative (0.3 to 0.8 m/s).\n"
        "- Keep reasoning under 20 words."
    )

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy-init Gemini client."""
        if self._client is None:
            import google.generativeai as genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(
                model_name=self.MODEL,
                system_instruction=self.SYSTEM_PROMPT,
            )
        return self._client

    def _build_user_prompt(self, telemetry, distance, distances, threshold):
        """Build the text portion of the user message."""
        alt = telemetry.get("altitude", 0.0)
        heading = telemetry.get("heading", 0)
        roll = math.degrees(telemetry.get("roll", 0.0))
        pitch = math.degrees(telemetry.get("pitch", 0.0))
        yaw = math.degrees(telemetry.get("yaw", 0.0))
        speed = telemetry.get("groundspeed", 0.0)
        mode = telemetry.get("mode", "UNKNOWN")
        dist_left = (distances or {}).get("left", 20.0) or 20.0
        dist_right = (distances or {}).get("right", 20.0) or 20.0

        return (
            f"Obstacle detected at {distance:.2f}m (threshold: {threshold}m).\n\n"
            f"Depth sensor readings:\n"
            f"- Center distance: {distance:.2f}m\n"
            f"- Left distance: {dist_left:.2f}m\n"
            f"- Right distance: {dist_right:.2f}m\n\n"
            f"Current telemetry:\n"
            f"- Altitude: {alt:.1f}m\n"
            f"- Heading: {heading} deg\n"
            f"- Roll: {roll:.1f} deg | Pitch: {pitch:.1f} deg | Yaw: {yaw:.1f} deg\n"
            f"- Ground Speed: {speed:.1f} m/s\n"
            f"- Mode: {mode}\n\n"
            f"Decide the avoidance maneuver based on the depth readings and telemetry."
        )

    def ask(self, telemetry, distance, distances=None, threshold=1.0):
        """Send telemetry + depth data to Gemini and return avoidance command.

        Args:
            telemetry: dict from MAVLinkComm.get_telemetry()
            distance: forward center distance in meters
            distances: dict with 'left', 'center', 'right' distances
            threshold: obstacle threshold in meters

        Returns:
            dict: {"vx", "vy", "vz", "duration", "reasoning"} or None on failure
        """
        client = self._get_client()
        user_text = self._build_user_prompt(telemetry, distance, distances, threshold)

        prompt_summary = (
            f"dist={distance:.2f}m alt={telemetry.get('altitude', 0):.1f}m "
            f"hdg={telemetry.get('heading', 0)}° "
            f"roll={math.degrees(telemetry.get('roll', 0)):.0f}° "
            f"pitch={math.degrees(telemetry.get('pitch', 0)):.0f}° "
            f"yaw={math.degrees(telemetry.get('yaw', 0)):.0f}°"
        )

        print(f"[GEMINI] Sending telemetry -> {self.MODEL} ({prompt_summary})")

        response = client.generate_content(
            user_text,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 200,
                "response_mime_type": "application/json",
            },
        )

        raw = response.text
        if not raw or not raw.strip():
            print("[GEMINI] Empty response")
            return None

        raw = raw.strip()
        print(f"[GEMINI] Raw response: {raw}")

        text = raw
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        if not text.startswith("{"):
            print(f"[GEMINI] Non-JSON response: {text[:80]}")
            return None

        cmd = json.loads(text)

        for key in ("vx", "vy", "vz", "duration"):
            if key not in cmd:
                print(f"[GEMINI] Missing key '{key}' in response")
                return None

        cmd["vx"] = max(-1.0, min(1.0, float(cmd["vx"])))
        cmd["vy"] = max(-1.0, min(1.0, float(cmd["vy"])))
        cmd["vz"] = max(-1.0, min(1.0, float(cmd["vz"])))
        cmd["duration"] = max(1.0, min(5.0, float(cmd["duration"])))
        cmd["reasoning"] = str(cmd.get("reasoning", ""))

        dirs = []
        if cmd["vx"] < -0.1:
            dirs.append("BACKWARD")
        elif cmd["vx"] > 0.1:
            dirs.append("FORWARD")
        if cmd["vy"] < -0.1:
            dirs.append("LEFT")
        elif cmd["vy"] > 0.1:
            dirs.append("RIGHT")
        if cmd["vz"] < -0.1:
            dirs.append("UP")
        elif cmd["vz"] > 0.1:
            dirs.append("DOWN")
        cmd["direction"] = "-".join(dirs) if dirs else "HOVER"

        print(
            f"[GEMINI] >> {cmd['direction']} "
            f"vx={cmd['vx']:.2f} vy={cmd['vy']:.2f} vz={cmd['vz']:.2f} "
            f"dur={cmd['duration']:.1f}s | {cmd['reasoning']}"
        )

        return cmd
