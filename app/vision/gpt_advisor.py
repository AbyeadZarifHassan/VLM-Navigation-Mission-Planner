"""GPT-4o Vision advisor for obstacle avoidance.

Discrete-action design: VLM picks ONE direction from a fixed action set.
Code translates direction -> (vx, vy, vz, duration) using context-adaptive
velocity scaling. This eliminates the failure mode where VLMs default to
backward motion as a "safety choice" — backward is not in the action set.

The drone is already HOVERING when the VLM is queried, so backward motion
gains nothing for safety. The "stuck" case (no clear path anywhere) is
handled by the emergency_climb action.
"""

import os
import json
import math
import base64
from io import BytesIO


# ── Action space → velocity translation ───────────────────────

# Each direction returns (vx_factor, vy_factor, vz_factor, base_duration)
# where vx is scaled adaptively by the center distance at translation time.
ACTION_SPACE = {
    "forward":       {"vx": +1.0, "vy":  0.0, "vz":  0.0, "dur": 2.5},
    "backward":      {"vx": -1.0, "vy":  0.0, "vz":  0.0, "dur": 1.5},  # capped: max 0.6m retreat
    "left":          {"vx": +0.4, "vy": -1.0, "vz":  0.0, "dur": 2.5},
    "right":         {"vx": +0.4, "vy": +1.0, "vz":  0.0, "dur": 2.5},
    "up":            {"vx": +0.3, "vy":  0.0, "vz": -1.0, "dur": 2.5},
    "down":          {"vx": +0.4, "vy":  0.0, "vz": +0.7, "dur": 2.5},
    "upper_left":    {"vx": +0.3, "vy": -0.8, "vz": -0.6, "dur": 2.5},
    "upper_right":   {"vx": +0.3, "vy": +0.8, "vz": -0.6, "dur": 2.5},
    "lower_left":    {"vx": +0.4, "vy": -0.8, "vz": +0.5, "dur": 2.5},
    "lower_right":   {"vx": +0.4, "vy": +0.8, "vz": +0.5, "dur": 2.5},
    # Emergency: drone is fully boxed in. Pure climb, no horizontal.
    "emergency_climb": {"vx": 0.0, "vy": 0.0, "vz": -1.0, "dur": 3.0},
}

VALID_DIRECTIONS = list(ACTION_SPACE.keys())

# Maximum speed for any axis (m/s)
MAX_SPEED = 0.6

# Center-distance gating for forward velocity.
# At very close range, forward component is reduced toward zero.
def _scale_forward_by_proximity(center_distance):
    """Adaptive vx scale factor based on how close the front obstacle is."""
    if center_distance is None:
        return 0.5
    if center_distance >= 1.5:
        return 1.0      # full forward — safe
    if center_distance >= 1.0:
        return 0.6      # gentle forward
    if center_distance >= 0.6:
        return 0.3      # very gentle
    return 0.0          # too close — pure lateral/vertical, no forward


def _direction_to_velocity(direction, center_distance, altitude,
                           distances_9seg=None):
    """Translate a direction action to (vx, vy, vz, duration)."""
    if direction not in ACTION_SPACE:
        return None
    a = ACTION_SPACE[direction]

    fwd_scale = _scale_forward_by_proximity(center_distance)
    vx = a["vx"] * MAX_SPEED * fwd_scale
    vy = a["vy"] * MAX_SPEED
    vz = a["vz"] * MAX_SPEED
    dur = a["dur"]

    # ── Backward: hard cap at vx=-0.4, dur=1.5s (Option 2) ──────────────
    # Just enough clearance-angle retreat, not a full physical retreat.
    if direction == "backward":
        vx = -0.4
        dur = 1.5

    # ── Lateral-priority: suppress climb when lateral path is feasible ───
    # If the chosen action has a vz component (up/diagonal), but the left
    # or right cell already has enough clearance, zero out the vertical so
    # the drone sidesteps instead of climbing unnecessarily.
    LATERAL_CLEARANCE = 1.5  # meters — "good enough to go sideways"
    if vz < 0 and direction not in ("up", "emergency_climb"):
        dists = distances_9seg or {}
        left_d  = dists.get("left")  or 0.0
        right_d = dists.get("right") or 0.0
        if left_d >= LATERAL_CLEARANCE or right_d >= LATERAL_CLEARANCE:
            vz = 0.0   # lateral is clear — stay at current altitude

    # ── Altitude safety ───────────────────────────────────────────────────
    if altitude is not None:
        if altitude < 2.0 and vz > 0:
            vz = 0.0    # don't go down close to ground
        if altitude > 15.0 and vz < 0:
            vz = 0.0    # don't go up too high

    return (round(vx, 2), round(vy, 2), round(vz, 2), dur)


# ── GPT advisor ────────────────────────────────────────────────

class GPTAdvisor:
    """Sends camera frame + depth map + telemetry to GPT-4o Vision."""

    MODEL = "gpt-4o"
    IMAGE_DETAIL = "low"
    MAX_IMAGE_WIDTH = 512

    SYSTEM_PROMPT = (
        "You are an autonomous drone obstacle avoidance system with vision.\n\n"
        "INPUTS:\n"
        "1. Camera image — drone's forward view.\n"
        "2. Colorized depth map (Inferno colormap) with 3x3 grid + per-cell "
        "distance labels in meters. Brighter (yellow/white) = CLOSER, "
        "darker (purple/black) = FARTHER.\n"
        "3. Per-segment metric distances for all 9 grid cells.\n"
        "4. Drone telemetry (altitude, mode, attitude).\n\n"

        "STATE: The drone is already STOPPED and hovering safely. Choose ONE "
        "direction to move so the drone goes around or over the obstacle.\n\n"

        "═══════════════════════════════════════════════════\n"
        "ACTION SPACE — pick exactly ONE direction:\n"
        "═══════════════════════════════════════════════════\n"
        "  'forward'       — only if center cell is clear (>= 1.5m)\n"
        "  'backward'      — short retreat (~0.6m) to gain clearance angle;\n"
        "                    use ONLY when obstacle is < 0.6m AND no lateral\n"
        "                    path is available. Duration is capped at 1.5s.\n"
        "  'left'          — slide left when LEFT cell is clearest\n"
        "  'right'         — slide right when RIGHT cell is clearest\n"
        "  'up'            — climb when UP cell is clearest AND lateral cells\n"
        "                    are both blocked (< 1.5m)\n"
        "  'down'          — descend when DOWN cell is clear AND alt > 2m\n"
        "  'upper_left'    — diagonal climb left (best when UL cell is clearest)\n"
        "  'upper_right'   — diagonal climb right (best when UR cell is clearest)\n"
        "  'lower_left'    — diagonal descend left (alt > 2m)\n"
        "  'lower_right'   — diagonal descend right (alt > 2m)\n"
        "  'emergency_climb' — pure straight up, only when ALL 9 cells < 1.0m\n"
        "\nLATERAL PRIORITY RULE: If LEFT or RIGHT cell >= 1.5m, prefer 'left'\n"
        "or 'right' over any action that climbs. Do NOT pick 'up', 'upper_left',\n"
        "or 'upper_right' when a lateral path is already clear. The system will\n"
        "automatically suppress the climb component of diagonal actions when\n"
        "lateral clearance exists — but picking a pure lateral action is cleaner.\n\n"
        "There is NO uncontrolled backward action. 'backward' is capped at 0.6m\n"
        "and should only be used as a last resort when all lateral paths are\n"
        "blocked AND the obstacle is critically close (< 0.6m).\n\n"

        "DECISION ALGORITHM:\n"
        "1. Identify the SINGLE cell with the largest distance in the grid.\n"
        "2. CHECK LATERAL FIRST: if LEFT or RIGHT >= 1.5m, pick 'left'/'right' — do NOT climb.\n"
        "3. If the clearest cell is in the upper row AND both lateral < 1.5m, pick up / upper_left / upper_right.\n"
        "4. If lower row AND altitude > 2m, pick down / lower_left / lower_right.\n"
        "5. If ALL 9 cells < 1.0m AND center < 0.6m AND no lateral path, pick 'backward'.\n"
        "6. If ALL 9 cells < 1.0m and even backward is risky, pick 'emergency_climb'.\n"
        "7. Use the camera image to break ties — what looks like real open space?\n\n"

        "EXAMPLES:\n"
        "  C=0.4m, L=2.0m, R=0.6m  -> 'left' (LEFT clearly clearest, no need to climb)\n"
        "  C=0.4m, L=0.5m, R=0.6m, UR=4.5m -> 'upper_right' (lateral blocked, UR is best)\n"
        "  C=0.4m, L=1.8m, UR=4.5m -> 'left' (lateral feasible — prefer left over climbing)\n"
        "  C=0.4m, all sides<1m, UP=2.5m -> 'up' (only UP is clear)\n"
        "  C=0.4m, all 9 cells<0.6m -> 'backward' (critically close, gain angle)\n"
        "  All 9 cells around 0.5m, backward unsafe -> 'emergency_climb'\n\n"

        "OUTPUT — JSON only, no markdown:\n"
        '{"direction": "<one of the 10 actions>", "reasoning": "brief why"}\n\n'

        "Reasoning under 25 words. Reference the specific cell distances + "
        "what you SEE in the camera that drove your choice."
    )

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    # ── Image encoding ──────────────────────────────────────

    def _frame_to_base64(self, rgb_frame):
        from PIL import Image
        pil = Image.fromarray(rgb_frame)
        if pil.width > self.MAX_IMAGE_WIDTH:
            ratio = self.MAX_IMAGE_WIDTH / pil.width
            new_size = (self.MAX_IMAGE_WIDTH, int(pil.height * ratio))
            pil = pil.resize(new_size, Image.LANCZOS)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Prompt building ─────────────────────────────────────

    def _build_user_text(self, telemetry, distances_9seg):
        alt = telemetry.get("altitude", 0.0)
        heading = telemetry.get("heading", 0)
        mode = telemetry.get("mode", "UNKNOWN")

        def fmt(name):
            d = distances_9seg.get(name)
            return f"{d:.2f}m" if d is not None else "n/a"

        # Identify the clearest cell — directive hint for the model
        valid = {k: v for k, v in distances_9seg.items() if v is not None}
        if valid:
            best_cell = max(valid, key=valid.get)
            best_dist = valid[best_cell]
            best_hint = f"BEST CELL: {best_cell.upper()} at {best_dist:.2f}m"
        else:
            best_hint = "BEST CELL: unknown"

        # All cells with >= 1.5m clearance
        clear_cells = [(k, v) for k, v in valid.items() if v >= 1.5]
        clear_cells.sort(key=lambda x: -x[1])
        if clear_cells:
            clear_str = ", ".join(f"{k.upper()} ({v:.2f}m)" for k, v in clear_cells)
        else:
            clear_str = "none (all cells < 1.5m)"

        return (
            f"Obstacle ahead. Drone is HOVERING. Pick ONE direction.\n\n"
            f"Depth grid (meters, 3x3, drone forward view):\n"
            f"  +-----------------+-----------------+-----------------+\n"
            f"  | UL: {fmt('upper_left'):>6s}  | UP: {fmt('up'):>6s}  | UR: {fmt('upper_right'):>6s}  |\n"
            f"  +-----------------+-----------------+-----------------+\n"
            f"  | L:  {fmt('left'):>6s}  | C:  {fmt('center'):>6s}  | R:  {fmt('right'):>6s}  |\n"
            f"  +-----------------+-----------------+-----------------+\n"
            f"  | LL: {fmt('lower_left'):>6s}  | DN: {fmt('down'):>6s}  | LR: {fmt('lower_right'):>6s}  |\n"
            f"  +-----------------+-----------------+-----------------+\n\n"
            f"{best_hint}\n"
            f"Cells with clearance >= 1.5m: {clear_str}\n\n"
            f"Telemetry:  alt={alt:.1f}m  hdg={heading} deg  mode={mode}\n\n"
            f"Pick the direction whose target cell has the most space. "
            f"Verify visually using the camera image."
        )

    # ── Main API ────────────────────────────────────────────

    def ask(self, telemetry, frame, depth_colored, distances_9seg, threshold=1.0):
        """Send vision + telemetry to GPT-4o, return velocity command.

        Returns:
            dict {"vx", "vy", "vz", "duration", "reasoning", "direction"}
            or None on failure.
        """
        client = self._get_client()

        center_d = distances_9seg.get("center")
        alt = telemetry.get("altitude", 0)
        center_str = f"{center_d:.2f}m" if center_d is not None else "n/a"
        print(
            f"[GPT] Sending vision request -> {self.MODEL} "
            f"(center={center_str} alt={alt:.1f}m)"
        )

        user_text = self._build_user_text(telemetry, distances_9seg)
        frame_b64 = self._frame_to_base64(frame)
        depth_b64 = self._frame_to_base64(depth_colored)

        response = client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}",
                                       "detail": self.IMAGE_DETAIL}},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{depth_b64}",
                                       "detail": self.IMAGE_DETAIL}},
                    ],
                },
            ],
            max_tokens=200,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        msg = response.choices[0].message
        raw = msg.content
        refusal = getattr(msg, "refusal", None)

        if refusal:
            print(f"[GPT] Refused: {refusal}")
            return None
        if not raw or not raw.strip():
            print("[GPT] Empty response")
            return None

        return _parse_and_translate(raw, center_d, alt, distances_9seg, source="GPT")


# ── Shared parser + direction → velocity translator ────────

def _parse_and_translate(raw, center_distance, altitude, distances_9seg=None, source="VLM"):
    """Parse JSON action, translate to velocity command. Returns dict or None."""
    raw = raw.strip()
    print(f"[{source}] Raw response: {raw[:200]}")

    text = raw
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()

    if not text.startswith("{"):
        print(f"[{source}] Non-JSON response: {text[:80]}")
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[{source}] JSON parse error: {e}")
        return None

    direction = str(data.get("direction", "")).strip().lower()
    reasoning = str(data.get("reasoning", ""))

    if direction not in ACTION_SPACE:
        print(f"[{source}] Invalid direction '{direction}' — falling back to emergency_climb")
        direction = "emergency_climb"
        reasoning = f"(invalid VLM direction; fallback) {reasoning}"

    vel = _direction_to_velocity(direction, center_distance, altitude, distances_9seg)
    if vel is None:
        print(f"[{source}] Translation failed for '{direction}'")
        return None
    vx, vy, vz, dur = vel

    cmd = {
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "duration": dur,
        "reasoning": reasoning,
        "direction": direction.upper().replace("_", "-"),
    }

    print(
        f"[{source}] >> {cmd['direction']}  "
        f"vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f}  "
        f"dur={dur:.1f}s | {reasoning}"
    )
    return cmd