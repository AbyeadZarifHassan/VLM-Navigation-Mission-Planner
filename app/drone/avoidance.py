"""Obstacle avoidance state machine with logic-based decisions."""

import enum
import time


class AvoidanceState(enum.Enum):
    IDLE = "IDLE"
    OBSTACLE_DETECTED = "OBSTACLE_DETECTED"
    HOVERING = "HOVERING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    CLEARANCE = "CLEARANCE"  # fly forward past the obstacle before resuming


class AvoidanceAction(enum.Enum):
    NONE = "NONE"
    BACKWARD = "BACKWARD"
    BACKWARD_UP = "BACKWARD_UP"
    BACKWARD_LEFT = "BACKWARD_LEFT"
    BACKWARD_RIGHT = "BACKWARD_RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class ObstacleAvoidance:
    """State machine for obstacle avoidance with logic-based decisions."""

    DISTANCE_THRESHOLD = 1.0  # meters
    HOVER_STABILIZE_TIME = 1.5  # seconds — min hover before executing maneuver
    MANEUVER_DURATION = 2.0  # seconds
    MANEUVER_SPEED = 0.5  # m/s
    MIN_ALTITUDE_FOR_DOWN = 2.0  # meters — don't go down if below this
    MAX_ALTITUDE_FOR_UP = 15.0  # meters — don't go up if above this
    CRITICAL_DISTANCE = 0.4  # meters — very close, strong backward
    CLEARANCE_DURATION = 4.0  # seconds — fly forward past obstacle before resuming
    CLEARANCE_SPEED = 0.5  # m/s — forward speed during clearance

    def __init__(self):
        self.state = AvoidanceState.IDLE
        self.current_action = AvoidanceAction.NONE
        self._state_enter_time = 0.0
        self._current_altitude = 0.0
        self._last_distance = None
        self._attempt_count = 0  # how many consecutive avoidance attempts
        self._last_avoidance_vy = 0.0  # lateral dir from last maneuver (for clearance)

        # Velocity/duration set by decide_action_smart() or GPT
        self._gpt_velocity = None  # (vx, vy, vz) or None
        self._gpt_duration = None  # seconds or None

    def update_altitude(self, altitude):
        """Update the current altitude for decision making."""
        self._current_altitude = altitude

    def decide_action_smart(self, distance, distances=None):
        """Choose avoidance maneuver based on distance, altitude, spatial clearance.

        Args:
            distance: forward center distance in meters
            distances: dict with 'left', 'center', 'right' distances (meters)

        Returns (action, vx, vy, vz, duration) tuple.
        """
        alt = self._current_altitude
        speed = self.MANEUVER_SPEED
        dist_left = (distances or {}).get("left") or 20.0
        dist_right = (distances or {}).get("right") or 20.0

        # Clearance margin — consider a side "clear" if > threshold
        CLEAR_MARGIN = self.DISTANCE_THRESHOLD * 1.5  # 1.5m

        # ── Critical proximity (< 0.4m) — emergency backward ──
        if distance is not None and distance < self.CRITICAL_DISTANCE:
            vx = -min(speed * 1.5, 1.0)
            if alt < self.MIN_ALTITUDE_FOR_DOWN:
                return (AvoidanceAction.BACKWARD_UP, vx, 0, -speed * 0.5, 3.0)
            # Even in emergency, check if a side is clear for combo move
            if dist_left > CLEAR_MARGIN and dist_left > dist_right:
                return (AvoidanceAction.BACKWARD_LEFT, vx, -speed * 0.5, 0, 3.0)
            elif dist_right > CLEAR_MARGIN:
                return (AvoidanceAction.BACKWARD_RIGHT, vx, speed * 0.5, 0, 3.0)
            return (AvoidanceAction.BACKWARD, vx, 0, 0, 3.0)

        # ── Multiple failed attempts — escalate ──
        if self._attempt_count >= 3:
            if alt < self.MAX_ALTITUDE_FOR_UP:
                self._attempt_count = 0
                return (AvoidanceAction.UP, 0, 0, -speed, 3.0)
            else:
                self._attempt_count = 0
                return (AvoidanceAction.BACKWARD, -speed, 0, 0, 3.0)

        # ── Check lateral clearance — prefer moving to the clearer side ──
        left_clear = dist_left > CLEAR_MARGIN
        right_clear = dist_right > CLEAR_MARGIN
        both_blocked = not left_clear and not right_clear

        # If one side has significantly more space, go there
        if not both_blocked:
            if left_clear and right_clear:
                # Both clear — pick the one with more space
                if dist_left > dist_right * 1.3:
                    vy = -speed * 0.8
                    return (AvoidanceAction.BACKWARD_LEFT, -speed * 0.3, vy, 0, 2.5)
                elif dist_right > dist_left * 1.3:
                    vy = speed * 0.8
                    return (AvoidanceAction.BACKWARD_RIGHT, -speed * 0.3, vy, 0, 2.5)
                else:
                    # Similar clearance — slight backward + right (default)
                    return (AvoidanceAction.BACKWARD_RIGHT, -speed * 0.3, speed * 0.6, 0, 2.5)
            elif left_clear:
                vy = -speed * 0.8
                return (AvoidanceAction.LEFT, -speed * 0.2, vy, 0, 2.5)
            elif right_clear:
                vy = speed * 0.8
                return (AvoidanceAction.RIGHT, -speed * 0.2, vy, 0, 2.5)

        # ── Both sides blocked — backward or up ──
        if alt < self.MIN_ALTITUDE_FOR_DOWN:
            return (AvoidanceAction.BACKWARD_UP, -speed, 0, -speed * 0.5, 2.5)

        if alt > self.MAX_ALTITUDE_FOR_UP:
            return (AvoidanceAction.BACKWARD, -speed, 0, 0, 2.0)

        # Scale backward speed by proximity
        if distance is not None:
            speed_factor = max(0.5, min(1.0, 1.0 - (distance / self.DISTANCE_THRESHOLD)))
            vx = -speed * (0.5 + speed_factor)
            vx = max(-1.0, vx)
        else:
            vx = -speed

        return (AvoidanceAction.BACKWARD, vx, 0, 0, 2.0)

    def get_velocity_for_action(self, action):
        """Return (vx, vy, vz) for the given avoidance action."""
        speed = self.MANEUVER_SPEED
        if action == AvoidanceAction.BACKWARD:
            return (-speed, 0, 0)
        elif action == AvoidanceAction.BACKWARD_UP:
            return (-speed, 0, -speed * 0.5)
        elif action == AvoidanceAction.BACKWARD_LEFT:
            return (-speed * 0.3, -speed, 0)
        elif action == AvoidanceAction.BACKWARD_RIGHT:
            return (-speed * 0.3, speed, 0)
        elif action == AvoidanceAction.UP:
            return (0, 0, -speed)
        elif action == AvoidanceAction.DOWN:
            return (0, 0, speed)
        elif action == AvoidanceAction.LEFT:
            return (0, -speed, 0)
        elif action == AvoidanceAction.RIGHT:
            return (0, speed, 0)
        return (0, 0, 0)

    def set_gpt_command(self, vx, vy, vz, duration):
        """Set velocity override for the next maneuver (from GPT or auto logic)."""
        self._gpt_velocity = (vx, vy, vz)
        self._gpt_duration = duration

    def _clear_gpt_command(self):
        self._gpt_velocity = None
        self._gpt_duration = None

    @property
    def waiting_for_gpt(self):
        """True while in HOVERING and no command has arrived yet."""
        return self.state == AvoidanceState.HOVERING and self._gpt_velocity is None

    def update(self, forward_distance):
        """Update the state machine. Returns (state, action, velocity_cmd).

        Args:
            forward_distance: estimated distance to forward object in meters.
                             None if no measurement available.

        Returns:
            tuple: (state, action, (vx, vy, vz))
        """
        now = time.time()
        self._last_distance = forward_distance

        if self.state == AvoidanceState.IDLE:
            if forward_distance is not None and forward_distance < self.DISTANCE_THRESHOLD:
                self.state = AvoidanceState.OBSTACLE_DETECTED
                self._state_enter_time = now
                self.current_action = AvoidanceAction.NONE
                self._clear_gpt_command()
            return (self.state, self.current_action, (0, 0, 0))

        elif self.state == AvoidanceState.OBSTACLE_DETECTED:
            # Stop all movement
            self.state = AvoidanceState.HOVERING
            self._state_enter_time = now
            return (self.state, AvoidanceAction.NONE, (0, 0, 0))

        elif self.state == AvoidanceState.HOVERING:
            elapsed = now - self._state_enter_time

            # ENFORCE minimum hover time — drone must stop and stabilize
            # BEFORE any maneuver, regardless of how fast command arrives
            if elapsed < self.HOVER_STABILIZE_TIME:
                return (self.state, self.current_action, (0, 0, 0))

            # Stabilization done — execute command if available
            if self._gpt_velocity is not None:
                self.state = AvoidanceState.EXECUTING
                self._state_enter_time = now
                vel = self._gpt_velocity
                return (self.state, self.current_action, vel)

            # Fallback: if GPT hasn't responded after 15s of hover, use smart auto
            if elapsed >= 15.0:
                action, vx, vy, vz, dur = self.decide_action_smart(forward_distance, {})
                self.current_action = action
                self.set_gpt_command(vx, vy, vz, dur)
                self.state = AvoidanceState.EXECUTING
                self._state_enter_time = now
                return (self.state, self.current_action, (vx, vy, vz))

            # Still waiting for command — keep hovering
            return (self.state, self.current_action, (0, 0, 0))

        elif self.state == AvoidanceState.EXECUTING:
            elapsed = now - self._state_enter_time

            # Use stored velocity command
            if self._gpt_velocity is not None:
                vel = self._gpt_velocity
                duration = self._gpt_duration or self.MANEUVER_DURATION
            else:
                vel = self.get_velocity_for_action(self.current_action)
                duration = self.MANEUVER_DURATION

            # Remember lateral direction for clearance phase
            self._last_avoidance_vy = vel[1]

            if elapsed >= duration:
                self.state = AvoidanceState.COMPLETED
                self._state_enter_time = now
                self._clear_gpt_command()
                return (self.state, self.current_action, (0, 0, 0))
            return (self.state, self.current_action, vel)

        elif self.state == AvoidanceState.COMPLETED:
            if forward_distance is None or forward_distance >= self.DISTANCE_THRESHOLD:
                # Obstacle cleared — enter CLEARANCE to fly past it
                self.state = AvoidanceState.CLEARANCE
                self._state_enter_time = now
                self._attempt_count = 0
            else:
                # Still blocked — increment attempt count and try again
                self._attempt_count += 1
                self.state = AvoidanceState.HOVERING
                self._state_enter_time = now
            return (self.state, self.current_action, (0, 0, 0))

        elif self.state == AvoidanceState.CLEARANCE:
            # Fly FORWARD + slight lateral (same direction as avoidance)
            # to pass the obstacle before resuming AUTO
            elapsed = now - self._state_enter_time

            if elapsed >= self.CLEARANCE_DURATION:
                # Clearance complete — safe to resume
                self.state = AvoidanceState.IDLE
                self.current_action = AvoidanceAction.NONE
                return (self.state, self.current_action, (0, 0, 0))

            # If obstacle reappears during clearance, abort and re-detect
            if forward_distance is not None and forward_distance < self.CRITICAL_DISTANCE:
                self.state = AvoidanceState.OBSTACLE_DETECTED
                self._state_enter_time = now
                self._clear_gpt_command()
                return (self.state, self.current_action, (0, 0, 0))

            # Forward + slight lateral in same direction we avoided
            vy_clearance = self._last_avoidance_vy * 0.3  # gentle lateral drift
            vel = (self.CLEARANCE_SPEED, vy_clearance, 0)
            return (self.state, self.current_action, vel)

        return (self.state, self.current_action, (0, 0, 0))

    def reset(self):
        """Reset state machine to IDLE."""
        self.state = AvoidanceState.IDLE
        self.current_action = AvoidanceAction.NONE
        self._attempt_count = 0
        self._clear_gpt_command()
