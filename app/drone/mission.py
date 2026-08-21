"""Mission waypoint data classes and helpers."""

import math
from dataclasses import dataclass, field
from pymavlink import mavutil


# Supported waypoint commands — basic set
COMMAND_NAMES = [
    "NAV_TAKEOFF",
    "NAV_WAYPOINT",
    "NAV_LOITER_TIME",
    "NAV_LAND",
    "NAV_RETURN_TO_LAUNCH",
]

COMMAND_IDS = {
    "NAV_TAKEOFF": mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    "NAV_WAYPOINT": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
    "NAV_LOITER_TIME": mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME,
    "NAV_LAND": mavutil.mavlink.MAV_CMD_NAV_LAND,
    "NAV_RETURN_TO_LAUNCH": mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
}

# Reverse lookup: numeric ID -> name
COMMAND_ID_TO_NAME = {v: k for k, v in COMMAND_IDS.items()}


@dataclass
class Waypoint:
    """A single mission waypoint."""
    seq: int = 0
    command: str = "NAV_WAYPOINT"
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 10.0
    p1: float = 0.0  # hold time (sec) / pitch angle / loiter time
    p2: float = 0.0  # acceptance radius (m)
    p3: float = 0.0  # pass radius (m)
    p4: float = 0.0  # yaw angle (deg)

    @property
    def command_id(self):
        return COMMAND_IDS.get(self.command, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT)

    def to_dict(self):
        return {
            "seq": self.seq,
            "command": self.command,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
            "p4": self.p4,
        }


def offset_to_latlon(lat_home, lon_home, north_m, east_m):
    """Convert (north, east) meters offset to (lat, lon) coordinates.

    Uses flat-earth approximation — accurate enough for missions up to ~1km.
    """
    # 1 degree latitude ≈ 111111 meters
    lat_new = lat_home + (north_m / 111111.0)
    # 1 degree longitude depends on latitude (shrinks toward poles)
    lon_new = lon_home + (east_m / (111111.0 * math.cos(math.radians(lat_home))))
    return (lat_new, lon_new)


def parse_waypoint_file(path):
    """Parse a Mission Planner .waypoint file.

    File format (QGC WPL 110):
        QGC WPL 110
        <seq> <current> <frame> <command> <p1> <p2> <p3> <p4> <lat> <lon> <alt> <autocontinue>

    Returns list[Waypoint].
    """
    waypoints = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines or not lines[0].startswith("QGC WPL"):
        raise ValueError("Not a valid Mission Planner .waypoint file (missing QGC WPL header)")

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            parts = line.split()
        if len(parts) < 12:
            continue
        try:
            seq = int(parts[0])
            # current = int(parts[1])
            # frame = int(parts[2])
            cmd_id = int(parts[3])
            p1 = float(parts[4])
            p2 = float(parts[5])
            p3 = float(parts[6])
            p4 = float(parts[7])
            lat = float(parts[8])
            lon = float(parts[9])
            alt = float(parts[10])
            # autocontinue = int(parts[11])

            cmd_name = COMMAND_ID_TO_NAME.get(cmd_id, "NAV_WAYPOINT")
            waypoints.append(Waypoint(
                seq=seq, command=cmd_name,
                lat=lat, lon=lon, alt=alt,
                p1=p1, p2=p2, p3=p3, p4=p4,
            ))
        except (ValueError, IndexError):
            continue

    return waypoints


def write_waypoint_file(path, waypoints):
    """Write waypoints to a Mission Planner .waypoint file."""
    with open(path, "w") as f:
        f.write("QGC WPL 110\n")
        for i, wp in enumerate(waypoints):
            cmd_id = COMMAND_IDS.get(wp.command, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT)
            current = 1 if i == 0 else 0
            # frame 3 = MAV_FRAME_GLOBAL_RELATIVE_ALT (altitude relative to home)
            frame = 3
            autocontinue = 1
            f.write(
                f"{i}\t{current}\t{frame}\t{cmd_id}\t"
                f"{wp.p1:.8f}\t{wp.p2:.8f}\t{wp.p3:.8f}\t{wp.p4:.8f}\t"
                f"{wp.lat:.8f}\t{wp.lon:.8f}\t{wp.alt:.6f}\t{autocontinue}\n"
            )
