"""Interactive Leaflet map for mission planning.

QWebEngineView + QWebChannel bridge for bidirectional JS↔Python communication.
"""

import os
import json
from PyQt5.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel


_HTML_PATH = os.path.join(os.path.dirname(__file__), "map_assets", "map.html")


class MapBridge(QObject):
    """Python-side bridge: JS calls these via QWebChannel."""

    waypoint_added_from_map = pyqtSignal(float, float)  # lat, lon
    waypoint_position_changed = pyqtSignal(int, float, float)  # seq, lat, lon

    @pyqtSlot(float, float)
    def add_waypoint_from_map(self, lat, lon):
        self.waypoint_added_from_map.emit(lat, lon)

    @pyqtSlot(int, float, float)
    def update_waypoint_position(self, seq, lat, lon):
        self.waypoint_position_changed.emit(seq, lat, lon)


class MapView(QWebEngineView):
    """Interactive map showing waypoints + live drone position."""

    waypoint_added = pyqtSignal(float, float)  # user clicked map
    waypoint_moved = pyqtSignal(int, float, float)  # seq, new_lat, new_lon

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ready = False
        self._pending_calls = []

        self.bridge = MapBridge()
        self.bridge.waypoint_added_from_map.connect(self.waypoint_added.emit)
        self.bridge.waypoint_position_changed.connect(self.waypoint_moved.emit)

        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.page().setWebChannel(self.channel)

        self.loadFinished.connect(self._on_load_finished)
        self.load(QUrl.fromLocalFile(_HTML_PATH))

    def _on_load_finished(self, ok):
        self._ready = ok
        # Flush any queued calls
        for call in self._pending_calls:
            self.page().runJavaScript(call)
        self._pending_calls = []

    def _run_js(self, code):
        if self._ready:
            self.page().runJavaScript(code)
        else:
            self._pending_calls.append(code)

    # ── Public API ────────────────────────────────────────

    def refresh_waypoints(self, waypoints):
        """Update all waypoint markers on the map.

        Args:
            waypoints: list of Waypoint objects with .lat, .lon, .alt, .command
        """
        data = [[wp.lat, wp.lon, wp.alt, wp.command] for wp in waypoints]
        self._run_js(f"refreshWaypoints({json.dumps(data)});")

    def update_drone_position(self, lat, lon):
        """Move the drone marker."""
        self._run_js(f"updateDronePosition({lat}, {lon});")

    def center_on(self, lat, lon, zoom=18):
        """Pan/zoom map to a location."""
        self._run_js(f"centerOn({lat}, {lon}, {zoom});")
