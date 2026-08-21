"""Mission Planner tab — waypoint table + mission control."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QFileDialog,
    QDialog, QDoubleSpinBox, QFormLayout, QDialogButtonBox, QCheckBox,
    QTextEdit, QMessageBox, QAbstractItemView, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal

from app.drone.mission import (
    Waypoint, COMMAND_NAMES, offset_to_latlon,
    parse_waypoint_file, write_waypoint_file,
)
from app.widgets.map_view import MapView


class OffsetDialog(QDialog):
    """Dialog to create a waypoint by entering N/E/Alt offsets from current position."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Offset Waypoint")
        self.setStyleSheet("background: #1e1e1e; color: #eee;")

        layout = QFormLayout(self)

        self.north_spin = QDoubleSpinBox()
        self.north_spin.setRange(-10000, 10000)
        self.north_spin.setDecimals(1)
        self.north_spin.setSuffix(" m")
        self.north_spin.setValue(10.0)

        self.east_spin = QDoubleSpinBox()
        self.east_spin.setRange(-10000, 10000)
        self.east_spin.setDecimals(1)
        self.east_spin.setSuffix(" m")
        self.east_spin.setValue(0.0)

        self.alt_spin = QDoubleSpinBox()
        self.alt_spin.setRange(0, 500)
        self.alt_spin.setDecimals(1)
        self.alt_spin.setSuffix(" m")
        self.alt_spin.setValue(10.0)

        for w in (self.north_spin, self.east_spin, self.alt_spin):
            w.setStyleSheet("background: #2a2a2a; color: #eee; padding: 4px;")

        layout.addRow("North:", self.north_spin)
        layout.addRow("East:", self.east_spin)
        layout.addRow("Altitude:", self.alt_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def values(self):
        return (self.north_spin.value(), self.east_spin.value(), self.alt_spin.value())


class MissionPlannerWidget(QWidget):
    """Tab for planning, uploading, and monitoring waypoint missions."""

    # Signals to main window
    upload_requested = pyqtSignal(list)  # list[Waypoint]
    download_requested = pyqtSignal()
    start_requested = pyqtSignal()
    clear_drone_requested = pyqtSignal()
    avoidance_during_mission_toggled = pyqtSignal(bool)

    COLUMNS = ["#", "Cmd", "Lat", "Lon", "Alt (m)", "P1", "P2"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_lat = 0.0
        self._current_lon = 0.0
        self._current_alt = 0.0
        self._default_alt = 10.0  # default altitude for map-click waypoints
        self._map_centered = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Vertical)
        outer.addWidget(splitter, stretch=1)

        # ── Map (top half) ────────────────────────────────
        map_group = QGroupBox("Map — click to add waypoint, drag markers to move")
        map_group.setStyleSheet(self._group_style())
        mg_layout = QVBoxLayout(map_group)
        mg_layout.setContentsMargins(4, 4, 4, 4)

        # Default altitude input for map clicks
        alt_row = QHBoxLayout()
        alt_row.addWidget(QLabel("Default alt for new WP:"))
        self.default_alt_spin = QDoubleSpinBox()
        self.default_alt_spin.setRange(0, 500)
        self.default_alt_spin.setDecimals(1)
        self.default_alt_spin.setSuffix(" m")
        self.default_alt_spin.setValue(10.0)
        self.default_alt_spin.setStyleSheet("background: #2a2a2a; color: #eee; padding: 4px;")
        self.default_alt_spin.valueChanged.connect(lambda v: setattr(self, "_default_alt", v))
        alt_row.addWidget(self.default_alt_spin)
        alt_row.addStretch()
        self.btn_center_drone = QPushButton("Center on Drone")
        self.btn_center_drone.setStyleSheet(self._btn_style("#444"))
        self.btn_center_drone.clicked.connect(self._center_on_drone)
        alt_row.addWidget(self.btn_center_drone)
        mg_layout.addLayout(alt_row)

        self.map_view = MapView()
        self.map_view.setMinimumHeight(150)
        self.map_view.waypoint_added.connect(self._on_map_click_add_waypoint)
        self.map_view.waypoint_moved.connect(self._on_map_marker_moved)
        mg_layout.addWidget(self.map_view)

        splitter.addWidget(map_group)

        # ── Bottom: table + controls ──────────────────────
        bottom_widget = QWidget()
        layout = QVBoxLayout(bottom_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ── Waypoint table ────────────────────────────────
        table_group = QGroupBox("Waypoints")
        table_group.setStyleSheet(self._group_style())
        tg_layout = QVBoxLayout(table_group)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { background: #1a1a1a; color: #eee; gridline-color: #333; "
            "alternate-background-color: #222; font-family: Consolas; font-size: 11px; }"
            "QHeaderView::section { background: #2a2a2a; color: #0af; padding: 4px; "
            "border: 1px solid #333; font-weight: bold; }"
        )
        self.table.itemChanged.connect(lambda _: self._refresh_map_markers())
        tg_layout.addWidget(self.table)
        layout.addWidget(table_group, stretch=3)

        # ── Add buttons (2 rows for responsiveness) ─────
        add_row1 = QHBoxLayout()
        self.btn_manual = QPushButton("+ Manual")
        self.btn_current = QPushButton("+ Current Pos")
        self.btn_offset = QPushButton("+ Offset...")
        self.btn_rth = QPushButton("+ Return Home")
        self.btn_rth.setStyleSheet(self._btn_style("#FF9800"))
        for b in (self.btn_manual, self.btn_current, self.btn_offset, self.btn_rth):
            if b is not self.btn_rth:
                b.setStyleSheet(self._btn_style("#444"))
            add_row1.addWidget(b)
        layout.addLayout(add_row1)

        add_row2 = QHBoxLayout()
        self.btn_import = QPushButton("Import .waypoint")
        self.btn_export = QPushButton("Export .waypoint")
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_clear = QPushButton("Clear All")
        for b in (self.btn_import, self.btn_export, self.btn_delete, self.btn_clear):
            b.setStyleSheet(self._btn_style("#444"))
            add_row2.addWidget(b)
        layout.addLayout(add_row2)

        self.btn_manual.clicked.connect(self._add_manual)
        self.btn_current.clicked.connect(self._add_current)
        self.btn_offset.clicked.connect(self._add_offset)
        self.btn_rth.clicked.connect(self._add_return_home)
        self.btn_import.clicked.connect(self._import_file)
        self.btn_export.clicked.connect(self._export_file)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_clear.clicked.connect(self._clear_all)

        # ── Mission control ───────────────────────────────
        ctrl_row = QHBoxLayout()
        self.btn_upload = QPushButton("Upload to Drone")
        self.btn_upload.setStyleSheet(self._btn_style("#2196F3"))
        self.btn_download = QPushButton("Download from Drone")
        self.btn_download.setStyleSheet(self._btn_style("#666"))
        self.btn_start = QPushButton("Start Mission")
        self.btn_start.setStyleSheet(self._btn_style("#4CAF50"))
        self.btn_clear_drone = QPushButton("Clear Drone Memory")
        self.btn_clear_drone.setStyleSheet(self._btn_style("#f44336"))

        for b in (self.btn_upload, self.btn_download, self.btn_start, self.btn_clear_drone):
            ctrl_row.addWidget(b)
        layout.addLayout(ctrl_row)

        self.btn_upload.clicked.connect(self._on_upload_clicked)
        self.btn_download.clicked.connect(self.download_requested.emit)
        self.btn_start.clicked.connect(self.start_requested.emit)
        self.btn_clear_drone.clicked.connect(self.clear_drone_requested.emit)

        # ── Options ───────────────────────────────────────
        opt_row = QHBoxLayout()
        self.chk_avoidance_mission = QCheckBox("Enable avoidance during mission")
        self.chk_avoidance_mission.setChecked(True)
        self.chk_avoidance_mission.setStyleSheet("color: #eee;")
        self.chk_avoidance_mission.toggled.connect(
            lambda checked: self.avoidance_during_mission_toggled.emit(checked)
        )
        opt_row.addWidget(self.chk_avoidance_mission)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        # ── Status display ────────────────────────────────
        status_group = QGroupBox("Mission Status")
        status_group.setStyleSheet(self._group_style())
        sg_layout = QVBoxLayout(status_group)

        self.progress_label = QLabel("No mission active")
        self.progress_label.setStyleSheet(
            "color: #0af; font-family: Consolas; font-size: 12px; "
            "background: #111; padding: 6px; border: 1px solid #333;"
        )
        sg_layout.addWidget(self.progress_label)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(80)
        self.status_log.setMinimumHeight(40)
        self.status_log.setStyleSheet(
            "background: #111; color: #ccc; font-family: Consolas; "
            "font-size: 11px; border: 1px solid #333;"
        )
        sg_layout.addWidget(self.status_log)

        layout.addWidget(status_group, stretch=1)

        splitter.addWidget(bottom_widget)
        splitter.setStretchFactor(0, 2)  # map takes more space
        splitter.setStretchFactor(1, 3)  # table below

    # ── Public slots ──────────────────────────────────────

    def set_drone_position(self, lat, lon, alt):
        """Called from main window on each telemetry update."""
        self._current_lat = lat
        self._current_lon = lon
        self._current_alt = alt
        # Push to map
        if lat != 0.0 or lon != 0.0:
            self.map_view.update_drone_position(lat, lon)
            # Center map on drone the first time we get a valid position
            if not self._map_centered:
                self.map_view.center_on(lat, lon, 18)
                self._map_centered = True

    def _center_on_drone(self):
        if self._current_lat != 0.0 or self._current_lon != 0.0:
            self.map_view.center_on(self._current_lat, self._current_lon, 18)

    def _on_map_click_add_waypoint(self, lat, lon):
        """User clicked on the map → add waypoint at that location."""
        wp = Waypoint(
            command="NAV_WAYPOINT",
            lat=lat, lon=lon, alt=self._default_alt,
        )
        self._add_waypoint_row(wp)
        self._refresh_map_markers()

    def _on_map_marker_moved(self, seq, lat, lon):
        """User dragged a waypoint marker on the map → update the table row."""
        if 0 <= seq < self.table.rowCount():
            self.table.item(seq, 2).setText(f"{lat:.7f}")
            self.table.item(seq, 3).setText(f"{lon:.7f}")
            self._refresh_map_markers()

    def _refresh_map_markers(self):
        """Read current table state and push to map."""
        self.map_view.refresh_waypoints(self.get_waypoints())

    def set_upload_progress(self, current, total):
        self.progress_label.setText(f"Uploading: {current}/{total}")

    def set_mission_progress(self, current_wp, total_wp):
        if total_wp > 0:
            self.progress_label.setText(f"Waypoint {current_wp}/{total_wp}")
        else:
            self.progress_label.setText("No mission active")

    def add_status(self, text):
        import time
        ts = time.strftime("%H:%M:%S")
        self.status_log.append(f"[{ts}] {text}")
        sb = self.status_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Row manipulation ──────────────────────────────────

    def _add_waypoint_row(self, wp):
        """Insert a Waypoint into the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(row)))
        self.table.item(row, 0).setFlags(Qt.ItemIsEnabled)  # read-only seq

        cmd_combo = QComboBox()
        cmd_combo.addItems(COMMAND_NAMES)
        cmd_combo.setCurrentText(wp.command)
        cmd_combo.setStyleSheet("background: #2a2a2a; color: #eee;")
        self.table.setCellWidget(row, 1, cmd_combo)

        self.table.setItem(row, 2, QTableWidgetItem(f"{wp.lat:.7f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{wp.lon:.7f}"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{wp.alt:.1f}"))
        self.table.setItem(row, 5, QTableWidgetItem(f"{wp.p1:.2f}"))
        self.table.setItem(row, 6, QTableWidgetItem(f"{wp.p2:.2f}"))

    def _add_manual(self):
        wp = Waypoint(
            command="NAV_WAYPOINT",
            lat=self._current_lat, lon=self._current_lon, alt=self._default_alt,
        )
        self._add_waypoint_row(wp)
        self._refresh_map_markers()

    def _add_current(self):
        if self._current_lat == 0.0 and self._current_lon == 0.0:
            QMessageBox.warning(
                self, "No GPS",
                "Drone position not available. Connect to the drone first."
            )
            return
        wp = Waypoint(
            command="NAV_WAYPOINT",
            lat=self._current_lat, lon=self._current_lon, alt=max(self._default_alt, self._current_alt),
        )
        self._add_waypoint_row(wp)
        self._refresh_map_markers()

    def _add_return_home(self):
        """Append a Return-to-Launch waypoint as the final mission step.

        When the drone completes all previous waypoints, it executes RTL which
        returns it to the home/launch position and lands.
        """
        wp = Waypoint(
            command="NAV_RETURN_TO_LAUNCH",
            lat=0.0, lon=0.0, alt=0.0,  # ignored for RTL — uses home position
        )
        self._add_waypoint_row(wp)
        self._refresh_map_markers()
        self.add_status("Return-to-Home waypoint added (final mission step)")

    def _add_offset(self):
        if self._current_lat == 0.0 and self._current_lon == 0.0:
            QMessageBox.warning(
                self, "No GPS",
                "Drone position not available. Connect to the drone first."
            )
            return
        dlg = OffsetDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            n, e, alt = dlg.values()
            lat, lon = offset_to_latlon(self._current_lat, self._current_lon, n, e)
            wp = Waypoint(command="NAV_WAYPOINT", lat=lat, lon=lon, alt=alt)
            self._add_waypoint_row(wp)
            self._refresh_map_markers()

    def _import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Waypoint File", "", "Waypoint Files (*.waypoint *.txt)"
        )
        if not path:
            return
        try:
            waypoints = parse_waypoint_file(path)
            self._clear_all()
            for wp in waypoints:
                self._add_waypoint_row(wp)
            self.add_status(f"Imported {len(waypoints)} waypoints from {path}")
            self._refresh_map_markers()
            # Center map on first waypoint
            if waypoints:
                self.map_view.center_on(waypoints[0].lat, waypoints[0].lon, 17)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def _export_file(self):
        waypoints = self.get_waypoints()
        if not waypoints:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Waypoint File", "mission.waypoint",
            "Waypoint Files (*.waypoint)"
        )
        if not path:
            return
        try:
            write_waypoint_file(path, waypoints)
            self.add_status(f"Exported {len(waypoints)} waypoints to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
        self._renumber()
        self._refresh_map_markers()

    def _clear_all(self):
        self.table.setRowCount(0)
        self._refresh_map_markers()

    def _renumber(self):
        for i in range(self.table.rowCount()):
            self.table.item(i, 0).setText(str(i))

    # ── Reading table → waypoint list ─────────────────────

    def get_waypoints(self):
        waypoints = []
        for i in range(self.table.rowCount()):
            try:
                cmd_combo = self.table.cellWidget(i, 1)
                cmd = cmd_combo.currentText() if cmd_combo else "NAV_WAYPOINT"
                lat = float(self.table.item(i, 2).text())
                lon = float(self.table.item(i, 3).text())
                alt = float(self.table.item(i, 4).text())
                p1 = float(self.table.item(i, 5).text() or 0)
                p2 = float(self.table.item(i, 6).text() or 0)
                waypoints.append(Waypoint(
                    seq=i, command=cmd, lat=lat, lon=lon, alt=alt, p1=p1, p2=p2,
                ))
            except (ValueError, AttributeError) as e:
                QMessageBox.warning(self, "Invalid Row", f"Row {i}: {e}")
                return []
        return waypoints

    def _on_upload_clicked(self):
        wps = self.get_waypoints()
        if not wps:
            QMessageBox.warning(self, "Empty Mission", "Add waypoints first.")
            return
        self.upload_requested.emit(wps)

    # ── Styles ────────────────────────────────────────────

    @staticmethod
    def _group_style():
        return """
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #0af;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        """

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{
                background-color: {color}; color: white; border: none;
                padding: 5px 8px; border-radius: 3px; font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {color}; opacity: 0.8; }}
            QPushButton:pressed {{ background-color: #333; }}
            QPushButton:disabled {{ background-color: #555; color: #999; }}
        """
