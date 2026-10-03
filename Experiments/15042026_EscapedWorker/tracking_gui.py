# title: "Animal Tracking GUI with Autodetection"
# date: "10/17/2025"
# author: "Babur Erdem"
# modified: "10/27/2025" # Added Start/End times to info file

import sys
import os
import cv2
import numpy as np
import math
import subprocess
import importlib
from collections import defaultdict

# --- Dependency Checker ---
REQUIRED = [
    ("PySide6", "PySide6"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib")
]


def _ensure(mod_name, pip_name):
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        try:
            print(f"Installing missing package: {pip_name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", pip_name], check=True)
            importlib.import_module(mod_name)
            return True
        except Exception as e:
            print(f"FATAL: Missing package '{pip_name}'. Please install it manually: pip install {pip_name}")
            print(f"Error details: {e}")
            return False


_missing = [p for m, p in REQUIRED if not _ensure(m, p)]
if _missing:
    sys.exit(1)

from PySide6 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from tracking_core import TrackingEngine


# --- Helper Functions ---
def parse_time_str(time_str):
    """Converts hh:mm:ss to seconds."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        return 0


def get_shape_center(shape_data):
    """Calculates the center point of a shape's geometry."""
    geom = shape_data['geom']
    if shape_data['shape'] == 'circle':
        return (geom[0], geom[1])
    elif shape_data['shape'] == 'rect':
        return (geom[0] + geom[2] // 2, geom[1] + geom[3] // 2)
    elif shape_data['shape'] == 'poly':
        pts = np.array(geom)
        return (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))
    return (0, 0)


# --- Custom QGraphicsView for Drawing ---
class InteractiveViewer(QtWidgets.QGraphicsView):
    shapes_updated = QtCore.Signal()
    line_drawn = QtCore.Signal(float, str)  # length, line_type

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)

        self.draw_config = {}
        self.current_shape_item = None
        self.start_pos = None
        self.drawing_poly_points = []
        self.item_map = {}  # Maps data object ID to QGraphicsItem
        self._preview_items = []  # For blob preview

    # --- Methods for Blob Preview ---
    def clear_preview_blobs(self):
        """Removes temporary blob markers from the canvas."""
        for item in self._preview_items:
            self.scene().removeItem(item)
        self._preview_items = []

    def show_preview_blobs(self, keypoints):
        """Draws temporary blob markers on the canvas."""
        self.clear_preview_blobs()
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 200), 1)  # Red outline
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 50))  # Transparent red fill
        for kp in keypoints:
            x, y = kp.pt
            r = kp.size / 2.0
            if r < 1: r = 1  # Ensure visibility
            ellipse = self.scene().addEllipse(x - r, y - r, 2 * r, 2 * r, pen, brush)
            ellipse.setZValue(10)  # Draw on top
            self._preview_items.append(ellipse)

    # --- End Preview Methods ---

    def keyPressEvent(self, event):
        """Handle key presses, specifically the Delete key."""
        if event.key() == QtCore.Qt.Key.Key_Delete:
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def set_frame(self, frame_bgr):
        if frame_bgr is None: return
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(frame_bgr.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self._pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, QtCore.Qt.KeepAspectRatio)
        self.clear_preview_blobs()  # Clear preview when frame changes

    def start_drawing(self, shape, category):
        self.draw_config = {'shape': shape, 'category': category}
        if shape == 'poly':
            self.drawing_poly_points = []
        self.setCursor(QtCore.Qt.CrossCursor)

    def finish_drawing(self):
        shape = self.draw_config.get('shape')
        if self.current_shape_item and shape == 'poly' and self.drawing_poly_points:
            self.scene().removeItem(self.current_shape_item)
            self.current_shape_item = None

        if shape == 'poly' and len(self.drawing_poly_points) > 2:
            geom = [(p.x(), p.y()) for p in self.drawing_poly_points]
            data = {'shape': 'poly', 'category': self.draw_config.get('category'), 'geom': geom}
            self.add_shape_item(data)

        self.drawing_poly_points = []
        self.draw_config = {}
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.shapes_updated.emit()

    def add_shape_item(self, data):
        """Adds a shape to the scene based on a data dictionary."""
        category = data.get('category')
        shape = data.get('shape')
        geom = data.get('geom')

        color = QtGui.QColor(0, 150, 255) if category == 'arena' else QtGui.QColor(255, 200, 0)
        pen = QtGui.QPen(color, 2, QtCore.Qt.SolidLine)
        brush = QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 50))

        item = None
        if shape == 'circle':
            cx, cy, r = geom
            rect = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
            item = self.scene().addEllipse(rect, pen, brush)
        elif shape == 'rect':
            x, y, w, h = geom
            item = self.scene().addRect(QtCore.QRectF(x, y, w, h), pen, brush)
        elif shape == 'poly':
            polygon = QtGui.QPolygonF([QtCore.QPointF(p[0], p[1]) for p in geom])
            item = self.scene().addPolygon(polygon, pen, brush)

        if item:
            item.setFlags(QtWidgets.QGraphicsItem.ItemIsSelectable | QtWidgets.QGraphicsItem.ItemIsMovable)
            item.setData(0, data)
            self.item_map[id(data)] = item
            text_item = QtWidgets.QGraphicsSimpleTextItem(item)
            text_item.setBrush(QtCore.Qt.white)
            text_item.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
            text_item.setZValue(1)
        return item

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        if not self.draw_config:
            super().mousePressEvent(event)
            return

        shape = self.draw_config.get('shape')
        if shape in ['circle', 'rect', 'line', 'measure_length', 'measure_width'] or shape.startswith(
                'measure_diameter_'):
            self.start_pos = pos
        elif shape == 'poly':
            self.drawing_poly_points.append(pos)
            if len(self.drawing_poly_points) > 1:
                if self.current_shape_item: self.scene().removeItem(self.current_shape_item)
                pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 200), 2, QtCore.Qt.DashLine)
                poly = QtGui.QPolygonF(self.drawing_poly_points)
                self.current_shape_item = self.scene().addPolygon(poly, pen)

    def mouseMoveEvent(self, event):
        if not self.start_pos or self.draw_config.get('shape') == 'poly':
            super().mouseMoveEvent(event)
            return

        if self.current_shape_item: self.scene().removeItem(self.current_shape_item)
        pos = self.mapToScene(event.position().toPoint())
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 200), 2, QtCore.Qt.DashLine)
        shape = self.draw_config.get('shape')

        if shape == 'rect':
            self.current_shape_item = self.scene().addRect(QtCore.QRectF(self.start_pos, pos).normalized(), pen)
        elif shape == 'circle':
            radius = math.sqrt((pos.x() - self.start_pos.x()) ** 2 + (pos.y() - self.start_pos.y()) ** 2)
            rect = QtCore.QRectF(self.start_pos.x() - radius, self.start_pos.y() - radius, 2 * radius, 2 * radius)
            self.current_shape_item = self.scene().addEllipse(rect, pen)
        elif shape in ['line', 'measure_length', 'measure_width'] or shape.startswith('measure_diameter_'):
            self.current_shape_item = self.scene().addLine(QtCore.QLineF(self.start_pos, pos), pen)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        if self.draw_config and self.start_pos:
            shape, category = self.draw_config.get('shape'), self.draw_config.get('category')

            if shape in ['circle', 'rect'] and self.current_shape_item:
                scene_br = self.current_shape_item.sceneBoundingRect()
                geom = None
                if shape == 'circle':
                    center = scene_br.center()
                    radius = scene_br.width() / 2.0
                    geom = [int(center.x()), int(center.y()), int(radius)]
                else:  # rect
                    geom = [int(scene_br.x()), int(scene_br.y()), int(scene_br.width()), int(scene_br.height())]

                data = {'shape': shape, 'category': category, 'geom': geom}
                self.add_shape_item(data)
                self.shapes_updated.emit()

            elif shape in ['line', 'measure_length', 'measure_width'] or shape.startswith('measure_diameter_'):
                end_pos = self.mapToScene(event.position().toPoint())
                length = math.sqrt((end_pos.x() - self.start_pos.x()) ** 2 + (end_pos.y() - self.start_pos.y()) ** 2)
                self.line_drawn.emit(length, shape)

            if self.current_shape_item: self.scene().removeItem(self.current_shape_item)
            if shape != 'poly':
                self.current_shape_item = None
                self.start_pos = None
                self.draw_config = {}
                self.setCursor(QtCore.Qt.ArrowCursor)
        else:
            self.shapes_updated.emit()

    def wheelEvent(self, event):
        selected = self.scene().selectedItems()
        if not selected:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            return
        item = selected[0]
        data = item.data(0)
        if data and data['shape'] == 'circle':
            delta = 5 if event.angleDelta().y() > 0 else -5

            scene_br = item.sceneBoundingRect()
            center = scene_br.center()
            new_r = max(5, scene_br.width() / 2 + delta)

            item.setPos(center.x() - new_r, center.y() - new_r)
            item.setRect(0, 0, 2 * new_r, 2 * new_r)

            data['geom'] = [int(center.x()), int(center.y()), int(new_r)]
            item.setData(0, data)
            self.shapes_updated.emit()

    def get_all_shapes_data(self):
        """Return shapes as dicts and update on-canvas IDs using row-major ordering."""

        def _center_and_vsize(data):
            geom = data['geom']
            if data['shape'] == 'circle':
                return (int(geom[0]), int(geom[1])), float(geom[2])
            elif data['shape'] == 'rect':
                x, y, w, h = geom
                return (int(x + w // 2), int(y + h // 2)), float(h / 2.0)
            elif data['shape'] == 'poly':
                xs = [p[0] for p in geom];
                ys = [p[1] for p in geom]
                cx = int(sum(xs) / len(xs));
                cy = int(sum(ys) / len(ys))
                vsize = (max(ys) - min(ys)) / 2.0 if len(ys) >= 2 else 20.0
                return (cx, cy), float(vsize)
            return (0, 0), 20.0

        arena_items, stim_items = [], []
        for item in self.scene().items():
            data = item.data(0)
            if not data or 'category' not in data: continue

            scene_br = item.sceneBoundingRect()
            if data['shape'] == 'circle':
                center = scene_br.center()
                data['geom'] = [int(center.x()), int(center.y()), int(scene_br.width() / 2)]
            elif data['shape'] == 'rect':
                data['geom'] = [int(scene_br.x()), int(scene_br.y()), int(scene_br.width()), int(scene_br.height())]
            elif data['shape'] == 'poly':
                scene_poly = item.mapToScene(item.polygon())
                data['geom'] = [(int(p.x()), int(p.y())) for p in scene_poly]

            (cx, cy), vsize = _center_and_vsize(data)
            rec = (item, data, (cx, cy), vsize)

            if data['category'] == 'arena':
                arena_items.append(rec)
            elif data['category'] == 'stim':
                stim_items.append(rec)

        def _row_major_sort(recs):
            if not recs: return []
            vs = sorted(v for *_, v in recs)
            tol = vs[len(vs) // 2] if vs else 20.0
            tol = max(10.0, tol * 0.75)
            recs_sorted = sorted(recs, key=lambda r: (r[2][1], r[2][0]))
            rows = []
            for r in recs_sorted:
                placed = False
                for row in rows:
                    mean_y = sum(rr[2][1] for rr in row) / len(row)
                    if abs(r[2][1] - mean_y) <= tol:
                        row.append(r);
                        placed = True;
                        break
                if not placed: rows.append([r])
            rows.sort(key=lambda row: sum(rr[2][1] for rr in row) / len(row))
            flat = []
            for row in rows:
                row.sort(key=lambda rr: rr[2][0])
                flat.extend(row)
            return flat

        arena_ordered = _row_major_sort(arena_items)
        stim_ordered = _row_major_sort(stim_items)

        def _apply_ids(ordered, prefix):
            out = []
            for i, (item, data, center, _) in enumerate(ordered, start=1):
                data['id'] = f"{prefix}{i}"
                text_item = next((c for c in item.childItems() if isinstance(c, QtWidgets.QGraphicsSimpleTextItem)),
                                 None)
                if not text_item:
                    text_item = QtWidgets.QGraphicsSimpleTextItem(item)
                    text_item.setBrush(QtCore.Qt.white)
                    text_item.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
                    text_item.setZValue(1)
                text_item.setText(data['id'])
                sr = item.boundingRect();
                tr = text_item.boundingRect()
                text_item.setPos(sr.center().x() - tr.width() / 2, sr.center().y() - tr.height() / 2)
                out.append(data)
            return out

        arenas_data = _apply_ids(arena_ordered, "A")
        stims_data = _apply_ids(stim_ordered, "S")
        return arenas_data + stims_data

    def delete_selected(self):
        """Finds selected items in the scene and removes them."""
        items_to_remove = self.scene().selectedItems()
        if not items_to_remove: return
        for item in items_to_remove:
            data = item.data(0)
            if data and id(data) in self.item_map: del self.item_map[id(data)]
            self.scene().removeItem(item)
        self.shapes_updated.emit()

    def clear_shapes_by_category(self, category_to_clear):
        """Removes all shapes of a specific category from the scene."""
        items_to_remove = []
        data_ids_to_remove = set()

        for item in self.scene().items():
            data = item.data(0)
            if data and data.get('category') == category_to_clear:
                items_to_remove.append(item)
                data_ids_to_remove.add(id(data))

        if not items_to_remove:
            return

        for item in items_to_remove:
            self.scene().removeItem(item)

        for data_id in data_ids_to_remove:
            if data_id in self.item_map:
                del self.item_map[data_id]

        self.shapes_updated.emit()


# --- Canvas Window ---
class CanvasWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canvas")
        self.viewer = InteractiveViewer(self)
        self.setCentralWidget(self.viewer)
        self.resize(800, 600)


# --- Analysis Worker ---
class AnalysisWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal(object, object)

    def __init__(self, settings):
        super().__init__();
        self.settings = settings

    def run(self):
        try:
            engine = TrackingEngine(self.settings)
            raw, filtered = engine.run_tracking(self.progress.emit)
            # Analysis results are calculated but not saved to separate files
            analysis_results = engine.analyze_results(filtered)
            self.finished.emit(
                {'raw_coords': raw, 'filtered_coords': filtered, 'analysis': analysis_results, 'engine': engine},
                None)
        except Exception as e:
            self.finished.emit(None, e)


# --- Main Application Controller ---
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.settings = {}
        self.measured_length = None
        self.measured_width = None
        self.all_shapes_cache = []
        self.canvas = CanvasWindow()
        self.controls = self.create_control_window()
        self.canvas.show()
        self.controls.show()

    def _create_autodetect_widget(self, category):
        """Helper to create the autodetection UI group."""
        group = QtWidgets.QGroupBox("Autodetection")
        layout = QtWidgets.QFormLayout(group)
        diameter_widget = QtWidgets.QWidget()
        diameter_layout = QtWidgets.QHBoxLayout(diameter_widget)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        edit_diameter = QtWidgets.QLineEdit("100")
        btn_measure_diameter = QtWidgets.QPushButton("Measure")
        diameter_layout.addWidget(edit_diameter)
        diameter_layout.addWidget(btn_measure_diameter)
        spin_threshold = QtWidgets.QSpinBox(minimum=0, maximum=255, value=127)
        btn_detect_circ = QtWidgets.QPushButton("Autodetect Circles")
        btn_detect_rect = QtWidgets.QPushButton("Autodetect Rectangles")
        btn_remove_category_areas = QtWidgets.QPushButton(f"Remove All {category.capitalize()} Areas")

        layout.addRow("Expected Diameter/Diagonal (px):", diameter_widget)
        layout.addRow("Detection Threshold (0-255):", spin_threshold)
        layout.addRow(btn_detect_circ)
        layout.addRow(btn_detect_rect)
        layout.addRow(btn_remove_category_areas)

        if category == 'arena':
            self.edit_arena_diameter = edit_diameter
            self.spin_arena_threshold = spin_threshold
        else:
            self.edit_stim_diameter = edit_diameter
            self.spin_stim_threshold = spin_threshold

        btn_measure_diameter.clicked.connect(lambda: self.measure_for_autodetect(category))
        btn_detect_circ.clicked.connect(lambda: self.autodetect_shapes(category, 'circle'))
        btn_detect_rect.clicked.connect(lambda: self.autodetect_shapes(category, 'rect'))
        btn_remove_category_areas.clicked.connect(lambda: self.canvas.viewer.clear_shapes_by_category(category))

        return group

    def _create_blob_params_widget(self):
        """Helper to create Blob Detection specific parameters UI group."""
        group = QtWidgets.QGroupBox("Blob Detection Parameters")
        layout = QtWidgets.QFormLayout(group)

        # Blur
        self.spin_blur_kernel_blob = QtWidgets.QSpinBox(minimum=1, maximum=21, value=5, singleStep=2)
        layout.addRow("Blur Kernel Size (Odd):", self.spin_blur_kernel_blob)

        # Thresholds
        self.spin_blob_min_thresh = QtWidgets.QSpinBox(minimum=0, maximum=255, value=50)
        self.spin_blob_max_thresh = QtWidgets.QSpinBox(minimum=0, maximum=255, value=220)
        layout.addRow("Min Threshold:", self.spin_blob_min_thresh)
        layout.addRow("Max Threshold:", self.spin_blob_max_thresh)

        # --- Connect signals for live preview ---
        self.spin_blur_kernel_blob.valueChanged.connect(self.update_blob_preview)
        self.spin_blob_min_thresh.valueChanged.connect(self.update_blob_preview)
        self.spin_blob_max_thresh.valueChanged.connect(self.update_blob_preview)

        return group

    def update_tracking_params_ui(self):
        """Shows/hides tracking parameter groups based on method selection."""
        method = self.combo_track_method.currentText()
        if method == "Blob Detection":
            self.blob_params_group.show()
            self.update_blob_preview()  # Trigger preview when switching
        else:
            self.blob_params_group.hide()
            self.canvas.viewer.clear_preview_blobs()  # Clear preview if switching away

    def update_blob_preview(self):
        """Processes the current frame and shows detected blobs based on GUI settings."""
        if self.combo_track_method.currentText() != "Blob Detection":
            return
        if 'first_frame' not in self.settings:
            return  # No frame loaded

        frame = self.settings['first_frame']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Read current GUI parameters ---
        blur_ksize = self.spin_blur_kernel_blob.value()
        min_thresh = self.spin_blob_min_thresh.value()
        max_thresh = self.spin_blob_max_thresh.value()
        min_area = self.spin_min_area.value()
        max_area = self.spin_max_area.value()

        # Basic validation
        if blur_ksize % 2 == 0: blur_ksize += 1  # Ensure odd

        # --- Preprocessing ---
        processed_image = cv2.medianBlur(gray, blur_ksize)

        # --- Blob Detection Setup ---
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = min_thresh
        params.maxThreshold = max_thresh
        params.thresholdStep = 10
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByColor = False
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False

        try:
            detector = cv2.SimpleBlobDetector_create(params)
            # Detect blobs on the preprocessed grayscale image
            keypoints = detector.detect(processed_image)
            self.canvas.viewer.show_preview_blobs(keypoints)
        except Exception as e:
            print(f"Error during blob detection preview: {e}")
            self.canvas.viewer.clear_preview_blobs()

    # --- Help Dialog ---
    def show_scrollable_help(self):
        """Displays help text in a scrollable dialog."""
        help_text = """
        <h2>Animal Tracking GUI Help</h2>

        <h3>1. Setup</h3>
        <ul>
        <li><b>Load Video:</b> Select the video file for tracking.</li>
        <li><b>Experiment Name:</b> Name for output files (e.g., 'TestRun1').</li>
        <li><b>Output Directory:</b> Where results (_info.txt, _coordinates.txt, etc.) will be saved. Defaults to video directory.</li>
        <li><b>Start/End Time:</b> Video segment to analyze (hh:mm:ss). Updates the canvas view.</li>
        <li><b>Scale:</b> Define real-world scale (mm per pixel). Enter known length (mm), click 'Draw Scale Line', draw corresponding line on canvas. Scale is saved in _info.txt.</li>
        <li><b>Measure Animal Size:</b> Click buttons and draw lines on the animal (canvas) to estimate Length/Width. This suggests Area thresholds.</li>
        <li><b>Animal Size Threshold:</b> Manually set Min/Max Area (pixels²) to filter detected objects. Only objects within this size range are tracked. Crucial for good results.</li>
        <li><b>Tracking Method:</b>
            <ul>
                <li><i>Background Subtraction:</i> Good for stable lighting. Detects moving objects against a static background (uses default sensitivity).</li>
                <li><i>Blob Detection:</i> Better for variable lighting. Uses blurring and internal thresholding to find blob-like objects based on Area.</li>
            </ul>
        </li>
        </ul>

        <h3>Blob Detection Parameters</h3> (Visible only when Blob Detection is selected)
        <ul>
        <li><b>Blur Kernel Size:</b> Amount of blurring (Median Blur) applied before detection (Odd number, e.g., 3, 5, 7). Higher values reduce more noise but can merge close objects or lose small ones.</li>
        <li><b>Min/Max Threshold:</b> Grayscale intensity range (0-255) used internally by the detector to find blobs. Adjust if animals are very dark (lower Max) or very light (raise Min). (Live preview on canvas updates as you change these).</li>
        </ul>
        <p><i>Live Preview:</i> When Blob Detection is active, changing parameters will show detected blobs (red circles) on the current canvas frame instantly.</p>

        <h3>2. Define Areas</h3> (Use Tabs: Arenas / Stimulus Areas)
        <ul>
        <li><b>Manual Drawing:</b> Click Circle/Rect/Poly to draw shapes on the canvas. Click 'Finish Poly' to complete polygon drawing. Arenas define where to track; Stimulus Areas are for analysis (time spent inside).</li>
        <li><b>Autodetection:</b>
            <ul>
                <li><i>Measure:</i> Measure diameter/diagonal of an example object on canvas.</li>
                <li><i>Threshold:</i> Sensitivity for detection (0-255). Lower is more sensitive.</li>
                <li><i>Autodetect Buttons:</i> Try to automatically find circles or rectangles of the measured size using Hough Circles or contour approximation.</li>
                <li><i>Remove All:</i> Clears all shapes of that category (Arena or Stim).</li>
            </ul>
        <li><b>Canvas Interaction:</b> Select shapes by clicking. Move selected shapes by dragging. Delete selected shapes using the 'Delete' key on your keyboard. Resize circles using the mouse wheel while selected. Shape IDs (A1, S1, etc.) update automatically based on position.</li>
        </ul>

        <h3>3. Visualization</h3>
        <ul>
        <li><b>Save Track Video:</b> Creates an MP4 video showing detected points (Red=Raw detection, Green=Kalman Filtered position).</li>
        <li><b>Draw Raw Trajectories (Image):</b> Includes the raw detection path (pale, slightly thicker line) in the `_trajectories.png` output image.</li>
        <li><b>Draw Filtered Trajectories (Image):</b> Includes the Kalman filtered path (solid, thickest line with gaps for lost tracking) in the `_trajectories.png` output image.</li>
        </ul>

        <h3>4. Run</h3>
        <ul>
        <li><b>Run Tracking & Analysis:</b> Starts the main processing using current settings and drawn shapes.</li>
        <li><b>Progress Bar:</b> Shows analysis progress.</li>
        </ul>

        <p><b>Help/Exit:</b> Shows this message / Closes the application.</p>
        """

        dialog = QtWidgets.QDialog(self.controls)
        dialog.setWindowTitle("Help")
        layout = QtWidgets.QVBoxLayout(dialog)

        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(help_text)  # Use setHtml for rich text

        scroll_area_help = QtWidgets.QScrollArea()  # Create scroll area
        scroll_area_help.setWidgetResizable(True)
        scroll_area_help.setWidget(text_edit)  # Put text edit inside scroll area

        layout.addWidget(scroll_area_help)  # Add scroll area to dialog layout

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)

        dialog.resize(600, 500)  # Make dialog reasonably sized
        dialog.exec()

    def create_control_window(self):
        win = QtWidgets.QMainWindow();
        win.setWindowTitle("Controls")
        self.btn_load_video = QtWidgets.QPushButton("Load Video")
        self.lbl_video_name = QtWidgets.QLabel("No video loaded.")
        self.lbl_video_name.setStyleSheet("font-style: italic;")
        self.edit_exp_name = QtWidgets.QLineEdit("MyExperiment")
        self.edit_start_time = QtWidgets.QLineEdit("00:00:00")
        self.edit_end_time = QtWidgets.QLineEdit("00:00:10")
        self.edit_output_dir = QtWidgets.QLineEdit();
        self.btn_browse_dir = QtWidgets.QPushButton("Browse...")
        self.edit_scale_len = QtWidgets.QLineEdit("10.0");
        self.btn_draw_scale = QtWidgets.QPushButton("Draw Scale Line")
        self.lbl_scale = QtWidgets.QLabel("Scale: Not Set")
        self.btn_measure_len = QtWidgets.QPushButton("Measure Length");
        self.btn_measure_wid = QtWidgets.QPushButton("Measure Width")
        self.lbl_animal_size = QtWidgets.QLabel("Animal Size: Not Set")
        self.spin_min_area = QtWidgets.QSpinBox(minimum=0, maximum=100000, value=100)
        self.spin_max_area = QtWidgets.QSpinBox(minimum=1, maximum=200000, value=10000)

        # --- Simplified Tracking Method Selection ---
        self.combo_track_method = QtWidgets.QComboBox()
        self.combo_track_method.addItems([
            "Background Subtraction",
            "Blob Detection"
        ])

        self.btn_add_arena_circ = QtWidgets.QPushButton("Circle");
        self.btn_add_arena_rect = QtWidgets.QPushButton("Rect");
        self.btn_add_arena_poly = QtWidgets.QPushButton("Poly");
        self.btn_arena_poly_done = QtWidgets.QPushButton("Finish Poly")
        self.btn_add_stim_circ = QtWidgets.QPushButton("Circle");
        self.btn_add_stim_rect = QtWidgets.QPushButton("Rect");
        self.btn_add_stim_poly = QtWidgets.QPushButton("Poly");
        self.btn_stim_poly_done = QtWidgets.QPushButton("Finish Poly")
        self.btn_run = QtWidgets.QPushButton("Run Tracking & Analysis")
        self.progress_bar = QtWidgets.QProgressBar();
        # --- Visualization Checkboxes ---
        self.chk_save_video = QtWidgets.QCheckBox("Save Track Video");
        self.chk_save_video.setChecked(True)
        self.chk_draw_raw_img = QtWidgets.QCheckBox("Draw Raw Trajectories (Image)")
        self.chk_draw_raw_img.setChecked(True)
        self.chk_draw_filtered_img = QtWidgets.QCheckBox("Draw Filtered Trajectories (Image)")
        self.chk_draw_filtered_img.setChecked(True)

        self.btn_help = QtWidgets.QPushButton("Help")  # Help Button
        self.btn_exit = QtWidgets.QPushButton("Exit Application")

        scroll_area = QtWidgets.QScrollArea();
        scroll_area.setWidgetResizable(True);
        win.setCentralWidget(scroll_area)
        main_widget = QtWidgets.QWidget();
        scroll_area.setWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)

        # --- 1. Setup ---
        file_group = QtWidgets.QGroupBox("1. Setup");
        file_layout = QtWidgets.QFormLayout(file_group)
        file_layout.addRow(self.btn_load_video)
        file_layout.addRow("Loaded Video:", self.lbl_video_name)
        file_layout.addRow("Experiment Name:", self.edit_exp_name)
        dir_layout = QtWidgets.QHBoxLayout();
        dir_layout.addWidget(self.edit_output_dir);
        dir_layout.addWidget(self.btn_browse_dir)
        file_layout.addRow("Output Directory:", dir_layout)
        file_layout.addRow("Start Time (h:m:s):", self.edit_start_time);
        file_layout.addRow("End Time (h:m:s):", self.edit_end_time)
        scale_layout = QtWidgets.QHBoxLayout();
        scale_layout.addWidget(QtWidgets.QLabel("Scale Length (mm):"));
        scale_layout.addWidget(self.edit_scale_len);
        scale_layout.addWidget(self.btn_draw_scale)
        file_layout.addRow(scale_layout);
        file_layout.addRow(self.lbl_scale)
        measure_layout = QtWidgets.QHBoxLayout();
        measure_layout.addWidget(self.btn_measure_len);
        measure_layout.addWidget(self.btn_measure_wid)
        file_layout.addRow("Measure Animal Size:", measure_layout);
        file_layout.addRow(self.lbl_animal_size)

        area_layout = QtWidgets.QHBoxLayout();
        area_layout.addWidget(QtWidgets.QLabel("Min Area (px²):"));
        area_layout.addWidget(self.spin_min_area);
        area_layout.addWidget(QtWidgets.QLabel("Max Area (px²):"));
        area_layout.addWidget(self.spin_max_area)
        file_layout.addRow("Animal Size Threshold:", area_layout)

        # --- Add tracking method combo ---
        file_layout.addRow("Tracking Method:", self.combo_track_method)

        # --- Add Blob specific params widget ---
        self.blob_params_group = self._create_blob_params_widget()
        file_layout.addRow(self.blob_params_group)

        main_layout.addWidget(file_group)

        # --- 2. Define Areas ---
        tabs_group = QtWidgets.QGroupBox("2. Define Areas");
        tabs_layout = QtWidgets.QVBoxLayout(tabs_group)
        tab_widget = QtWidgets.QTabWidget();
        arena_tab = QtWidgets.QWidget();
        stim_tab = QtWidgets.QWidget()
        tab_widget.addTab(arena_tab, "Arenas");
        tab_widget.addTab(stim_tab, "Stimulus Areas")
        tabs_layout.addWidget(tab_widget)
        main_layout.addWidget(tabs_group)

        arena_layout = QtWidgets.QVBoxLayout(arena_tab);
        arena_manual_group = QtWidgets.QGroupBox("Manual Drawing");
        arena_manual_layout = QtWidgets.QHBoxLayout(arena_manual_group)
        arena_manual_layout.addWidget(self.btn_add_arena_circ);
        arena_manual_layout.addWidget(self.btn_add_arena_rect);
        arena_manual_layout.addWidget(self.btn_add_arena_poly);
        arena_manual_layout.addWidget(self.btn_arena_poly_done)
        arena_layout.addWidget(arena_manual_group)
        self.arena_autodetect_group = self._create_autodetect_widget('arena')
        arena_layout.addWidget(self.arena_autodetect_group)

        stim_layout = QtWidgets.QVBoxLayout(stim_tab);
        stim_manual_group = QtWidgets.QGroupBox("Manual Drawing");
        stim_manual_layout = QtWidgets.QHBoxLayout(stim_manual_group)
        stim_manual_layout.addWidget(self.btn_add_stim_circ);
        stim_manual_layout.addWidget(self.btn_add_stim_rect);
        stim_manual_layout.addWidget(self.btn_add_stim_poly);
        stim_manual_layout.addWidget(self.btn_stim_poly_done)
        stim_layout.addWidget(stim_manual_group)
        self.stim_autodetect_group = self._create_autodetect_widget('stim')
        stim_layout.addWidget(self.stim_autodetect_group)

        # --- 3. Visualization ---
        vis_group = QtWidgets.QGroupBox("3. Visualization");
        vis_layout = QtWidgets.QFormLayout(vis_group)
        vis_layout.addRow(self.chk_save_video);
        vis_layout.addRow(self.chk_draw_raw_img);
        vis_layout.addRow(self.chk_draw_filtered_img);
        main_layout.addWidget(vis_group)

        # --- 4. Run ---
        run_group = QtWidgets.QGroupBox("4. Run");
        run_layout = QtWidgets.QFormLayout(run_group)
        run_layout.addRow(self.btn_run);
        run_layout.addRow(self.progress_bar)
        main_layout.addWidget(run_group)

        main_layout.addStretch();
        # --- Help and Exit Buttons ---
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.btn_help)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_exit)
        main_layout.addLayout(button_layout)

        # Connect signals
        self.btn_load_video.clicked.connect(self.load_video);
        self.btn_browse_dir.clicked.connect(self.select_output_directory);
        self.edit_start_time.editingFinished.connect(self.update_canvas_frame)  # Calls update_blob_preview
        self.btn_draw_scale.clicked.connect(self.draw_scale);
        self.btn_measure_len.clicked.connect(lambda: self.canvas.viewer.start_drawing('measure_length', 'measure'));
        self.btn_measure_wid.clicked.connect(lambda: self.canvas.viewer.start_drawing('measure_width', 'measure'))
        self.canvas.viewer.line_drawn.connect(self.handle_line_drawn);
        self.canvas.viewer.shapes_updated.connect(self.update_shape_lists)
        self.combo_track_method.currentIndexChanged.connect(self.update_tracking_params_ui)  # Connect method change
        self.spin_min_area.valueChanged.connect(self.update_blob_preview)  # Connect area changes to preview
        self.spin_max_area.valueChanged.connect(self.update_blob_preview)  # Connect area changes to preview
        self.btn_add_arena_circ.clicked.connect(lambda: self.canvas.viewer.start_drawing('circle', 'arena'));
        self.btn_add_arena_rect.clicked.connect(lambda: self.canvas.viewer.start_drawing('rect', 'arena'));
        self.btn_add_arena_poly.clicked.connect(lambda: self.canvas.viewer.start_drawing('poly', 'arena'));
        self.btn_arena_poly_done.clicked.connect(self.canvas.viewer.finish_drawing)
        self.btn_add_stim_circ.clicked.connect(lambda: self.canvas.viewer.start_drawing('circle', 'stim'));
        self.btn_add_stim_rect.clicked.connect(lambda: self.canvas.viewer.start_drawing('rect', 'stim'));
        self.btn_add_stim_poly.clicked.connect(lambda: self.canvas.viewer.start_drawing('poly', 'stim'));
        self.btn_stim_poly_done.clicked.connect(self.canvas.viewer.finish_drawing)
        self.btn_run.clicked.connect(self.run_analysis);
        self.btn_help.clicked.connect(self.show_scrollable_help)  # Connect help button
        self.btn_exit.clicked.connect(self.close_app)

        # Initial setup
        self.set_controls_enabled(False);
        self.btn_load_video.setEnabled(True)
        self.update_tracking_params_ui()  # Set initial visibility

        return win

    def close_app(self):
        QtWidgets.QApplication.instance().quit()

    def set_controls_enabled(self, enabled):
        widgets = [self.edit_start_time, self.edit_end_time, self.edit_scale_len, self.btn_draw_scale,
                   self.btn_add_arena_circ, self.btn_add_arena_rect, self.btn_add_arena_poly, self.btn_arena_poly_done,
                   self.btn_add_stim_circ, self.btn_add_stim_rect, self.btn_add_stim_poly, self.btn_stim_poly_done,
                   self.btn_run, self.chk_save_video, self.edit_output_dir, self.btn_browse_dir, self.btn_measure_len,
                   self.btn_measure_wid, self.spin_min_area, self.spin_max_area, self.combo_track_method,
                   self.blob_params_group, self.chk_draw_raw_img, self.chk_draw_filtered_img]  # Added vis checkboxes
        for w in widgets: w.setEnabled(enabled)
        if hasattr(self, 'arena_autodetect_group'):
            self.arena_autodetect_group.setEnabled(enabled)
            self.stim_autodetect_group.setEnabled(enabled)

    def load_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.controls, "Open Video", "",
                                                             "Video Files (*.mp4 *.avi *.mov)")
        if not file_path: return
        self.settings['video_path'] = file_path
        self.lbl_video_name.setText(os.path.basename(file_path))
        self.edit_output_dir.setText(os.path.dirname(file_path))
        self.update_canvas_frame();
        self.set_controls_enabled(True)
        self.update_tracking_params_ui()

    def select_output_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self.controls, "Select Output Directory")
        if directory: self.edit_output_dir.setText(directory)

    def update_canvas_frame(self):
        if 'video_path' not in self.settings: return
        cap = cv2.VideoCapture(self.settings['video_path'])
        if not cap.isOpened(): return
        self.settings['fps'] = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = int(parse_time_str(self.edit_start_time.text()) * self.settings['fps'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read();
        cap.release()
        if ret:
            self.settings['first_frame'] = frame;
            self.canvas.viewer.set_frame(frame)
            self.update_blob_preview()  # Update preview if needed
        else:
            QtWidgets.QMessageBox.warning(self.controls, "Warning",
                                          f"Could not seek to {self.edit_start_time.text()}. Showing first frame.");
            self.edit_start_time.setText("00:00:00");
            self.update_canvas_frame()

    def draw_scale(self):
        self.canvas.viewer.start_drawing('line', 'scale')

    def handle_line_drawn(self, pixel_length, line_type):
        if line_type == 'line':
            try:
                real_length = float(self.edit_scale_len.text())
                if real_length > 0 and pixel_length > 0:
                    mm_per_pixel = real_length / pixel_length
                    self.settings['pixel_to_mm'] = mm_per_pixel
                    self.lbl_scale.setText(f"Scale: 1 mm = {1.0 / mm_per_pixel:.2f} pixels")
            except ValueError:
                QtWidgets.QMessageBox.warning(self.controls, "Input Error",
                                              "Please enter a valid number for the scale length.")
        elif line_type == 'measure_length':
            self.measured_length = pixel_length;
            self.update_animal_size()
        elif line_type == 'measure_width':
            self.measured_width = pixel_length;
            self.update_animal_size()
        elif line_type == 'measure_diameter_arena':
            self.edit_arena_diameter.setText(f"{pixel_length:.1f}")
        elif line_type == 'measure_diameter_stim':
            self.edit_stim_diameter.setText(f"{pixel_length:.1f}")

    def update_animal_size(self):
        text = []
        if self.measured_length: text.append(f"L: {self.measured_length:.1f}px")
        if self.measured_width: text.append(f"W: {self.measured_width:.1f}px")
        if self.measured_length and self.measured_width:
            area = self.measured_length * self.measured_width
            text.append(f"Area: ~{area:.0f} px²")
            # Automatically suggest area thresholds
            if self.spin_min_area.value() == 100:
                self.spin_min_area.setValue(max(1, int(area * 0.3)))
            if self.spin_max_area.value() == 10000:
                self.spin_max_area.setValue(int(area * 3.0))

        self.lbl_animal_size.setText(" | ".join(text))

    def update_shape_lists(self):
        self.canvas.viewer.get_all_shapes_data()

    def measure_for_autodetect(self, category):
        self.canvas.viewer.start_drawing(f'measure_diameter_{category}', 'measure')

    def autodetect_shapes(self, category, shape_type):
        if 'first_frame' not in self.settings:
            QtWidgets.QMessageBox.warning(self.controls, "Error", "Load a video first.")
            return
        if category == 'arena':
            diameter_str = self.edit_arena_diameter.text()
            threshold_val = self.spin_arena_threshold.value()
        else:
            diameter_str = self.edit_stim_diameter.text()
            threshold_val = self.spin_stim_threshold.value()
        try:
            expected_size = float(diameter_str)
            if expected_size <= 0: raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self.controls, "Input Error",
                                          "Please enter a valid positive number for diameter/diagonal.")
            return
        frame = self.settings['first_frame']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        found_count = 0
        if shape_type == 'circle':
            expected_radius = expected_size / 2.0
            radius_tolerance = 0.20
            min_rad = int(expected_radius * (1 - radius_tolerance))
            max_rad = int(expected_radius * (1 + radius_tolerance))
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min_rad * 2, param1=50, param2=threshold_val,
                                       minRadius=min_rad, maxRadius=max_rad)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for c in circles[0, :]:
                    geom = [int(c[0]), int(c[1]), int(c[2])]
                    data = {'shape': 'circle', 'category': category, 'geom': geom}
                    self.canvas.viewer.add_shape_item(data)
                    found_count += 1
        elif shape_type == 'rect':
            size_tolerance = 0.25
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    diagonal = math.sqrt(w ** 2 + h ** 2)
                    if abs(diagonal - expected_size) < expected_size * size_tolerance:
                        geom = [x, y, w, h]
                        data = {'shape': 'rect', 'category': category, 'geom': geom}
                        self.canvas.viewer.add_shape_item(data)
                        found_count += 1
        if found_count > 0:
            self.canvas.viewer.shapes_updated.emit()
            QtWidgets.QMessageBox.information(self.controls, "Success",
                                              f"Found and added {found_count} {shape_type}(s).")
        else:
            QtWidgets.QMessageBox.warning(self.controls, "Not Found",
                                          f"No {shape_type}s found with the specified parameters. Try adjusting the thresholds.")

    def run_analysis(self):
        self.canvas.viewer.clear_preview_blobs()  # Clear preview before run

        if 'video_path' not in self.settings or 'pixel_to_mm' not in self.settings:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Please load a video and set the scale first.");
            return
        all_shapes = self.canvas.viewer.get_all_shapes_data()
        self.settings['arenas'] = [s for s in all_shapes if s['category'] == 'arena']
        if not self.settings['arenas']:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Please define at least one arena.");
            return
        self.settings['stimulus_areas'] = [s for s in all_shapes if s['category'] == 'stim']
        self.settings['exp_name'] = self.edit_exp_name.text()
        self.settings['output_dir'] = self.edit_output_dir.text()
        self.settings['start_time_s'] = parse_time_str(self.edit_start_time.text())
        self.settings['end_time_s'] = parse_time_str(self.edit_end_time.text())
        self.settings['min_area'] = self.spin_min_area.value()
        self.settings['max_area'] = self.spin_max_area.value()

        # --- Get tracking settings ---
        self.settings['tracking_method'] = self.combo_track_method.currentText()
        if self.settings['tracking_method'] == 'Blob Detection':
            self.settings['blur_kernel_size'] = self.spin_blur_kernel_blob.value()
            self.settings['blob_min_threshold'] = self.spin_blob_min_thresh.value()
            self.settings['blob_max_threshold'] = self.spin_blob_max_thresh.value()
            # --- Validation ---
            if self.settings['blur_kernel_size'] % 2 == 0:
                QtWidgets.QMessageBox.warning(self.controls, "Input Error", "Blur Kernel Size must be an odd number.")
                return
        else:
            # Set defaults for non-blob case if needed by core
            self.settings['blur_kernel_size'] = 5
            self.settings['blob_min_threshold'] = 50
            self.settings['blob_max_threshold'] = 220

        # --- Get Visualization Settings ---
        self.settings['draw_raw_img'] = self.chk_draw_raw_img.isChecked()
        self.settings['draw_filtered_img'] = self.chk_draw_filtered_img.isChecked()

        self.btn_run.setEnabled(False);
        self.progress_bar.setValue(0)
        self.worker = AnalysisWorker(self.settings)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def on_analysis_finished(self, output, error):
        self.btn_run.setEnabled(True)
        if error:
            import traceback;
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self.controls, "Analysis Error", str(error));
            return

        # --- REMOVED "Success" message ---

        # Save files and show the final confirmation
        output_dir = self.save_output_files(output)
        QtWidgets.QMessageBox.information(self.controls, "Finished",
                                          f"All output files have been saved to:\n{output_dir}")

    def save_output_files(self, output_data):
        base_name = self.settings['exp_name']
        output_dir = self.settings.get('output_dir') or os.path.dirname(self.settings['video_path'])
        output_dir = os.path.join(output_dir, f"{base_name}_output");
        os.makedirs(output_dir, exist_ok=True)

        # --- Save Meta/Info File ---
        meta_file_path = os.path.join(output_dir, f"{base_name}_info.txt")  # Prepend exp name

        # Get engine for frame dimensions
        engine = output_data.get('engine')

        with open(meta_file_path, 'w') as f:
            # --- Write Video Info (FPS, Dimensions, Times) ---
            f.write(f"# Video Info\n")
            if 'fps' in self.settings:
                f.write(f"fps\t{self.settings['fps']:.4f}\n")
            else:
                f.write(f"fps\t30.0000\n")  # Fallback

            if engine:
                f.write(f"frame_width\t{engine.frame_width}\n")
                f.write(f"frame_height\t{engine.frame_height}\n")
            else:
                f.write(f"frame_width\t1920\n")  # Fallback
                f.write(f"frame_height\t1080\n")  # Fallback

            f.write(f"start_time_str\t{self.edit_start_time.text()}\n")
            f.write(f"end_time_str\t{self.edit_end_time.text()}\n\n")

            # --- Write Scale Info ---
            if 'pixel_to_mm' in self.settings:
                f.write(f"# Scale\n")
                f.write(f"pixel_to_mm\t{self.settings['pixel_to_mm']:.8f}\n\n")
            else:
                f.write(f"# Scale\nScale not set\n\n")

            f.write("# Arenas\n")
            f.write("id\tshape\tcategory\tgeom\n")
            arenas = self.settings.get('arenas', [])
            if not arenas:
                f.write("None\n")
            else:
                for shape in arenas: f.write(
                    f"{shape['id']}\t{shape['shape']}\t{shape['category']}\t{str(shape['geom'])}\n")

            f.write("\n# Stimulus Areas\n")
            f.write("id\tshape\tcategory\tgeom\n")
            stims = self.settings.get('stimulus_areas', [])
            if not stims:
                f.write("None\n")
            else:
                for shape in stims: f.write(
                    f"{shape['id']}\t{shape['shape']}\t{shape['category']}\t{str(shape['geom'])}\n")

        # Save Raw Coordinates
        header = "\t".join([f"Animal_{s['id']}_X\tAnimal_{s['id']}_Y" for s in self.settings['arenas']])
        coords_raw_flat = [[item for pos in frame for item in (str(pos[0]), str(pos[1]))] for frame in
                           output_data['raw_coords']]
        np.savetxt(os.path.join(output_dir, f"{base_name}_coordinates_raw.txt"), coords_raw_flat, delimiter='\t',
                   header=header, fmt='%s', comments='')

        # Save Filtered Coordinates
        coords_filtered_flat = [[item for pos in frame for item in (str(pos[0]), str(pos[1]))] for frame in
                                output_data['filtered_coords']]
        np.savetxt(os.path.join(output_dir, f"{base_name}_coordinates_filtered.txt"), coords_filtered_flat,
                   delimiter='\t', header=header, fmt='%s', comments='')

        # --- REMOVED Metrics and Stimulus Analysis File Saving ---

        self.save_trajectory_plot(output_data, output_dir)
        if self.chk_save_video.isChecked(): self.save_track_video(output_data, output_dir)
        return output_dir

    def save_trajectory_plot(self, output_data, output_dir):
        engine = output_data['engine']
        filtered_coords = output_data['filtered_coords']
        raw_coords = output_data['raw_coords']

        cap = cv2.VideoCapture(self.settings['video_path'])
        end_frame_idx = int(parse_time_str(self.edit_end_time.text()) * self.settings['fps'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame_idx)
        ret, last_frame = cap.read();
        cap.release()
        background = last_frame if ret else self.settings['first_frame']

        fig, ax = plt.subplots(figsize=(engine.frame_width / 100, engine.frame_height / 100), dpi=200)
        ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB));
        ax.set_xlim(0, engine.frame_width);
        ax.set_ylim(engine.frame_height, 0);
        ax.set_aspect('equal');
        ax.axis('off')

        from matplotlib.patches import Circle, Rectangle, Polygon
        all_shapes_for_plot = self.settings['arenas'] + self.settings['stimulus_areas']
        for shape in all_shapes_for_plot:
            color = 'cyan' if shape['category'] == 'arena' else 'yellow'
            lw = 1.5
            if shape['shape'] == 'circle':
                ax.add_patch(
                    Circle((shape['geom'][0], shape['geom'][1]), shape['geom'][2], fill=False, ec=color, lw=lw))
            elif shape['shape'] == 'rect':
                ax.add_patch(
                    Rectangle((shape['geom'][0], shape['geom'][1]), shape['geom'][2], shape['geom'][3], fill=False,
                              ec=color, lw=lw))
            elif shape['shape'] == 'poly':
                ax.add_patch(Polygon(np.array(shape['geom']), fill=False, ec=color, lw=lw))
            center = get_shape_center(shape)
            ax.text(center[0], center[1], shape['id'], color='white', ha='center', va='center', fontsize=10,
                    weight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

        arena_id_to_index = {arena['id']: i for i, arena in enumerate(self.settings['arenas'])}
        sorted_arena_ids = sorted(arena_id_to_index.keys(), key=lambda id_str: int(id_str[1:]))
        arena_colors = plt.get_cmap('viridis', len(self.settings['arenas']))  # Updated get_cmap call

        # Read visualization settings
        draw_raw = self.settings.get('draw_raw_img', True)
        draw_filtered = self.settings.get('draw_filtered_img', True)

        for i, arena_id in enumerate(sorted_arena_ids):
            original_index = arena_id_to_index[arena_id]
            arena_color = arena_colors(i)

            # Raw Trajectories
            if draw_raw:
                raw_points = np.array(
                    [p[original_index] for p in raw_coords if p[original_index] and p[original_index][0] != -1])
                if len(raw_points) > 1:
                    ax.plot(raw_points[:, 0], raw_points[:, 1], color=arena_color, linewidth=1.0,
                            alpha=0.4)  # Thicker raw line

            # Filtered Trajectories (Segmented)
            if draw_filtered:
                points_segments = []
                current_segment = []
                all_filtered_points = [p[original_index] for p in filtered_coords]
                for pos in all_filtered_points:
                    if pos and pos[0] != -1:
                        current_segment.append(pos)
                    else:
                        if len(current_segment) > 1: points_segments.append(np.array(current_segment))
                        current_segment = []
                if len(current_segment) > 1: points_segments.append(np.array(current_segment))
                for segment in points_segments: ax.plot(segment[:, 0], segment[:, 1], color=arena_color, linewidth=1.5)

        plt.savefig(os.path.join(output_dir, f"{self.settings['exp_name']}_trajectories.png"), bbox_inches='tight',
                    pad_inches=0);
        plt.close(fig)

    def save_track_video(self, output_data, output_dir):
        engine = output_data['engine']
        raw_coords = output_data['raw_coords']
        filtered_coords = output_data['filtered_coords']

        writer = cv2.VideoWriter(os.path.join(output_dir, f"{self.settings['exp_name']}_track.mp4"),
                                 cv2.VideoWriter_fourcc(*'mp4v'), engine.fps, (engine.frame_width, engine.frame_height))
        cap = cv2.VideoCapture(self.settings['video_path']);
        cap.set(cv2.CAP_PROP_POS_FRAMES, engine.start_frame)

        for frame_idx, (frame_raw, frame_filtered) in enumerate(zip(raw_coords, filtered_coords)):
            ret, frame = cap.read()
            if not ret: break

            for i, (raw_pos, filtered_pos) in enumerate(zip(frame_raw, frame_filtered)):
                animal_id = self.settings['arenas'][i]['id']
                if raw_pos and raw_pos[0] != -1: cv2.circle(frame, (raw_pos[0], raw_pos[1]), 5, (0, 0, 255),
                                                            -1)  # Raw = Red
                if filtered_pos and filtered_pos[0] != -1:
                    cv2.circle(frame, (filtered_pos[0], filtered_pos[1]), 5, (0, 255, 0), -1)  # Filtered = Green
                    cv2.putText(frame, animal_id, (filtered_pos[0] + 10, filtered_pos[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)
        writer.release();
        cap.release()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_controller = MainWindow()
    sys.exit(app.exec())