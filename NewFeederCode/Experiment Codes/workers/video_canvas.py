import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import (
    QImage, QPixmap, QPainter,
    QMouseEvent, QWheelEvent, QKeyEvent
)

# --- Video Canvas ---
# A custom QLabel for displaying video and drawing the zone
class VideoCanvas(QLabel):
    # Emits x, y, radius
    zone_selected = Signal(int, int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1A1A1A; color: #f6e000; border: 1px solid #555;")
        self.setText("Camera Disconnected")

        self.img_size = (0, 0)  # w, h of the actual video frame
        self.pixmap_size = (0, 0)  # w, h of the scaled pixmap
        self.pixmap_offset = (0, 0)  # x, y offset of the pixmap

        # --- Zone definition ---
        self.zone_center = None  # (x, y) in image coordinates
        self.zone_radius = 0

        # --- Drawing state ---
        self.is_drawing = False
        self.is_dragging = False
        self.start_point = None  # (x, y) in image coordinates
        self.drag_offset = None  # (x, y) offset from circle center

        # Allow this widget to receive key presses (for 'Del')
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame(self, rgb_frame: np.ndarray):
        """Update the label with a new video frame."""
        if rgb_frame is None or rgb_frame.size == 0:
            self.clear_frame()
            return

        h, w, ch = rgb_frame.shape
        self.img_size = (w, h)
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale the pixmap to fit the label while maintaining aspect ratio
        self.q_pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = self.q_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.pixmap_size = (scaled_pixmap.width(), scaled_pixmap.height())

        # Calculate offsets for centering the pixmap
        self.pixmap_offset = (
            (self.width() - self.pixmap_size[0]) // 2,
            (self.height() - self.pixmap_size[1]) // 2
        )

        # We need a new pixmap to draw on
        canvas_pixmap = QPixmap(self.size())
        canvas_pixmap.fill(Qt.black)  # Fill background

        painter = QPainter(canvas_pixmap)
        # Draw the video frame
        painter.drawPixmap(self.pixmap_offset[0], self.pixmap_offset[1], scaled_pixmap)

        # --- "TWO CIRCLES" BUG FIX ---
        # The circle is now drawn by feeder_core.py *before* this function.
        # We no longer draw a second circle here.

        painter.end()
        self.setPixmap(canvas_pixmap)

    def clear_frame(self):
        """Show a blank screen when disconnected."""
        self.img_size = (0, 0)
        blank_pixmap = QPixmap(self.size())
        blank_pixmap.fill(Qt.black)
        self.setPixmap(blank_pixmap)
        self.setText("Camera Disconnected")

    def _pixmap_to_image(self, p_pos):
        """Convert QPoint from pixmap coordinates to image coordinates."""
        if self.img_size == (0, 0) or self.pixmap_size == (0, 0) or self.pixmap_size[0] == 0 or self.pixmap_size[
            1] == 0:
            return None

        # Adjust for pixmap offset
        px = p_pos.x() - self.pixmap_offset[0]
        py = p_pos.y() - self.pixmap_offset[1]

        # Scale to image coordinates
        scale_x = self.img_size[0] / self.pixmap_size[0]
        scale_y = self.img_size[1] / self.pixmap_size[1]

        img_x = int(px * scale_x)
        img_y = int(py * scale_y)

        # Clamp to image boundaries
        img_x = max(0, min(self.img_size[0] - 1, img_x))
        img_y = max(0, min(self.img_size[1] - 1, img_y))

        return (img_x, img_y)

    def _image_to_pixmap(self, i_pos):
        """Convert image coordinates to pixmap coordinates."""
        if self.img_size == (0, 0) or self.pixmap_size == (0, 0) or self.img_size[0] == 0 or self.img_size[1] == 0:
            return (0, 0)

        scale_x = self.pixmap_size[0] / self.img_size[0]
        scale_y = self.pixmap_size[1] / self.img_size[1]

        px = int(i_pos[0] * scale_x) + self.pixmap_offset[0]
        py = int(i_pos[1] * scale_y) + self.pixmap_offset[1]

        return (px, py)

    def _is_inside_zone(self, img_pos):
        """Check if a point (in image coords) is inside the existing zone."""
        if not self.zone_center or self.zone_radius == 0:
            return False

        dist = np.sqrt((img_pos[0] - self.zone_center[0]) ** 2 + (img_pos[1] - self.zone_center[1]) ** 2)
        return dist < self.zone_radius

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton or self.img_size == (0, 0):
            return

        img_pos = self._pixmap_to_image(ev.position().toPoint())
        if not img_pos:
            return

        if self._is_inside_zone(img_pos):
            # Start dragging the existing circle
            self.is_dragging = True
            self.drag_offset = (img_pos[0] - self.zone_center[0], img_pos[1] - self.zone_center[1])
        else:
            # Start drawing a new circle
            self.is_drawing = True
            self.start_point = img_pos
            self.zone_center = img_pos
            self.zone_radius = 0

        self.update()  # Force repaint

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.img_size == (0, 0):
            return

        img_pos = self._pixmap_to_image(ev.position().toPoint())
        if not img_pos:
            return

        if self.is_dragging:
            # Move the circle center
            self.zone_center = (img_pos[0] - self.drag_offset[0], img_pos[1] - self.drag_offset[1])
            self.zone_selected.emit(self.zone_center[0], self.zone_center[1], self.zone_radius)

        elif self.is_drawing and self.start_point:
            # Calculate new radius
            r = np.sqrt((img_pos[0] - self.start_point[0]) ** 2 + (img_pos[1] - self.start_point[1]) ** 2)
            self.zone_radius = int(r)
            self.zone_center = self.start_point
            self.zone_selected.emit(self.zone_center[0], self.zone_center[1], self.zone_radius)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton:
            return

        if self.is_dragging:
            self.is_dragging = False
            self.drag_offset = None

        elif self.is_drawing:
            self.is_drawing = False
            self.start_point = None

    def wheelEvent(self, ev: QWheelEvent):
        """Handle resizing the circle with the scroll wheel."""
        if self.img_size == (0, 0) or not self.zone_center:
            return

        img_pos = self._pixmap_to_image(ev.position().toPoint())
        if not img_pos or not self._is_inside_zone(img_pos):
            return  # Only resize if mouse is inside the circle

        # Determine scroll direction
        delta = ev.angleDelta().y()
        if delta > 0:
            self.zone_radius = max(5, self.zone_radius + 5)  # Increase radius
        elif delta < 0:
            self.zone_radius = max(5, self.zone_radius - 5)  # Decrease radius

        self.zone_selected.emit(self.zone_center[0], self.zone_center[1], self.zone_radius)

    def keyPressEvent(self, ev: QKeyEvent):
        """Handle deleting the circle with the 'Del' key."""
        if ev.key() == Qt.Key_Delete:
            self.zone_center = None
            self.zone_radius = 0
            self.zone_selected.emit(0, 0, 0)  # Emit a cleared zone

    @Slot(int, int, int)
    def update_zone_visual(self, x, y, r):
        """Slot to update the visual from the core (e.g., if set by another user)."""
        self.zone_center = (x, y)
        self.zone_radius = r