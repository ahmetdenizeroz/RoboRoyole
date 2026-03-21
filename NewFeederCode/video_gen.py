import cv2
import numpy as np
import random
import math

# ---------- PARAMETERS ----------
W, H = 1920, 1080
FPS = 30
FRAMES = 900
NUM_OBJECTS = 4

# OpenCV uses BGR
BG_COLOR = (0, 0, 255)          # red background
BODY_COLOR = (255, 0, 0)        # blue oval in BGR
HEAD_COLOR = (255, 0, 0)        # blue head

BODY_MAJOR = 58
BODY_MINOR = 36

MARKER_SIZE = 72
MARKER_PAD = 8                  # white margin around marker
PLATE_SIZE = MARKER_SIZE + 2 * MARKER_PAD

MAX_SPEED = 3.0
MIN_SPEED = 1.4

TURN_GAIN = 0.12
MAX_TURN = 0.14
WAYPOINT_REACH = 40.0

SWAY_AMPL = 4.0
SWAY_FREQ = 0.23

OUTSIDE_MARGIN = 180
INSIDE_MARGIN = 130

OUTPUT_FILE = "aruco_objects_red_bg_blue_body_white_plate.mp4"

# ---------- ARUCO ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

# ---------- VIDEO WRITER ----------
out = cv2.VideoWriter(
    OUTPUT_FILE,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W, H)
)

# ---------- HELPERS ----------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def angle_to(vec):
    return math.atan2(vec[1], vec[0])

def random_inside_point():
    return np.array([
        random.uniform(INSIDE_MARGIN, W - INSIDE_MARGIN),
        random.uniform(INSIDE_MARGIN, H - INSIDE_MARGIN)
    ], dtype=np.float64)

def spawn_and_entry_for_side(side):
    if side == "left":
        spawn = np.array([
            -OUTSIDE_MARGIN,
            random.uniform(0.18 * H, 0.82 * H)
        ], dtype=np.float64)
        entry = np.array([
            random.uniform(80, 180),
            random.uniform(0.18 * H, 0.82 * H)
        ], dtype=np.float64)

    elif side == "right":
        spawn = np.array([
            W + OUTSIDE_MARGIN,
            random.uniform(0.18 * H, 0.82 * H)
        ], dtype=np.float64)
        entry = np.array([
            random.uniform(W - 180, W - 80),
            random.uniform(0.18 * H, 0.82 * H)
        ], dtype=np.float64)

    elif side == "top":
        spawn = np.array([
            random.uniform(0.18 * W, 0.82 * W),
            -OUTSIDE_MARGIN
        ], dtype=np.float64)
        entry = np.array([
            random.uniform(0.18 * W, 0.82 * W),
            random.uniform(80, 180)
        ], dtype=np.float64)

    else:  # bottom
        spawn = np.array([
            random.uniform(0.18 * W, 0.82 * W),
            H + OUTSIDE_MARGIN
        ], dtype=np.float64)
        entry = np.array([
            random.uniform(0.18 * W, 0.82 * W),
            random.uniform(H - 180, H - 80)
        ], dtype=np.float64)

    heading = angle_to(entry - spawn)
    return spawn, entry, heading

# ---------- WALKER ----------
class Walker:
    def __init__(self, obj_id, side, meeting_point):
        self.id = obj_id
        self.side = side

        spawn, entry, heading = spawn_and_entry_for_side(side)
        self.pos = spawn.astype(np.float64)
        self.heading = heading

        self.speed = random.uniform(1.8, 2.5)
        self.phase = random.uniform(0, 2 * math.pi)

        meet_jitter = np.array([
            random.uniform(-40, 40),
            random.uniform(-40, 40)
        ], dtype=np.float64)

        meet_local = meeting_point + meet_jitter
        meet_local[0] = clamp(meet_local[0], INSIDE_MARGIN, W - INSIDE_MARGIN)
        meet_local[1] = clamp(meet_local[1], INSIDE_MARGIN, H - INSIDE_MARGIN)

        roam1 = random_inside_point()
        roam2 = random_inside_point()

        self.waypoints = [entry, meet_local, roam1, roam2]
        self.current_wp = 0

    def update(self, others):
        target = self.waypoints[self.current_wp]
        to_target = target - self.pos
        dist_target = np.linalg.norm(to_target)

        if dist_target < WAYPOINT_REACH and self.current_wp < len(self.waypoints) - 1:
            self.current_wp += 1
            target = self.waypoints[self.current_wp]
            to_target = target - self.pos

        desired_heading = angle_to(to_target)
        heading_error = wrap_angle(desired_heading - self.heading)
        turn = clamp(TURN_GAIN * heading_error, -MAX_TURN, MAX_TURN)
        turn += random.uniform(-0.02, 0.02)

        repel_turn = 0.0
        speed_adjust = 0.0

        for other in others:
            if other is self:
                continue

            delta = other.pos - self.pos
            d = np.linalg.norm(delta)
            if d < 1e-6:
                continue

            rel_ang = wrap_angle(angle_to(delta) - self.heading)

            if d < 95:
                repel_turn -= 0.05 * np.sign(rel_ang)
                speed_adjust -= 0.08
            elif d < 150:
                repel_turn -= 0.02 * np.sign(rel_ang)

        self.heading = wrap_angle(self.heading + turn + repel_turn)

        self.speed += random.uniform(-0.04, 0.04) + speed_adjust
        self.speed = clamp(self.speed, MIN_SPEED, MAX_SPEED)

        vel = np.array([
            math.cos(self.heading) * self.speed,
            math.sin(self.heading) * self.speed
        ], dtype=np.float64)

        self.pos += vel
        self.phase += SWAY_FREQ + random.uniform(-0.015, 0.015)

    def draw(self, frame, marker_img):
        sway = SWAY_AMPL * math.sin(self.phase)
        perp = np.array([
            -math.sin(self.heading),
             math.cos(self.heading)
        ], dtype=np.float64)

        center = self.pos + sway * perp
        cx = int(round(center[0]))
        cy = int(round(center[1]))
        angle_deg = -math.degrees(self.heading)

        # Body
        cv2.ellipse(
            frame,
            (cx, cy),
            (BODY_MAJOR, BODY_MINOR),
            angle_deg,
            0,
            360,
            BODY_COLOR,
            -1
        )

        # Head
        head_offset = np.array([
            math.cos(self.heading) * BODY_MAJOR * 0.72,
            math.sin(self.heading) * BODY_MAJOR * 0.72
        ], dtype=np.float64)

        hx = int(round(center[0] + head_offset[0]))
        hy = int(round(center[1] + head_offset[1]))
        cv2.circle(frame, (hx, hy), 10, HEAD_COLOR, -1)

        # White square plate under the marker
        plate = np.full((PLATE_SIZE, PLATE_SIZE, 3), 255, dtype=np.uint8)

        # Marker itself
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        # Put marker in the middle of the white plate
        plate[MARKER_PAD:MARKER_PAD + MARKER_SIZE,
              MARKER_PAD:MARKER_PAD + MARKER_SIZE] = marker_bgr

        # Rotate the whole white plate + marker together
        rot_mat = cv2.getRotationMatrix2D(
            (PLATE_SIZE / 2, PLATE_SIZE / 2),
            angle_deg,
            1.0
        )
        rotated_plate = cv2.warpAffine(
            plate,
            rot_mat,
            (PLATE_SIZE, PLATE_SIZE),
            flags=cv2.INTER_NEAREST,
            borderValue=BG_COLOR
        )

        x1 = cx - PLATE_SIZE // 2
        y1 = cy - PLATE_SIZE // 2
        x2 = x1 + PLATE_SIZE
        y2 = y1 + PLATE_SIZE

        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return

        roi = frame[y1:y2, x1:x2]

        # Paste everything that is not background-red from rotated plate
        bg = np.array(BG_COLOR, dtype=np.uint8)
        mask = np.any(rotated_plate != bg, axis=2)
        roi[mask] = rotated_plate[mask]

# ---------- BUILD SCENE ----------
sides = ["left", "right", "top", "bottom"]
random.shuffle(sides)

meeting_point = np.array([
    random.uniform(0.30 * W, 0.70 * W),
    random.uniform(0.30 * H, 0.70 * H)
], dtype=np.float64)

markers = []
for i in range(NUM_OBJECTS):
    marker = cv2.aruco.generateImageMarker(aruco_dict, i, MARKER_SIZE)
    markers.append(marker)

objects = [Walker(i, sides[i], meeting_point) for i in range(NUM_OBJECTS)]

# ---------- MAIN LOOP ----------
for frame_idx in range(FRAMES):
    frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

    for obj in objects:
        obj.update(objects)

    for obj in sorted(objects, key=lambda o: o.pos[1]):
        obj.draw(frame, markers[obj.id])

    out.write(frame)

out.release()
print(f"Saved as {OUTPUT_FILE}")