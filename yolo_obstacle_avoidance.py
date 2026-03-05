#!/usr/bin/env python3
# encoding: utf-8
"""
TurboPi YOLO Obstacle Avoidance — Conflict-Free ROS2 Version
--------------------------------------------------------------
Safely pauses all competing ROS2 nodes before running, then
restores them on exit. Publishes to cmd_vel via mecanum_chassis_node.

Architecture:
    Camera (/dev/video0)
        → OpenCV
        → YOLOv8n (ultralytics 8.2.38 — already installed)
        → /cmd_vel topic
        → mecanum_chassis_node
        → /ros_robot_controller → motors

Run INSIDE the TurboPi Docker container:
    docker exec -it -u ubuntu TurboPi /bin/zsh
    source ~/.zshrc
    python3 yolo_obstacle_avoidance.py

Copy script into container from the Pi:
    docker cp yolo_obstacle_avoidance.py TurboPi:/home/ubuntu/
"""

import cv2
import time
import sys
import threading
import signal

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — tweak these to tune behaviour
# ─────────────────────────────────────────────────────────────
DRIVE_SPEED       = 0.3    # m/s forward
TURN_SPEED        = 0.8    # rad/s angular turn speed
CONF_THRESHOLD    = 0.50   # YOLO confidence cutoff (0–1)
OBSTACLE_CLASSES  = None   # None = all 80 COCO classes
                           # e.g. ['person','chair','bottle','cup']
CENTRE_ZONE       = 0.40   # fraction of frame width = centre danger band
MIN_BOX_AREA_FRAC = 0.04   # ignore boxes smaller than 4% of frame area
CAMERA_INDEX      = 0      # /dev/video0
SHOW_WINDOW       = False  # True only if a display is connected

# Competing nodes to pause on startup and resume on exit.
# Each entry: (service_name, service_type)
#   SetBool → call with data=False to pause, data=True to resume
#   Trigger → call to stop (no resume possible — these are one-shot)
COMPETING_NODES = [
    ('/avoidance_node/set_running',       'SetBool'),
    ('/line_following/set_running',       'SetBool'),
    ('/object_tracking/set_running',      'SetBool'),
    ('/gesture_control_node/set_running', 'SetBool'),
]


# ─────────────────────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────────────────────
def analyse_detections(results, frame_w, frame_h, model):
    """
    Finds the largest close-enough obstacle and returns where it is.
    Returns: (direction, centre_x, label)
      direction ∈ {'clear', 'obstacle_left', 'obstacle_right', 'obstacle_centre'}
    """
    frame_area   = frame_w * frame_h
    cx_min = frame_w * (0.5 - CENTRE_ZONE / 2)
    cx_max = frame_w * (0.5 + CENTRE_ZONE / 2)

    best_area, best_cx, best_label = 0, None, None

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            if OBSTACLE_CLASSES and label not in OBSTACLE_CLASSES:
                continue
            if float(box.conf[0]) < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area / frame_area < MIN_BOX_AREA_FRAC:
                continue
            if area > best_area:
                best_area  = area
                best_cx    = (x1 + x2) / 2
                best_label = label

    if best_cx is None:
        return 'clear', None, None
    if best_cx < cx_min:
        return 'obstacle_left',   best_cx, best_label
    if best_cx > cx_max:
        return 'obstacle_right',  best_cx, best_label
    return 'obstacle_centre', best_cx, best_label


def draw_overlay(frame, results, model, direction, frame_w, frame_h):
    lx = int(frame_w * (0.5 - CENTRE_ZONE / 2))
    rx = int(frame_w * (0.5 + CENTRE_ZONE / 2))
    cv2.line(frame, (lx, 0), (lx, frame_h), (0, 200, 255), 1)
    cv2.line(frame, (rx, 0), (rx, frame_h), (0, 200, 255), 1)

    for result in results:
        for box in result.boxes:
            if float(box.conf[0]) < CONF_THRESHOLD:
                continue
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) if direction != 'clear' else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {float(box.conf[0]):.2f}",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    status_map = {
        'clear':           ('CLEAR – moving forward',        (0, 200,   0)),
        'obstacle_left':   ('OBSTACLE LEFT – turning right', (0, 165, 255)),
        'obstacle_right':  ('OBSTACLE RIGHT – turning left', (0, 165, 255)),
        'obstacle_centre': ('OBSTACLE AHEAD – stopping',     (0,   0, 255)),
    }
    text, colour = status_map.get(direction, ('UNKNOWN', (128, 128, 128)))
    cv2.rectangle(frame, (0, 0), (frame_w, 34), (20, 20, 20), -1)
    cv2.putText(frame, text, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return frame


# ─────────────────────────────────────────────────────────────
# ROS2 NODE
# ─────────────────────────────────────────────────────────────
class YoloAvoidanceNode(Node):

    def __init__(self):
        super().__init__('yolo_avoidance_node')

        # Motor output — same topic mecanum_chassis_node subscribes to
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)

        # ── Step 1: pause all competing nodes ────────────────
        self._pause_competing_nodes()

        # ── Step 2: load YOLO ─────────────────────────────────
        self.get_logger().info('Loading YOLOv8n model…')
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info('YOLOv8n ready.')

        # ── Step 3: open camera ───────────────────────────────
        # The ROS2 /usb_cam node already uses /dev/video0.
        # We open /dev/video1 as a second handle to avoid blocking it.
        # If video1 fails we fall back to video0.
        self.cap = self._open_camera()

        self.prev_direction = None
        self.running = True

        # Detection runs in a background thread
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

        self.get_logger().info('YOLO avoidance running. Ctrl+C to stop.')

    # ── Camera helper ─────────────────────────────────────────
    def _open_camera(self):
        for idx in [1, 0]:   # prefer video1 so we don't clash with /usb_cam
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.get_logger().info(f'Camera opened on /dev/video{idx}')
                return cap
        self.get_logger().error(
            'No camera found on video0 or video1. '
            'Check that the camera is plugged in and passed through to Docker.'
        )
        raise RuntimeError('Camera not available')

    # ── Pause / resume competing nodes ───────────────────────
    def _call_set_running(self, service_name, enable: bool, timeout=2.0):
        """Call a SetBool service. Returns True on success."""
        client = self.create_client(SetBool, service_name)
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(
                f'Service {service_name} not available — skipping.'
            )
            return False
        req = SetBool.Request()
        req.data = enable
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is not None:
            state = 'RESUMED' if enable else 'PAUSED'
            self.get_logger().info(f'{service_name} → {state}')
            return True
        self.get_logger().warn(f'No response from {service_name}')
        return False

    def _pause_competing_nodes(self):
        self.get_logger().info('Pausing competing nodes…')
        for svc, svc_type in COMPETING_NODES:
            if svc_type == 'SetBool':
                self._call_set_running(svc, enable=False)
        # Give nodes a moment to stop publishing
        time.sleep(0.5)
        # Send a zero-velocity to clear any in-flight commands
        self.stop()
        self.get_logger().info('All competing nodes paused. Starting YOLO.')

    def _resume_competing_nodes(self):
        self.get_logger().info('Restoring competing nodes…')
        for svc, svc_type in COMPETING_NODES:
            if svc_type == 'SetBool':
                self._call_set_running(svc, enable=True)

    # ── cmd_vel helpers ───────────────────────────────────────
    def _publish(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x  = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def move_forward(self): self._publish(linear_x=DRIVE_SPEED,  angular_z=0.0)
    def turn_right(self):   self._publish(linear_x=0.0, angular_z=-TURN_SPEED)
    def turn_left(self):    self._publish(linear_x=0.0, angular_z= TURN_SPEED)
    def stop(self):         self._publish(linear_x=0.0, angular_z=0.0)

    # ── Detection loop (background thread) ────────────────────
    def detection_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Frame grab failed, retrying…')
                time.sleep(0.05)
                continue

            frame_h, frame_w = frame.shape[:2]
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)

            direction, best_cx, best_label = analyse_detections(
                results, frame_w, frame_h, self.model
            )

            # Only publish on state change to avoid flooding cmd_vel
            if direction != self.prev_direction:
                if direction == 'clear':
                    self.get_logger().info('Path CLEAR → FORWARD')
                    self.move_forward()
                elif direction == 'obstacle_left':
                    self.get_logger().info(f'Obstacle LEFT  ({best_label}) → TURN RIGHT')
                    self.turn_right()
                elif direction == 'obstacle_right':
                    self.get_logger().info(f'Obstacle RIGHT ({best_label}) → TURN LEFT')
                    self.turn_left()
                elif direction == 'obstacle_centre':
                    self.get_logger().info(f'Obstacle AHEAD ({best_label}) → STOP')
                    self.stop()
                self.prev_direction = direction

            if SHOW_WINDOW:
                frame = draw_overlay(
                    frame, results, self.model, direction, frame_w, frame_h
                )
                cv2.imshow('TurboPi YOLO Avoidance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

    # ── Cleanup ───────────────────────────────────────────────
    def destroy_node(self):
        self.get_logger().info('Shutting down…')
        self.running = False
        self.stop()                      # zero velocity before exit
        time.sleep(0.2)
        self._resume_competing_nodes()   # restore all paused nodes
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = YoloAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print('Shutdown complete.')

if __name__ == '__main__':
    main()
