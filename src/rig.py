# src/rig.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import cv2
import mediapipe as mp


# Public list of supported attachment names (pose-only for now)
ATTACH_POINTS: List[str] = [
    "none",
    "shoulders_mid",
    "chest",
    "torso",
    "hips_mid",
    "head",          # uses NOSE
    "face_center",   # uses midpoint between LEFT_EYE and RIGHT_EYE (more stable for masks)
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",     # wrist
    "right_hand",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",     # ankle
    "right_foot",
]


@dataclass
class RigState:
    # smoothed pose state
    attach_xy: np.ndarray
    angle_deg: float
    ref_dist: float  # used for scaling


class PoseRig:
    def __init__(self):
        self.last_angle_raw: Optional[float] = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last: Optional[RigState] = None

    def close(self):
        self.pose.close()

    def _pt(self, lm, name: str, w: int, h: int) -> np.ndarray:
        i = getattr(self.mp_pose.PoseLandmark, name).value
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

    def estimate(
        self,
        frame_bgr: np.ndarray,
        attach: str,
        rotation: str,
        smoothing: float,
        reuse_last_on_miss: bool,
    ) -> Optional[RigState]:
        """
        Returns smoothed RigState or None if no pose and reuse disabled.
        """
        if attach == "none":
            return None

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        res = self.pose.process(rgb)
        have_pose = res.pose_landmarks is not None

        if not have_pose:
            if reuse_last_on_miss and self.last is not None:
                return self.last
            return None

        lm = res.pose_landmarks.landmark

        Ls = self._pt(lm, "LEFT_SHOULDER", w, h)
        Rs = self._pt(lm, "RIGHT_SHOULDER", w, h)
        if Rs[0] < Ls[0]:
            Ls, Rs = Rs, Ls
        Lh = self._pt(lm, "LEFT_HIP", w, h)
        Rh = self._pt(lm, "RIGHT_HIP", w, h)
        mid_sh = (Ls + Rs) * 0.5
        mid_hp = (Lh + Rh) * 0.5
        shoulder_vec = (Rs - Ls)
        shoulder_dist = float(np.linalg.norm(shoulder_vec) + 1e-6)

        # --- attachment point ---
        if attach == "shoulders_mid":
            attach_xy = mid_sh

        elif attach == "chest":
            # slightly below shoulders (better for wings / backpacks)
            attach_xy = mid_sh + np.array([0.0, 0.12 * shoulder_dist], dtype=np.float32)

        elif attach == "torso":
            attach_xy = mid_sh * 0.75 + mid_hp * 0.25

        elif attach == "hips_mid":
            attach_xy = mid_hp

        elif attach == "head":
            # uses NOSE (can drift when leaning)
            attach_xy = self._pt(lm, "NOSE", w, h)

        elif attach == "face_center":
            Le = self._pt(lm, "LEFT_EYE", w, h)
            Re = self._pt(lm, "RIGHT_EYE", w, h)
            attach_xy = (Le + Re) * 0.5

        elif attach == "left_shoulder":
            attach_xy = Ls

        elif attach == "right_shoulder":
            attach_xy = Rs

        elif attach == "left_elbow":
            attach_xy = self._pt(lm, "LEFT_ELBOW", w, h)

        elif attach == "right_elbow":
            attach_xy = self._pt(lm, "RIGHT_ELBOW", w, h)

        elif attach == "left_hand":
            attach_xy = self._pt(lm, "LEFT_WRIST", w, h)

        elif attach == "right_hand":
            attach_xy = self._pt(lm, "RIGHT_WRIST", w, h)

        elif attach == "left_hip":
            attach_xy = Lh

        elif attach == "right_hip":
            attach_xy = Rh

        elif attach == "left_knee":
            attach_xy = self._pt(lm, "LEFT_KNEE", w, h)

        elif attach == "right_knee":
            attach_xy = self._pt(lm, "RIGHT_KNEE", w, h)

        elif attach == "left_foot":
            attach_xy = self._pt(lm, "LEFT_ANKLE", w, h)

        elif attach == "right_foot":
            attach_xy = self._pt(lm, "RIGHT_ANKLE", w, h)

        else:
            raise ValueError(f"Unknown attach point: {attach}. Allowed: {ATTACH_POINTS}")

        # --- rotation ---
        angle = 0.0
        if rotation == "shoulders":
            angle = float(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))

            # unwrap to stay close to previous raw angle
            if self.last_angle_raw is not None:
                while angle - self.last_angle_raw > 180:
                    angle -= 360
                while angle - self.last_angle_raw < -180:
                    angle += 360
            self.last_angle_raw = angle

        # --- smoothing ---
        if self.last is None:
            sm = RigState(attach_xy=attach_xy, angle_deg=angle, ref_dist=shoulder_dist)
        else:
            t = 1.0 - float(smoothing)
            sm_xy = self.last.attach_xy * (1.0 - t) + attach_xy * t
            sm_ang = self.last.angle_deg * (1.0 - t) + angle * t
            sm_ref = self.last.ref_dist * (1.0 - t) + shoulder_dist * t
            sm = RigState(attach_xy=sm_xy, angle_deg=sm_ang, ref_dist=float(sm_ref))

        self.last = sm
        return sm
