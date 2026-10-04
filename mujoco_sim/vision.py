"""Overhead RGB-D camera: PCB pose estimation and operator detection.

The camera sits 1950 mm above the bench centre (brief, Part A) looking
straight down. Everything here works only from the rendered images and the
camera's calibration (intrinsics + the camera-to-origin transform of
Part A.5) — no simulator ground truth.

  PCB    colour segmentation of the blue board inside a region of interest
         around Position 4, back-projection of the pixels onto the PCB's top
         plane (known height, Part A), principal-axis orientation, and a white
         fiducial to resolve the board's 180 deg symmetry.
  person skin / sleeve segmentation, 3D points from the depth image, and the
         distance from those points to the UR10e's links (from the robot's own
         joint positions), for speed and separation monitoring.

Usage:
    python vision.py        # camera calibration, Part A.5 transforms, a test detection
"""

from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_scene import (CAM_FOVY, CAM_LENS_DROP, FIDUCIAL, LAYOUT, MM, PADS)

W_FULL, H_FULL = 1920, 1080      # PCB measurement resolution
W_HAND, H_HAND = 960, 540        # operator monitoring resolution (2x binned)
PCB_TOP_Z = (LAYOUT["pcb_height"] + LAYOUT["pcb"][2]) * MM
ROI = 0.15                       # m, half-size of the search window around Pos4
EXPOSURE = 0.6                   # camera exposure: scales the scene lights while rendering


class Camera:
    """Pinhole model in the OpenCV convention (x right, y down, z forward)."""

    def __init__(self, width, height):
        self.W, self.H = width, height
        self.f = (height / 2) / np.tan(np.deg2rad(CAM_FOVY) / 2)
        self.cx, self.cy = width / 2, height / 2
        c = LAYOUT["camera"]
        self.C = np.array([c[0] * MM, c[1] * MM, c[2] * MM - CAM_LENS_DROP])
        # columns: camera x = +Y_world, camera y = +X_world, camera z = -Z_world
        self.R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])

    @property
    def K(self):
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]])

    @property
    def T_world_cam(self):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.C
        return T

    def project(self, p_world):
        pc = self.R.T @ (np.asarray(p_world) - self.C)
        return np.array([self.f * pc[0] / pc[2] + self.cx, self.f * pc[1] / pc[2] + self.cy])

    def rays(self, u, v):
        d = np.column_stack([(u - self.cx) / self.f, (v - self.cy) / self.f, np.ones_like(u)])
        return d @ self.R.T                                      # world directions, z_c = 1

    def backproject_plane(self, u, v, z):
        d = self.rays(u, v)
        t = (z - self.C[2]) / d[:, 2]
        return self.C + t[:, None] * d

    def backproject_depth(self, u, v, depth):
        return self.C + depth[:, None] * self.rays(u, v)


class Overhead:
    """Renders the overhead camera and runs the detectors."""

    def __init__(self, m):
        self.m = m
        self.cam_id = m.camera("overhead").id
        self.full, self.small = Camera(W_FULL, H_FULL), Camera(W_HAND, H_HAND)
        self.r_full = mujoco.Renderer(m, H_FULL, W_FULL)
        self.r_small = mujoco.Renderer(m, H_HAND, W_HAND)
        self.r_depth = mujoco.Renderer(m, H_HAND, W_HAND)
        self.r_depth.enable_depth_rendering()
        self.rng = np.random.default_rng(7)
        self.last_person = None          # (time, distance) for the fail-safe
        self.background = None           # reference frame of the cell without a person

    def _render(self, r, d):
        """Render with the camera's exposure: dim the lights, render, restore."""
        hl = self.m.vis.headlight
        saved = hl.ambient.copy(), hl.diffuse.copy(), self.m.light_diffuse.copy()
        hl.ambient[:], hl.diffuse[:] = saved[0] * EXPOSURE, saved[1] * EXPOSURE
        self.m.light_diffuse[:] = saved[2] * EXPOSURE
        r.update_scene(d, camera=self.cam_id)
        img = r.render()
        hl.ambient[:], hl.diffuse[:] = saved[0], saved[1]
        self.m.light_diffuse[:] = saved[2]
        return img

    # ── PCB ─────────────────────────────────────────────────────────────────
    def measure_pcb(self, d):
        rgb = self._render(self.r_full, d)
        return estimate_pcb_pose(rgb, self.full), rgb

    # ── person ──────────────────────────────────────────────────────────────
    def monitor(self, d, t, link_pts):
        rgb = self._render(self.r_small, d)
        if self.background is None:      # first frame: the cell is clear of people
            self.background = rgb.astype(np.int16)
        self.r_depth.update_scene(d, camera=self.cam_id)
        depth = self.r_depth.render()
        depth = depth + self.rng.normal(0, 0.003, depth.shape)        # 3 mm depth noise
        changed = np.abs(rgb.astype(np.int16) - self.background).max(axis=2) > 40
        mask = person_mask(rgb) & changed
        dist = np.inf
        pts = None
        if mask.sum() >= 15:
            v, u = np.nonzero(mask)
            pts = self.small.backproject_depth(u + 0.5, v + 0.5, depth[v, u])
            dist = distance_to_robot(pts, link_pts)
            self.last_person = (t, dist)
        elif self.last_person is not None and t - self.last_person[0] < 2.0 \
                and self.last_person[1] < 0.4:
            # Lost from view while close (e.g. under the robot's arm): assume the
            # worst case and report contact, so the robot stops, until the
            # person is seen again or 2 s pass without any sighting.
            dist = 0.0
        return dist, rgb, mask, pts


def _hsv(rgb):
    f = rgb.astype(np.float32) / 255
    mx, mn = f.max(axis=2), f.min(axis=2)
    return f[..., 0], f[..., 1], f[..., 2], mx, (mx - mn) / np.maximum(mx, 1e-6)


def pcb_mask(rgb):
    """Saturated blue: the board (the robots' light-blue caps are far less saturated)."""
    r, g, b, mx, sat = _hsv(rgb)
    return (b == mx) & (sat > 0.45) & (b - r > 0.32) & (b > 0.5)


def white_mask(rgb):
    r, g, b, mx, sat = _hsv(rgb)
    return (sat < 0.1) & (rgb.min(axis=2) > 0.72 * 255)


def person_mask(rgb):
    """Skin and the operator's hi-vis orange sleeve (combined with background
    subtraction in Overhead.monitor, so static objects never count)."""
    r, g, b, mx, sat = _hsv(rgb)
    skin = (r == mx) & (r > 0.5) & (r - b > 0.15) & (sat > 0.15) & (sat < 0.45)
    gr = g / np.maximum(r, 1e-6)
    hivis = (r == mx) & (sat > 0.7) & (gr > 0.3) & (gr < 0.62) & (r > 0.5)   # orange, not red
    return skin | hivis


def estimate_pcb_pose(rgb, cam):
    """PCB centre (x, y, z) and yaw (rad, +u axis) from the overhead image."""
    pos4 = np.array([LAYOUT["pos4"][0] * MM, LAYOUT["pos4"][1] * MM, PCB_TOP_Z])
    (u0, v0) = cam.project(pos4 - [ROI, ROI, 0])
    (u1, v1) = cam.project(pos4 + [ROI, ROI, 0])
    ul, uh = int(min(u0, u1)), int(max(u0, u1)) + 1
    vl, vh = int(min(v0, v1)), int(max(v0, v1)) + 1
    roi = rgb[vl:vh, ul:uh]
    vv, uu = np.nonzero(pcb_mask(roi))
    if len(uu) < 200:
        return None
    blue = cam.backproject_plane(uu + ul + 0.5, vv + vl + 0.5, PCB_TOP_Z)[:, :2]
    # first pass on the blue pixels: centre and axes of the board
    c = blue.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((blue - c).T))
    axis = evecs[:, np.argmax(evals)]                   # long (85 mm) side
    normal = np.array([-axis[1], axis[0]])
    # the fiducial: white pixels inside the board's outline (it is a hole in
    # the blue mask). Add them back and refit.
    wv, wu = np.nonzero(white_mask(roi))
    wpts = cam.backproject_plane(wu + ul + 0.5, wv + vl + 0.5, PCB_TOP_Z)[:, :2]
    hl, hw = LAYOUT["pcb"][0] / 2 * MM - 0.002, LAYOUT["pcb"][1] / 2 * MM - 0.002
    rel = wpts - c
    inside = (np.abs(rel @ axis) < hl) & (np.abs(rel @ normal) < hw)
    if inside.sum() < 4:
        return None
    pts = np.vstack([blue, wpts[inside]])
    c = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
    axis = evecs[:, np.argmax(evals)]
    fid = wpts[inside].mean(axis=0)
    uu = np.r_[uu, wu[inside]]
    if np.dot(fid - c, axis) < 0:                       # fiducial sits at +u
        axis = -axis
    yaw = np.arctan2(axis[1], axis[0])
    normal = np.array([-axis[1], axis[0]])
    mirrored = np.dot(fid - c, normal) < 0              # fiducial should be at +v
    return dict(pos=np.array([c[0], c[1], PCB_TOP_Z - LAYOUT["pcb"][2] * MM / 2]), yaw=yaw,
                quat=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), n_pixels=len(uu),
                fiducial=fid, mirrored=bool(mirrored), roi=(ul, vl, uh, vh))


def pads_from_pose(pose):
    c, yaw = pose["pos"], pose["yaw"]
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return np.array([[*(c[:2] + R @ (np.array([u, v]) * MM)), PCB_TOP_Z] for u, v in PADS])


def ur10e_link_points(m, d):
    """Joint centres of the UR10e plus the solder tip, from its own kinematics."""
    names = ["ur10e_base", "ur10e_shoulder_link", "ur10e_upper_arm_link", "ur10e_forearm_link",
             "ur10e_wrist_1_link", "ur10e_wrist_2_link", "ur10e_wrist_3_link"]
    pts = [d.xpos[m.body(n).id] for n in names] + [d.site_xpos[m.site("ur10e_tcp").id]]
    return np.array(pts)


def distance_to_robot(points, link_pts, radius=0.065):
    """Minimum distance from 3D points to the robot, modelled as capsules
    between consecutive joint centres."""
    a, b = link_pts[:-1], link_pts[1:]
    ab = b - a
    ap = points[:, None, :] - a[None]
    tt = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    closest = a[None] + tt[..., None] * ab[None]
    return float(np.linalg.norm(points[:, None, :] - closest, axis=-1).min() - radius)


# ── inset overlay ───────────────────────────────────────────────────────────
def _font(size):
    for n in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def overlay(rgb, cam, pcb=None, mask=None, dist=None, state="", width=330):
    """Camera image with detections drawn on it, scaled to `width` pixels."""
    img = Image.fromarray(rgb).convert("RGB")
    if mask is not None and mask.any():
        red = np.zeros((*mask.shape, 4), np.uint8)
        red[mask] = [227, 73, 72, 150]
        tint = Image.fromarray(red, "RGBA").resize(img.size, Image.NEAREST)
        img = Image.alpha_composite(img.convert("RGBA"), tint).convert("RGB")
    dr = ImageDraw.Draw(img)
    sx = img.size[0] / cam.W
    if pcb is not None:
        c, yaw = pcb["pos"], pcb["yaw"]
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        hl, hw = LAYOUT["pcb"][0] / 2 * MM, LAYOUT["pcb"][1] / 2 * MM
        corners = [c[:2] + R @ np.array(k) for k in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        poly = [tuple(cam.project([*p, PCB_TOP_Z]) * sx) for p in corners]
        dr.line(poly + [poly[0]], fill=(40, 220, 90), width=max(2, int(3 * sx)))
        for p in pads_from_pose(pcb):
            u, v = cam.project(p) * sx
            dr.ellipse([u - 3, v - 3, u + 3, v + 3], outline=(255, 210, 0), width=2)
    scale = width / img.size[0]
    img = img.resize((width, int(img.size[1] * scale)), Image.LANCZOS)
    dr = ImageDraw.Draw(img)
    col = {"STOP": (227, 73, 72), "SLOW": (237, 161, 0)}.get(state, (40, 200, 90))
    dr.rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], outline=col, width=4)
    dr.rectangle([0, 0, img.size[0], 22], fill=(20, 20, 20))
    txt = "overhead camera"
    if dist is not None and np.isfinite(dist):
        txt += f"  ·  person {1000 * dist:.0f} mm"
    if state:
        txt += f"  ·  {state}"
    dr.text((6, 4), txt, font=_font(13), fill=(255, 255, 255))
    return img


if __name__ == "__main__":
    from build_scene import build_model
    m = build_model()
    d = mujoco.MjData(m)
    cam = Camera(W_FULL, H_FULL)
    np.set_printoptions(precision=4, suppress=True)
    print(f"intrinsics: f = {cam.f:.1f} px, principal point ({cam.cx}, {cam.cy}); "
          f"{1000 * (cam.C[2] - PCB_TOP_Z) / cam.f:.2f} mm/px at the PCB")
    print("T_origin_camera =\n", cam.T_world_cam)
    print("T_camera_origin =\n", np.linalg.inv(cam.T_world_cam))
    Tcw = np.linalg.inv(cam.T_world_cam)
    for name, xy in (("UR3 base", LAYOUT["ur3_base"]), ("UR10e base", LAYOUT["ur10e_base"]),
                     ("Pos3 holder", LAYOUT["pos3"]), ("Pos4 holder", LAYOUT["pos4"])):
        p = Tcw @ np.array([xy[0] * MM, xy[1] * MM, 0 if "base" in name else 0.14, 1])
        print(f"  {name:12s} in camera frame: {np.round(1000 * p[:3]).astype(int)} mm")
    # test: place the PCB on Pos4 at a known yaw and measure it
    pcb = m.body_mocapid[m.body("pcb").id]
    true_pos = np.array([0.407, 1.052, PCB_TOP_Z - LAYOUT["pcb"][2] * MM / 2])
    yaw = np.deg2rad(181.5)
    d.mocap_pos[pcb], d.mocap_quat[pcb] = true_pos, [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)
    ov = Overhead(m)
    est, rgb = ov.measure_pcb(d)
    err = 1000 * np.linalg.norm(est["pos"][:2] - true_pos[:2])
    dyaw = np.rad2deg((est["yaw"] - yaw + np.pi) % (2 * np.pi) - np.pi)
    print(f"PCB estimate: position error {err:.2f} mm, yaw error {dyaw:+.2f} deg, "
          f"{est['n_pixels']} px, mirrored={est['mirrored']}")
