"""Assemble the soldering cell from the assignment brief (Figure 2) in MuJoCo.

  * UR10e: official MuJoCo Menagerie model (robots/ur10e, BSD-3)
  * UR3:   built from Universal Robots' description by make_ur3.py (robots/ur3)
  * Robotiq Hand-E gripper on the UR3 (152 mm, Figure 5) and the 28 cm solder
    tool on the UR10e (Figure 4), modelled from the brief's dimensions
  * bench 1945 x 1250 mm, holders 150 x 90 mm with the PCB 140 mm above the
    bench, protection shield, overhead camera 1950 mm above the bench centre
  * PCB 85 x 55 x 1.6 mm with four header pads to solder (the brief asks for
    points *on* the PCB)
  * an operator's hand (mocap) used to inject "unexpected human movement"

World frame = the brief's origin: top-left corner of the bench in Figure 2,
X along the 125 cm side, Y along the 194.5 cm side, Z up (right-handed).
All original robot actuators are replaced by joint-torque motors.

Usage:
    python build_scene.py        # compile, check, and write workcell.xml
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e-3

# ── layout (mm), from the brief's Figure 2 ──────────────────────────────────
LAYOUT = dict(
    bench=(1250, 1945),                          # X, Y extent
    ur3_base=(400, 370), ur10e_base=(200, 1725),  # (X, Y) centres
    pos3=(945, 505), pos4=(405, 1055),           # PCB centres (X, Y)
    holder=(150, 90, 10), pcb=(85, 55, 1.6), pcb_height=140,
    shield=dict(x=740, y=1230, length=700, thick=40, height=500),
    camera=(625, 972.5, 1950),
)
# Base yaw: point each robot's zero direction into the work area
BASE_YAW = {"ur3": np.deg2rad(55.0), "ur10e": np.deg2rad(80.0)}
TORQUE = {"ur3": [56, 56, 28, 12, 12, 12], "ur10e": [330, 330, 150, 56, 56, 56]}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

GRIPPER_TCP = 0.151    # m from flange: 1 mm short of the fingertips (Figure 5: 152 mm)
# Fingers close on the PCB's top and bottom faces (1.6 mm gap when closed).
IRON_TCP = 0.280       # m from flange: solder tip (Figure 4)
# Header pads on the PCB, in the PCB frame (u along 85 mm, v along 55 mm), mm
PADS = [(-30.0, v) for v in (-3.81, -1.27, 1.27, 3.81)]


def _quat_z(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def _xy(p):
    return [p[0] * MM, p[1] * MM]


def _workcell_spec():
    L = LAYOUT
    bx, by = L["bench"][0] * MM, L["bench"][1] * MM
    hz = L["pcb_height"] * MM
    s = mujoco.MjSpec()
    s.option.timestep = 0.001
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.visual.global_.offwidth, s.visual.global_.offheight = 1600, 900
    s.visual.headlight.ambient = [0.35, 0.35, 0.35]
    s.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    s.visual.quality.shadowsize = 4096
    s.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                  rgb1=[0.93, 0.94, 0.96], rgb2=[0.70, 0.74, 0.80], width=512, height=512)
    s.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                  rgb1=[0.82, 0.83, 0.85], rgb2=[0.76, 0.77, 0.79], width=512, height=512)
    mat = s.add_material(name="floor", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
    w = s.worldbody
    w.add_light(pos=[0.6, 1.0, 3.0], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6], castshadow=True)
    w.add_light(pos=[2.0, -0.5, 2.0], dir=[-0.6, 0.3, -0.7], diffuse=[0.3, 0.3, 0.3])

    def box(name, size, pos, rgba, body=w):
        return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                             rgba=rgba, contype=0, conaffinity=0)

    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1],
               pos=[0.6, 1.0, -0.76], material="floor", contype=0, conaffinity=0)
    box("bench", [bx / 2, by / 2, 0.02], [bx / 2, by / 2, -0.02], [0.18, 0.18, 0.19, 1])
    for lx in (0.05, bx - 0.05):
        for ly in (0.05, by - 0.05):
            box("", [0.03, 0.03, 0.36], [lx, ly, -0.40], [0.35, 0.35, 0.37, 1])
    # origin marker (the brief's reference frame): X red, Y green, Z blue
    for i, col in enumerate(([0.9, 0.2, 0.2, 1], [0.2, 0.8, 0.2, 1], [0.2, 0.4, 0.9, 1])):
        fr = [0, 0, 0.001, 0, 0, 0.001]
        fr[3 + i] += 0.10
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=fr, size=[0.004, 0, 0],
                   rgba=col, contype=0, conaffinity=0)
    # PCB holders, 150 x 90 mm (Figure 2): 150 mm along Y at Pos3, rotated
    # 90 deg at Pos4. Each has a 30 mm slot on the UR3's side so the lower
    # gripper finger can pass under the board's edge.
    hl, hw, ht = (v * MM for v in L["holder"])
    slot_w, slot_d = 0.030, 0.055
    for name, p, along_x in (("pos3", L["pos3"], True), ("pos4", L["pos4"], False)):
        x, y = _xy(p)
        a_len = hw if along_x else hw          # depth along the approach axis = 90 mm
        b_len = hl                             # width across the approach = 150 mm
        # approach axis: +X at Pos3, +Y at Pos4 (from the UR3 base)
        def piece(a0, a1, b0, b1, tag):
            ca, cb = (a0 + a1) / 2, (b0 + b1) / 2
            sa, sb = (a1 - a0) / 2, (b1 - b0) / 2
            pos = [x + ca, y + cb] if along_x else [x + cb, y + ca]
            size = [sa, sb] if along_x else [sb, sa]
            box(f"holder_{name}_{tag}", [*size, ht / 2], [*pos, hz - ht / 2], [0.55, 0.55, 0.58, 1])
        a_near = -a_len / 2
        piece(a_near, a_len / 2, -b_len / 2, -slot_w / 2, "l")
        piece(a_near, a_len / 2, slot_w / 2, b_len / 2, "r")
        piece(a_near + slot_d, a_len / 2, -slot_w / 2, slot_w / 2, "back")
        post = [x + 0.025, y] if along_x else [x, y + 0.025]
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, (hz - ht) / 2, 0],
                   pos=[*post, (hz - ht) / 2], rgba=[0.4, 0.4, 0.42, 1], contype=0, conaffinity=0)
    sh = L["shield"]
    box("shield", [sh["thick"] * MM / 2, sh["length"] * MM / 2, sh["height"] * MM / 2],
        [sh["x"] * MM, sh["y"] * MM, sh["height"] * MM / 2], [1.0, 0.85, 0.1, 0.30])
    cx, cy, cz = (v * MM for v in L["camera"])
    box("camera_box", [0.06, 0.06, 0.04], [cx, cy, cz], [0.95, 0.95, 0.95, 1])
    w.add_camera(name="overhead", pos=[cx, cy, cz - 0.05], xyaxes=[0, -1, 0, 1, 0, 0], fovy=60)

    # PCB (mocap: carried kinematically while gripped) with header pads
    pl, pw, pt = (v * MM for v in L["pcb"])
    x3, y3 = _xy(L["pos3"])
    pcb = w.add_body(name="pcb", mocap=True, pos=[x3, y3, hz + pt / 2],
                     quat=_quat_z(np.pi / 2))      # at Pos3 the 85 mm side runs along Y
    box("pcb", [pl / 2, pw / 2, pt / 2], [0, 0, 0], [0.05, 0.25, 0.55, 1], body=pcb)
    for i, (u, v) in enumerate(PADS):
        pcb.add_geom(name=f"pad{i + 1}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[0.0011, 0.0003, 0], pos=[u * MM, v * MM, pt / 2 + 0.0003],
                     rgba=[0.85, 0.65, 0.2, 1], contype=0, conaffinity=0)
        pcb.add_site(name=f"solder{i + 1}", pos=[u * MM, v * MM, pt / 2], size=[0.0015, 0, 0],
                     rgba=[1, 0.8, 0, 0])

    # operator's hand + forearm (mocap), parked off the bench until used.
    # Fingers point along +x; the forearm trails behind along -x.
    hand = w.add_body(name="hand", mocap=True, pos=[3.0, 3.0, -3.0])
    hand.add_geom(name="hand", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.02, 0.012],
                  rgba=[0.87, 0.67, 0.55, 1], contype=0, conaffinity=0)
    hand.add_geom(name="forearm", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[-0.05, 0, 0.005, -0.32, 0, 0.03], size=[0.035, 0, 0],
                  rgba=[0.25, 0.35, 0.55, 1], contype=0, conaffinity=0)
    return s


def _add_gripper(wrist3, site_pos, site_quat):
    tool = wrist3.add_body(name="ur3_tool", pos=site_pos, quat=site_quat)
    g = dict(contype=0, conaffinity=0)
    tool.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0375, 0.038, 0],
                  pos=[0, 0, 0.038], rgba=[0.06, 0.06, 0.07, 1], mass=0.7, **g)
    tool.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.031, 0.0145, 0.015],
                  pos=[0, 0, 0.0915], rgba=[0.12, 0.12, 0.13, 1], mass=0.2, **g)
    for side, name in ((1, "a"), (-1, "b")):
        f = tool.add_body(name=f"ur3_finger_{name}", pos=[0, 0, 0])
        f.add_joint(name=f"ur3_finger_{name}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[side, 0, 0], range=[0, 0.025], damping=5, armature=0.01)
        f.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.003, 0.0105, 0.0228],
                   pos=[side * 0.0038, 0, 0.1065 + 0.0228], rgba=[0.75, 0.76, 0.78, 1],
                   mass=0.03, **g)
    tool.add_site(name="ur3_tcp", pos=[0, 0, GRIPPER_TCP], size=[0.003, 0, 0], rgba=[1, 0, 0, 0])


def _add_iron(wrist3, site_pos, site_quat):
    tool = wrist3.add_body(name="ur10e_tool", pos=site_pos, quat=site_quat)
    g = dict(contype=0, conaffinity=0)
    tool.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.032, 0.03, 0], pos=[0, 0, 0.03],
                  rgba=[0.45, 0.46, 0.48, 1], mass=0.5, **g)
    tool.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0.06, 0, 0, 0.215],
                  size=[0.016, 0, 0], rgba=[0.62, 0.63, 0.66, 1], mass=0.3, **g)
    tool.add_geom(name="ur10e_tip", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[0, 0, 0.215, 0, 0, IRON_TCP - 0.001], size=[0.0018, 0, 0],
                  rgba=[0.80, 0.80, 0.82, 1], mass=0.01, **g)
    tool.add_site(name="ur10e_tcp", pos=[0, 0, IRON_TCP], size=[0.002, 0, 0], rgba=[1, 0, 0, 0])


def build_spec():
    s = _workcell_spec()
    for prefix, path, base in (("ur3", "robots/ur3/ur3.xml", LAYOUT["ur3_base"]),
                               ("ur10e", "robots/ur10e/ur10e.xml", LAYOUT["ur10e_base"])):
        child = mujoco.MjSpec.from_file(os.path.join(HERE, path))
        for a in list(child.actuators):
            child.delete(a)
        for k in list(child.keys):
            child.delete(k)
        for lt in list(child.lights):
            child.delete(lt)
        frame = s.worldbody.add_frame(pos=[*_xy(base), 0], quat=_quat_z(BASE_YAW[prefix]))
        s.attach(child, prefix=f"{prefix}_", frame=frame)
    for prefix in ("ur3", "ur10e"):
        site = s.site(f"{prefix}_attachment_site")
        wrist3 = s.body(f"{prefix}_wrist_3_link")
        (_add_gripper if prefix == "ur3" else _add_iron)(wrist3, site.pos, site.quat)
        for j, lim in zip(JOINTS, TORQUE[prefix]):
            s.add_actuator(name=f"{prefix}_{j}", target=f"{prefix}_{j}_joint",
                           trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-lim, lim],
                           ctrllimited=True, gainprm=[1] + [0] * 9)
    for name in ("a", "b"):    # gripper fingers: position servos
        s.add_actuator(name=f"ur3_finger_{name}", target=f"ur3_finger_{name}",
                       trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[0, 0.025], ctrllimited=True,
                       gainprm=[400] + [0] * 9, biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                       biasprm=[0, -400, -20] + [0] * 7)
    return s


def build_model():
    return build_spec().compile()


def robot_indices(m, prefix):
    """qpos addresses, dof addresses and actuator ids of a robot's 6 arm joints."""
    j = [m.joint(f"{prefix}_{n}_joint").id for n in JOINTS]
    return (np.array([m.jnt_qposadr[i] for i in j]), np.array([m.jnt_dofadr[i] for i in j]),
            np.array([m.actuator(f"{prefix}_{n}").id for n in JOINTS]))


if __name__ == "__main__":
    spec = build_spec()
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"compiled: nq={m.nq} nu={m.nu} nmesh={m.nmesh}; robot masses "
          f"UR3 {m.body_subtreemass[m.body('ur3_base').id]:.1f} kg, "
          f"UR10e {m.body_subtreemass[m.body('ur10e_base').id]:.1f} kg")
