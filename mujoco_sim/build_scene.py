"""Generate the MuJoCo workcell (MJCF) from the MATLAB DH parameters.

Each robot is built body-by-body from the modified-DH table, so the MuJoCo
kinematics are identical to fk_dh.m: body i sits at Rx(alpha_{i-1}) Tx(a_{i-1})
Tz(d_i) Rz(theta_offset_i) relative to its parent and rotates about its local
z axis. Link masses and torque limits are the published UR values; link shapes
are simple capsules/cylinders (no vendor meshes needed).

Usage:
    python build_scene.py            # writes workcell.xml next to this file
"""

from __future__ import annotations

import os

import numpy as np

from cell_model import CELL, ROBOTS, yxz_mm_to_world_m

HERE = os.path.dirname(os.path.abspath(__file__))

# Real UR wrist-3-to-flange distances; the rest of d6 is the tool.
FLANGE_D6 = {"UR3": 81.9, "UR10E": 116.55}
STYLE = {
    "UR3": dict(r_link=0.032, r_joint=0.045, h_joint=0.050,
                accent="0.10 0.70 0.70 1", prefix="ur3"),
    "UR10E": dict(r_link=0.055, r_joint=0.075, h_joint=0.085,
                  accent="1.00 0.60 0.10 1", prefix="ur10e"),
}
LINK_RGBA = "0.86 0.87 0.89 1"
JOINT_RGBA = "0.20 0.34 0.56 1"


def _quat_rx_rz(alpha, theta):
    """Quaternion (w x y z) of Rx(alpha) * Rz(theta)."""
    qa = np.array([np.cos(alpha / 2), np.sin(alpha / 2), 0, 0])
    qt = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qt
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _f(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def robot_xml(name: str) -> str:
    rob, st = ROBOTS[name], STYLE[name]
    p = st["prefix"]
    a = np.array(rob["a"]) / 1000.0
    d = np.array(rob["d"]) / 1000.0
    al, off, mass = rob["alpha"], rob["theta_offset"], rob["link_mass"]
    base = yxz_mm_to_world_m(rob["baseYXZ"])

    # child origin of body i, expressed in body i-1's frame
    child_pos = [np.array([a[i], -np.sin(al[i]) * d[i], np.cos(al[i]) * d[i]])
                 for i in range(6)]

    lines = [f'<body name="{p}_base" pos="{_f(base)}">',
             f'  <geom type="cylinder" size="{st["r_joint"] * 1.25:.4f} 0.012" '
             f'pos="0 0 0.012" rgba="{JOINT_RGBA}" contype="0" conaffinity="0" mass="1"/>']
    indent = "  "
    for i in range(6):
        quat = _quat_rx_rz(al[i], off[i])
        # body i's own position must not include the joint rotation offset's
        # effect on translation: Rx Tx Tz (Rz commutes with Tz).
        lines.append(f'{indent}<body name="{p}_link{i + 1}" pos="{_f(child_pos[i])}" '
                     f'quat="{_f(quat)}">')
        indent += "  "
        lines.append(f'{indent}<joint name="{p}_j{i + 1}" type="hinge" axis="0 0 1" '
                     f'range="-6.2832 6.2832" damping="{0.5 if name == "UR3" else 2.0}" '
                     f'armature="{0.05 if name == "UR3" else 0.3}"/>')
        # joint housing (cylinder along the joint axis)
        lines.append(f'{indent}<geom name="{p}_housing{i + 1}" type="cylinder" '
                     f'size="{st["r_joint"]:.4f} {st["h_joint"] / 2:.4f}" '
                     f'rgba="{JOINT_RGBA}" mass="{0.4 * mass[i]:.4f}" group="1"/>')
        if i < 5:
            nxt = child_pos[i + 1]
            # the next body's translation is fixed in this body's frame
            if np.linalg.norm(nxt) > 1e-6:
                lines.append(f'{indent}<geom name="{p}_link{i + 1}" type="capsule" '
                             f'fromto="0 0 0 {_f(nxt)}" size="{st["r_link"]:.4f}" '
                             f'rgba="{LINK_RGBA}" mass="{0.6 * mass[i]:.4f}" group="1"/>')
            else:
                lines[-1] = lines[-1].replace(f'mass="{0.4 * mass[i]:.4f}"',
                                              f'mass="{mass[i]:.4f}"')
        else:
            # body 6 origin is the TCP; joint 6 axis runs back along -z to the
            # wrist. Split that segment into flange (grey) and tool (accent).
            d6 = d[5]
            flange = FLANGE_D6[name] / 1000.0
            lines.append(f'{indent}<geom name="{p}_flange" type="capsule" '
                         f'fromto="0 0 {-d6:.4f} 0 0 {-d6 + flange:.4f}" '
                         f'size="{st["r_link"] * 0.9:.4f}" rgba="{LINK_RGBA}" '
                         f'mass="{0.6 * mass[i]:.4f}" group="1"/>')
            if name == "UR3":   # simple two-finger gripper
                lines += [
                    f'{indent}<geom name="{p}_tool" type="cylinder" '
                    f'fromto="0 0 {-d6 + flange:.4f} 0 0 -0.030" size="0.022" '
                    f'rgba="{st["accent"]}" mass="0.3" group="1"/>',
                    f'{indent}<geom name="{p}_finger_a" type="box" size="0.006 0.012 0.022" '
                    f'pos="0.018 0 -0.012" rgba="0.25 0.25 0.27 1" mass="0.02" group="1"/>',
                    f'{indent}<geom name="{p}_finger_b" type="box" size="0.006 0.012 0.022" '
                    f'pos="-0.018 0 -0.012" rgba="0.25 0.25 0.27 1" mass="0.02" group="1"/>',
                ]
            else:               # soldering iron
                lines += [
                    f'{indent}<geom name="{p}_tool" type="cylinder" '
                    f'fromto="0 0 {-d6 + flange:.4f} 0 0 -0.060" size="0.020" '
                    f'rgba="{st["accent"]}" mass="0.4" group="1"/>',
                    f'{indent}<geom name="{p}_tip" type="capsule" fromto="0 0 -0.060 0 0 0" '
                    f'size="0.004" rgba="0.75 0.75 0.78 1" mass="0.02" group="1"/>',
                ]
            lines.append(f'{indent}<site name="{p}_tcp" pos="0 0 0" size="0.006" '
                         f'rgba="1 0 0 0"/>')
    for _ in range(6):
        indent = indent[:-2]
        lines.append(f"{indent}</body>")
    lines.append("</body>")
    return "\n".join("    " + ln for ln in lines)


def actuators_xml(name: str) -> str:
    p = STYLE[name]["prefix"]
    lim = ROBOTS[name]["torque_limit"]
    return "\n".join(
        f'    <motor name="{p}_m{i + 1}" joint="{p}_j{i + 1}" gear="1" '
        f'ctrlrange="{-lim[i]} {lim[i]}" ctrllimited="true"/>' for i in range(6))


def workcell_xml() -> str:
    c = CELL
    bx, by = c["benchX"] / 1000, c["benchY"] / 1000          # world x, y extent
    h3 = c["holder"]
    p3 = yxz_mm_to_world_m(c["Pos3"])
    p4 = yxz_mm_to_world_m(c["Pos4"])
    sc = c["screen"]
    g = []
    # bench top (z = 0) + legs
    g.append(f'<geom name="bench" type="box" size="{bx / 2} {by / 2} 0.02" '
             f'pos="{bx / 2} {by / 2} -0.02" rgba="0.72 0.58 0.42 1"/>')
    for lx in (0.05, bx - 0.05):
        for ly in (0.05, by - 0.05):
            g.append(f'<geom type="box" size="0.03 0.03 0.36" pos="{lx} {ly} -0.40" '
                     f'rgba="0.35 0.35 0.37 1"/>')
    # Pos3 holder: 90 mm along X, 150 mm along Y; Pos4 holder rotated 90 deg,
    # as in draw_workcell.m
    for nm, p, sx, sy in (("pos3", p3, h3["X"], h3["Y"]), ("pos4", p4, h3["Y"], h3["X"])):
        g.append(f'<geom name="holder_{nm}" type="box" '
                 f'size="{sx / 2000} {sy / 2000} {h3["Z"] / 2000}" pos="{_f(p)}" '
                 f'rgba="0.60 0.60 0.62 1" contype="0" conaffinity="0"/>')
        g.append(f'<geom type="cylinder" size="0.012 {(p[2] - 0.005) / 2:.4f}" '
                 f'pos="{p[0]} {p[1]} {(p[2] - 0.005) / 2:.4f}" rgba="0.45 0.45 0.47 1" '
                 f'contype="0" conaffinity="0"/>')
    # solder points
    for i, pt in enumerate(yxz_mm_to_world_m(c["P"])):
        g.append(f'<site name="solder{i + 1}" pos="{pt[0]} {pt[1]} {pt[2] + 0.005}" '
                 f'size="0.005" rgba="1 0.8 0 1"/>')
    # screen (semi-transparent)
    g.append(f'<geom name="screen" type="box" size="{sc["t"] / 2000} {sc["L"] / 2000} '
             f'{sc["h"] / 2000}" pos="{sc["X"] / 1000} {sc["Y"] / 1000} {sc["h"] / 2000}" '
             f'rgba="1 0.9 0 0.25" contype="0" conaffinity="0"/>')
    # overhead camera housing (as in draw_workcell.m)
    cam = np.array([c["benchX"] / 2, c["benchY"] / 2, 1950]) / 1000
    g.append(f'<geom type="box" size="0.06 0.06 0.06" pos="{_f(cam)}" '
             f'rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>')
    return "\n".join("    " + x for x in g), cam


def build() -> str:
    cell_geoms, cam = workcell_xml()
    pcb = CELL["pcb"]
    p3 = yxz_mm_to_world_m(CELL["Pos3"])
    pcb_z = p3[2] + CELL["holder"]["Z"] / 1000
    return f"""<mujoco model="ur3_ur10e_workcell">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <visual>
    <global offwidth="1600" offheight="900" azimuth="225" elevation="-25"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.94 0.96"
             rgb2="0.70 0.74 0.80" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.82 0.83 0.85"
             rgb2="0.76 0.77 0.79" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="8 8"/>
  </asset>
  <default>
    <!-- Robot links never collide physically: the simplified wrist shapes
         overlap, and robot-robot clearance is computed geometrically
         (simulate.py) rather than by contact. -->
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light pos="1 0.6 3" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.76" material="floor"/>
    <camera name="overview" pos="3.10 -1.35 1.55" xyaxes="0.55 0.83 0 -0.33 0.22 0.92"/>
    <camera name="overhead" pos="{_f(cam - [0, 0, 0.08])}" xyaxes="0 -1 0 1 0 0"/>
{cell_geoms}
    <body name="pcb" mocap="true" pos="{p3[0]} {p3[1]} {pcb_z}">
      <geom name="pcb" type="box" size="{pcb['X'] / 2000} {pcb['Y'] / 2000} {pcb['Z'] / 2000}"
            rgba="0.15 0.60 0.20 1" contype="0" conaffinity="0"/>
    </body>
{robot_xml("UR3")}
{robot_xml("UR10E")}
  </worldbody>
  <actuator>
{actuators_xml("UR3")}
{actuators_xml("UR10E")}
  </actuator>
</mujoco>
"""


def write(path=os.path.join(HERE, "workcell.xml")):
    with open(path, "w", encoding="utf-8") as f:
        f.write(build())
    return path


if __name__ == "__main__":
    import mujoco
    from cell_model import pose, tool_position_yxz

    path = write()
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    print(f"wrote {path}: nq={m.nq}, nu={m.nu}, total robot mass "
          f"{sum(m.body_subtreemass[m.body(n).id] for n in ('ur3_base', 'ur10e_base')):.1f} kg")
    # kinematic check: MuJoCo TCP vs MATLAB FK at every task pose
    worst = 0.0
    for rob, names, prefix, off in (("UR3", ["HOME", "POS3", "POS4"], "ur3", 0),
                                    ("UR10E", ["HOME", "P1", "P2", "P3", "P4"], "ur10e", 6)):
        for n in names:
            d.qpos[off:off + 6] = pose(rob, n)
            mujoco.mj_kinematics(m, d)
            mj = d.site(f"{prefix}_tcp").xpos
            ref = yxz_mm_to_world_m(tool_position_yxz(pose(rob, n), ROBOTS[rob]))
            worst = max(worst, 1000 * np.linalg.norm(mj - ref))
    print(f"MuJoCo TCP vs MATLAB fk_dh at task poses: max error {worst:.4f} mm")
