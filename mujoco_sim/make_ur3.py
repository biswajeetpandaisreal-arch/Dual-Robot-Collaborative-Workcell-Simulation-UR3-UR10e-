"""Build a MuJoCo model of the UR3 from Universal Robots' official description.

MuJoCo Menagerie ships a UR10e but no UR3, so this script converts the UR3
files from github.com/UniversalRobots/Universal_Robots_ROS2_Description
(BSD-3, copied into robots/ur3/source/):

  * kinematic chain: joint origins from default_kinematics.yaml, exactly as
    ur_macro.xacro assembles them (URDF rpy = Rz(yaw) Ry(pitch) Rx(roll))
  * visual meshes: the .dae files, split by material colour into .obj files,
    placed with the mesh_offset values from visual_parameters.yaml
  * collision: one convex hull per link (used for clearance checks)
  * masses: physical_parameters.yaml; inertia approximated from link bounds

Joint angles follow the UR controller convention, like the Menagerie UR10e.

Usage (needs trimesh + pycollada; the generated files are committed):
    python make_ur3.py
"""

from __future__ import annotations

import os
import re

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "robots", "ur3", "source")
OUT = os.path.join(HERE, "robots", "ur3")

LINKS = ["base", "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]
MESH = {"base": "base", "shoulder": "shoulder", "upper_arm": "upperarm",
        "forearm": "forearm", "wrist_1": "wrist1", "wrist_2": "wrist2", "wrist_3": "wrist3"}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
TORQUE = [56, 56, 28, 12, 12, 12]


def _yaml_block(text, key):
    """Tiny parser for the flat x/y/z/roll/pitch/yaw blocks in UR's yaml files."""
    blk = text.split(f"\n  {key}:")[1]
    out = {}
    for k in ("x", "y", "z", "roll", "pitch", "yaw"):
        m = re.search(rf"\n\s+{k}:\s*(!degrees\s*)?(-?[\d.eE+-]+)", blk)
        v = float(m.group(2))
        out[k] = np.deg2rad(v) if m.group(1) else v
    return out


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _f(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def _palette(col):
    """Map the .dae's dark export colours onto the Menagerie UR palette, so
    the UR3 and the UR10e look like the same product family."""
    r, g, b = (c / 255.0 for c in col)
    if b - r > 0.08:                       # blue end caps
        return np.array([0.49, 0.678, 0.8, 1])
    if r < 0.10:                           # black rings / seals
        return np.array([0.033, 0.033, 0.033, 1])
    if r < 0.20 or r > 0.30:               # joint housings and link tubes
        return np.array([0.82, 0.82, 0.82, 1])
    return np.array([0.278, 0.278, 0.278, 1])   # base plate


def convert_meshes(link, offset):
    """Split a .dae by colour into .obj parts; return [(file, rgba)], hull, bounds.

    Bounds and hull are expressed in the link frame (mesh offset applied)."""
    scene = trimesh.load(os.path.join(SRC, MESH[link] + ".dae"))
    by_colour = {}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname].copy()
        g.apply_transform(T)
        try:
            col = tuple(int(c) for c in g.visual.material.main_color[:3])
        except AttributeError:
            col = (200, 200, 200)
        by_colour.setdefault(col, []).append(g)
    parts = []
    T_off = np.eye(4)
    T_off[:3, :3] = rpy_to_mat(offset["roll"], offset["pitch"], offset["yaw"])
    T_off[:3, 3] = [offset["x"], offset["y"], offset["z"]]
    all_link = []
    for i, (col, meshes) in enumerate(sorted(by_colour.items())):
        m = trimesh.util.concatenate(meshes)
        name = f"{MESH[link]}_{i}"
        m.export(os.path.join(OUT, "assets", name + ".obj"))
        parts.append((name, _palette(col)))
        ml = m.copy()
        ml.apply_transform(T_off)
        all_link.append(ml)
    merged = trimesh.util.concatenate(all_link)
    hull = merged.convex_hull
    hull.export(os.path.join(OUT, "assets", f"{MESH[link]}_hull.obj"))
    return parts, merged.bounds


def build():
    os.makedirs(os.path.join(OUT, "assets"), exist_ok=True)
    kin = open(os.path.join(SRC, "default_kinematics.yaml")).read()
    vis = open(os.path.join(SRC, "visual_parameters.yaml")).read()
    phys = open(os.path.join(SRC, "physical_parameters.yaml")).read()
    mass = {k: float(re.search(rf"{k}_mass:\s*([\d.]+)", phys).group(1))
            for k in ["shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]}
    mass["base"] = 2.0

    assets, bodies = [], []
    for link in LINKS:
        off = _yaml_block(vis.split("mesh_files:", 1)[1], link)
        parts, bounds = convert_meshes(link, off)
        geoms = []
        q_off = mat_to_quat(rpy_to_mat(off["roll"], off["pitch"], off["yaw"]))
        for name, rgba in parts:
            assets.append(f'    <mesh file="{name}.obj"/>')
            geoms.append(f'<geom class="visual" mesh="{name}" pos="{_f([off["x"], off["y"], off["z"]])}" '
                         f'quat="{_f(q_off)}" rgba="{_f(rgba)}"/>')
        assets.append(f'    <mesh name="{MESH[link]}_hull" file="{MESH[link]}_hull.obj"/>')
        geoms.append(f'<geom class="collision" mesh="{MESH[link]}_hull"/>')
        size = np.maximum(bounds[1] - bounds[0], 1e-3)
        m = mass[link]
        inertia = m / 12 * np.array([size[1]**2 + size[2]**2, size[0]**2 + size[2]**2,
                                     size[0]**2 + size[1]**2])
        geoms.insert(0, f'<inertial pos="{_f((bounds[0] + bounds[1]) / 2)}" mass="{m}" '
                        f'diaginertia="{_f(inertia)}"/>')
        bodies.append(geoms)

    # joint origins (ur_macro.xacro): base_link -> base_link_inertia rotated pi about z
    chain = []
    for link, joint in zip(LINKS[1:], JOINTS):
        k = _yaml_block(kin, link)
        chain.append((link, joint, [k["x"], k["y"], k["z"]],
                      mat_to_quat(rpy_to_mat(k["roll"], k["pitch"], k["yaw"]))))

    # flange / tool0: wrist_3 -> flange rpy(0,-pi/2,-pi/2) -> tool0 rpy(pi/2,0,pi/2)
    R_tool0 = rpy_to_mat(0, -np.pi / 2, -np.pi / 2) @ rpy_to_mat(np.pi / 2, 0, np.pi / 2)

    lines = ['<mujoco model="ur3">',
             '  <compiler angle="radian" meshdir="assets" autolimits="true"/>',
             '  <default>',
             '    <default class="ur3">',
             '      <joint axis="0 0 1" range="-6.28319 6.28319" armature="0.05" damping="1"/>',
             '      <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2"/></default>',
             '      <default class="collision"><geom type="mesh" contype="0" conaffinity="0" group="3" rgba="1 0 0 0.3"/></default>',
             '      <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>',
             '    </default>',
             '  </default>',
             '  <asset>', *assets, '  </asset>',
             '  <worldbody>',
             '    <body name="base" childclass="ur3">',
             '      <body name="base_link_inertia" quat="0 0 0 1">']
    lines += ["        " + g for g in bodies[0]]
    indent = "        "
    for i, (link, joint, pos, quat) in enumerate(chain):
        lines.append(f'{indent}<body name="{link}_link" pos="{_f(pos)}" quat="{_f(quat)}">')
        indent += "  "
        rng = ' range="-3.14159 3.14159"' if joint == "elbow" else ""
        lines.append(f'{indent}<joint name="{joint}_joint"{rng}/>')
        lines += [indent + g for g in bodies[i + 1]]
    lines.append(f'{indent}<site name="attachment_site" quat="{_f(mat_to_quat(R_tool0))}"/>')
    for _ in chain:
        indent = indent[:-2]
        lines.append(f"{indent}</body>")
    lines += ['      </body>', '    </body>', '  </worldbody>', '  <actuator>']
    lines += [f'    <motor name="{j}" joint="{j}_joint" ctrlrange="{-t} {t}"/>'
              for j, t in zip(JOINTS, TORQUE)]
    lines += ['  </actuator>', '</mujoco>', '']
    path = os.path.join(OUT, "ur3.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


if __name__ == "__main__":
    import mujoco
    p = build()
    m = mujoco.MjModel.from_xml_path(p)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # check against UR's official DH: at q = 0 the UR3 flange sits at
    # x = -(a2 + a3) = -0.4569 ... in the 'base' frame convention used by UR
    print(f"wrote {p}: nq={m.nq}, mass {m.body_subtreemass[0]:.2f} kg, "
          f"flange at q=0: {np.round(d.site('attachment_site').xpos, 4)}")
