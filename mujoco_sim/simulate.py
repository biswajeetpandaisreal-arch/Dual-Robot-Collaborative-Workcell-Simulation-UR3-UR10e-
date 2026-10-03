"""Dynamic MuJoCo simulation of the UR3 + UR10e workcell.

What the MATLAB version idealises, this simulates:
  * full rigid-body dynamics (gravity, inertia, coupling) at 1 kHz
  * the MATLAB PID+FF law, either applied directly as joint torque ("pid") or
    as the outer loop of a computed-torque controller ("ct"), which is the
    physical controller the MATLAB double-integrator model implicitly assumes
  * a real disturbance: a 50 ms force push on each robot's tool
  * the UR3 actually carrying the PCB from Pos3 to Pos4, and the UR10e's
    iron tip being checked against the four solder points
  * coordination: the UR10e can start before the UR3 has finished, with the
    start time chosen so the two arms keep a minimum clearance

Usage:
    python simulate.py                          # computed torque, sequential
    python simulate.py --schedule overlap       # earliest safe UR10e start
    python simulate.py --controller pid         # MATLAB law as raw torque
    python simulate.py --model-error 0.15       # controller model 15% off
    python simulate.py --gains tuned            # stiffer outer-loop gains
    python simulate.py --view                   # live MuJoCo viewer
"""

from __future__ import annotations

import argparse
import os
import time

import mujoco
import numpy as np

from cell_model import CELL, build_trajectory, yxz_mm_to_world_m

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "workcell.xml")

# Outer-loop gains (acceleration units). "matlab" = cfg_partC.m. Its integral
# gain puts a closed-loop pole at s = -0.10 (a ~10 s time constant), so errors
# after a disturbance or under model error linger. "tuned" places the poles at
# s = -28 and -5.8 +/- 6.1j.
GAINS = {
    "matlab": dict(kp=[50, 50, 50, 30, 30, 30], ki=[5, 5, 5, 3, 3, 3],
                   kd=[10, 10, 10, 6, 6, 6]),
    "tuned": dict(kp=[400] * 6, ki=[2000] * 6, kd=[40] * 6),
}
INT_LIMIT = 1.0

ROBOTS = {  # name -> (prefix, dof slice, disturbance time [s, robot-local], force [N])
    "UR3": ("ur3", slice(0, 6), 2.5, 40.0),
    "UR10E": ("ur10e", slice(6, 12), 3.0, 150.0),
}
SAFETY_MARGIN = 0.05    # m, minimum robot-robot clearance for the overlap schedule
SOLDER_TOL = 0.005      # m, tip must be within this of a solder point while hot


# ── reference schedule ─────────────────────────────────────────────────────
class Reference:
    """Trajectory of one robot, holding its first/last pose outside [t0, t0+T]."""

    def __init__(self, name, t0, dt):
        self.tr = build_trajectory(name, dt)
        self.t0, self.dt = t0, dt
        self.T = self.tr["t"][-1]

    def at(self, t):
        k = int(round((t - self.t0) / self.dt))
        k = min(max(k, 0), len(self.tr["t"]) - 1)
        moving = 0.0 <= t - self.t0 <= self.T
        z = np.zeros(6)
        return (self.tr["q"][k],
                self.tr["qd"][k] if moving else z,
                self.tr["qdd"][k] if moving else z,
                self.tr["tool"][k])


def robot_geoms(m, prefix):
    return [g for g in range(m.ngeom)
            if (m.geom(g).name or "").startswith(prefix + "_")
            or m.body(m.geom_bodyid[g]).name.startswith(prefix + "_")]


def min_clearance(m, d, g_a, g_b, distmax=0.6):
    best = distmax
    fromto = np.zeros(6)
    for a in g_a:
        for b in g_b:
            best = min(best, mujoco.mj_geomDistance(m, d, a, b, distmax, fromto))
    return best


def plan_overlap(m, dt=0.01, margin=SAFETY_MARGIN):
    """Earliest UR10e start (kinematic sweep) that keeps `margin` clearance and
    lets the UR3 place the PCB before the iron reaches the first point."""
    d = mujoco.MjData(m)
    ga, gb = robot_geoms(m, "ur3"), robot_geoms(m, "ur10e")
    r3 = Reference("UR3", 0.0, dt)
    tool = r3.tr["tool"]
    release_t = r3.tr["t"][np.where(np.diff(tool) < 0)[0][0] + 1]
    r10 = Reference("UR10E", 0.0, dt)
    first_hot = r10.tr["t"][np.argmax(r10.tr["tool"] > 0)]
    sweep = []
    for start in np.arange(0.0, r3.T + 1e-9, 0.1):
        r10.t0 = start
        if start + first_hot < release_t + 0.2:     # PCB must be placed first
            sweep.append((start, np.nan))
            continue
        worst = np.inf
        for t in np.arange(start, r3.T + 1e-9, 0.05):  # overlap window only
            d.qpos[:6] = r3.at(t)[0]
            d.qpos[6:] = r10.at(t)[0]
            mujoco.mj_kinematics(m, d)
            worst = min(worst, min_clearance(m, d, ga, gb))
        sweep.append((start, worst))
    ok = [s for s, c in sweep if not np.isnan(c) and c >= margin]
    return (ok[0] if ok else r3.T), np.array(sweep), release_t


# ── controller ─────────────────────────────────────────────────────────────
class Controller:
    def __init__(self, m, kind, model_error=0.0, gains="matlab"):
        self.kind = kind
        g = GAINS[gains]
        self.kp, self.ki, self.kd = (np.tile(np.array(g[k], float), 2)
                                     for k in ("kp", "ki", "kd"))
        self.mc = mujoco.MjModel.from_xml_path(XML)     # controller's own model
        if model_error:
            for b in range(self.mc.nbody):
                if self.mc.body(b).name.startswith(("ur3_", "ur10e_")):
                    self.mc.body_mass[b] *= 1 + model_error
                    self.mc.body_inertia[b] *= 1 + model_error
        self.dc = mujoco.MjData(self.mc)
        self.M = np.zeros((m.nv, m.nv))
        self.e_int = np.zeros(m.nv)

    def __call__(self, d, q_ref, qd_ref, qdd_ref, dt):
        e = q_ref - d.qpos
        ed = qd_ref - d.qvel
        self.e_int = np.clip(self.e_int + e * dt, -INT_LIMIT, INT_LIMIT)
        v = qdd_ref + self.kp * e + self.ki * self.e_int + self.kd * ed
        if self.kind == "pid":   # MATLAB law, output used directly as torque
            return v
        self.dc.qpos[:] = d.qpos
        self.dc.qvel[:] = d.qvel
        mujoco.mj_forward(self.mc, self.dc)
        mujoco.mj_fullM(self.mc, self.dc, self.M)
        # computed torque: M(q) v + Coriolis/gravity - passive joint damping
        return self.M @ v + self.dc.qfrc_bias - self.dc.qfrc_passive


# ── simulation ─────────────────────────────────────────────────────────────
def run(controller="ct", schedule="sequential", disturb=True, model_error=0.0,
        view=False, log_every=10, quiet=False, gains="matlab", frame_cb=None):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep

    if schedule == "overlap":
        start10, sweep, _ = plan_overlap(m)
    else:
        start10, sweep = Reference("UR3", 0, 0.01).T, None
    refs = {"UR3": Reference("UR3", 0.0, dt), "UR10E": Reference("UR10E", start10, dt)}
    t_end = max(r.t0 + r.T for r in refs.values()) + 0.5

    ctrl = Controller(m, controller, model_error, gains)
    for name, (_, sl, _, _) in ROBOTS.items():
        d.qpos[sl] = refs[name].at(0.0)[0]
    mujoco.mj_forward(m, d)

    ga, gb = robot_geoms(m, "ur3"), robot_geoms(m, "ur10e")
    tcp = {n: m.site(f"{p}_tcp").id for n, (p, *_) in ROBOTS.items()}
    tool_body = {n: m.body(f"{p}_link6").id for n, (p, *_) in ROBOTS.items()}
    pcb_mocap = m.body_mocapid[m.body("pcb").id]
    solder_sites = [m.site(f"solder{i + 1}").id for i in range(4)]
    solder_pts = yxz_mm_to_world_m(CELL["P"])
    closest = np.full(4, np.inf)     # closest hot-tip approach per point [m]
    grasp_off = None
    prev_tool3 = 0.0

    ref_data = mujoco.MjData(m)        # for reference TCP positions
    log = {k: [] for k in ("t", "q", "q_ref", "tau", "tcp", "tcp_ref", "clear",
                           "push3", "push10")}

    viewer = mujoco.viewer.launch_passive(m, d) if view else None
    wall0 = time.time()
    n_steps = int(t_end / dt)
    for k in range(n_steps):
        t = k * dt
        q_ref, qd_ref, qdd_ref, tools = np.zeros(12), np.zeros(12), np.zeros(12), {}
        for name, (_, sl, _, _) in ROBOTS.items():
            q_ref[sl], qd_ref[sl], qdd_ref[sl], tools[name] = refs[name].at(t)

        d.ctrl[:] = ctrl(d, q_ref, qd_ref, qdd_ref, dt)

        # disturbance: 50 ms horizontal push on each tool, at its MATLAB time
        d.xfrc_applied[:] = 0
        pushes = {}
        for name, (_, _, t_dist, f) in ROBOTS.items():
            local = t - refs[name].t0
            on = disturb and t_dist <= local < t_dist + 0.05
            pushes[name] = on
            if on:
                d.xfrc_applied[tool_body[name], :3] = [0.0, f, 0.0]

        mujoco.mj_step(m, d)

        # UR3 gripper: carry the PCB while closed
        tool3 = tools["UR3"]
        R = d.site_xmat[tcp["UR3"]].reshape(3, 3)
        p = d.site_xpos[tcp["UR3"]]
        if tool3 > 0.5 and prev_tool3 <= 0.5:
            pcb_R = np.zeros(9)
            mujoco.mju_quat2Mat(pcb_R, d.mocap_quat[pcb_mocap])
            grasp_off = (R.T @ (d.mocap_pos[pcb_mocap] - p),
                         R.T @ pcb_R.reshape(3, 3))
        if tool3 > 0.5 and grasp_off is not None:
            d.mocap_pos[pcb_mocap] = p + R @ grasp_off[0]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (R @ grasp_off[1]).flatten())
            d.mocap_quat[pcb_mocap] = q
        prev_tool3 = tool3

        # UR10e iron: record how close the hot tip gets to each solder point
        if tools["UR10E"] > 0.5:
            tip = d.site_xpos[tcp["UR10E"]]
            for i, sp in enumerate(solder_pts):
                closest[i] = min(closest[i], np.linalg.norm(tip - sp))
                if closest[i] < SOLDER_TOL:
                    m.site_rgba[solder_sites[i]] = [0.75, 0.75, 0.8, 1]

        if k % log_every == 0:
            ref_data.qpos[:] = q_ref
            mujoco.mj_kinematics(m, ref_data)
            log["t"].append(t)
            log["q"].append(d.qpos.copy())
            log["q_ref"].append(q_ref.copy())
            log["tau"].append(d.ctrl.copy())
            log["tcp"].append(np.r_[d.site_xpos[tcp["UR3"]], d.site_xpos[tcp["UR10E"]]])
            log["tcp_ref"].append(np.r_[ref_data.site_xpos[tcp["UR3"]],
                                        ref_data.site_xpos[tcp["UR10E"]]])
            log["clear"].append(min_clearance(m, d, ga, gb))
            log["push3"].append(pushes["UR3"])
            log["push10"].append(pushes["UR10E"])

        if frame_cb is not None:
            frame_cb(m, d, t)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            lag = t - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)
    if viewer is not None:
        viewer.close()

    out = {k: np.array(v) for k, v in log.items()}
    out.update(start10=start10, sweep=sweep, solder_closest=closest,
               pcb_final=d.mocap_pos[pcb_mocap].copy(), controller=controller,
               schedule=schedule, model_error=model_error, gains=gains,
               ur3_T=refs["UR3"].T, ur10_T=refs["UR10E"].T)
    if not quiet:
        summarise(out)
    return out


def summarise(o):
    t = o["t"]
    e3 = 1000 * np.linalg.norm(o["tcp"][:, :3] - o["tcp_ref"][:, :3], axis=1)
    e10 = 1000 * np.linalg.norm(o["tcp"][:, 3:] - o["tcp_ref"][:, 3:], axis=1)
    pcb_target = yxz_mm_to_world_m(CELL["Pos4"]) + [0, 0, CELL["holder"]["Z"] / 1000]
    cycle = max(o["ur3_T"], o["start10"] + o["ur10_T"])
    print(f"controller={o['controller']}  gains={o['gains']}  schedule={o['schedule']}  "
          f"model_error={o['model_error']:+.0%}")
    print(f"  cycle time {cycle:.2f} s (UR10e starts at {o['start10']:.2f} s)")
    print(f"  TCP error  UR3  : mean {e3.mean():6.2f} mm  max {e3.max():7.2f} mm")
    print(f"  TCP error  UR10e: mean {e10.mean():6.2f} mm  max {e10.max():7.2f} mm")
    print(f"  min robot-robot clearance {1000 * o['clear'].min():.0f} mm "
          f"at t = {t[o['clear'].argmin()]:.2f} s")
    print(f"  PCB placed {1000 * np.linalg.norm(o['pcb_final'] - pcb_target):.1f} mm "
          f"from the Pos4 holder centre")
    hits = [f"P{i + 1} {1000 * c:.2f} mm" + ("" if c < SOLDER_TOL else " (MISSED)")
            for i, c in enumerate(o["solder_closest"])]
    print("  closest hot-tip approach: " + ", ".join(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["ct", "pid"], default="ct")
    ap.add_argument("--schedule", choices=["sequential", "overlap"], default="sequential")
    ap.add_argument("--no-disturb", action="store_true")
    ap.add_argument("--model-error", type=float, default=0.0)
    ap.add_argument("--gains", choices=list(GAINS), default="matlab")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--save", default=None, help="save the log as .npz")
    a = ap.parse_args()
    if a.view:
        import mujoco.viewer  # noqa: F401
    o = run(a.controller, a.schedule, not a.no_disturb, a.model_error, a.view,
            gains=a.gains)
    if a.save:
        np.savez(a.save, **{k: v for k, v in o.items() if isinstance(v, np.ndarray)})


if __name__ == "__main__":
    if not os.path.exists(XML):
        from build_scene import write
        write()
    main()
