"""Dynamic MuJoCo simulation of the soldering cell (assignment brief, Part C).

Sequence (as the brief specifies): the UR3 picks the PCB from Pos3 and places
it on Pos4; once it has withdrawn, the UR10e solders four header pads. The
UR10e is planned from the PCB pose *as placed* (what the overhead camera would
measure), so placement error does not become soldering error.

Control: computed torque, tau = M(q)(qdd_ref + Kp e + Ki int(e) + Kd de) + h(q, dq),
on full rigid-body dynamics at 1 kHz. "pid" applies the outer loop directly as
torque (no dynamics model) for comparison.

Induced errors handled by the controller (Part C):
  push   a 50 ms force on each tool (an unexpected collision)
  human  an operator's hand reaches round the shield toward the PCB while the
         UR10e is soldering; speed and separation monitoring slows the UR10e
         below 400 mm and stops it below 250 mm, resuming when clear

Usage:
    python simulate.py --view                 # watch it (tuned gains, both errors)
    python simulate.py --gains matlab         # coursework gains
    python simulate.py --controller pid       # no dynamics model
    python simulate.py --model-error 0.15     # controller's model 15 % too heavy
    python simulate.py --no-push --no-human   # nominal cycle
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from build_scene import build_model
from planner import DT_PLAN, HOME_UR, Robot, plan_ur10e, plan_ur3

GAINS = {   # outer-loop gains (acceleration units); "matlab" = cfg_partC.m
    "matlab": dict(kp=[50, 50, 50, 30, 30, 30], ki=[5, 5, 5, 3, 3, 3], kd=[10, 10, 10, 6, 6, 6]),
    "tuned": dict(kp=[400] * 6, ki=[2000] * 6, kd=[40] * 6),
}
INT_LIMIT = 1.0
FINGER_OPEN, FINGER_CLOSED = 0.012, 0.0
SSM_STOP, SSM_SLOW = 0.25, 0.40       # m: speed and separation monitoring
SOLDER_TOL = 0.001                    # m: tip must be within 1 mm of a pad


class Reference:
    """A planned path played back on its own clock (so it can be paused)."""

    def __init__(self, plan):
        self.p = plan
        self.tau = 0.0              # path time
        self.T = plan["t"][-1]

    def sample(self):
        x = min(self.tau / DT_PLAN, len(self.p["t"]) - 1.001)
        i, f = int(x), x - int(x)

        def lerp(a):
            return (1 - f) * a[i] + f * a[i + 1]
        return (lerp(self.p["q"]), lerp(self.p["qd"]), lerp(self.p["qdd"]),
                self.p["tool"][i], self.p["label"][i])

    @property
    def done(self):
        return self.tau >= self.T


class Controller:
    def __init__(self, robots, kind="ct", gains="tuned", model_error=0.0):
        self.kind = kind
        self.mc = build_model()                      # the controller's own model
        if model_error:
            for b in range(self.mc.nbody):
                if self.mc.body(b).name.startswith(("ur3_", "ur10e_")):
                    self.mc.body_mass[b] *= 1 + model_error
                    self.mc.body_inertia[b] *= 1 + model_error
        self.dc = mujoco.MjData(self.mc)
        self.M = np.zeros((self.mc.nv, self.mc.nv))
        self.qadr = np.concatenate([r.qadr for r in robots])
        self.dofs = np.concatenate([r.dadr for r in robots])
        g = GAINS[gains]
        self.kp, self.ki, self.kd = (np.tile(np.array(g[k], float), 2) for k in ("kp", "ki", "kd"))
        self.e_int = np.zeros(12)

    def __call__(self, d, q_ref, qd_ref, qdd_ref, dt):
        e = q_ref - d.qpos[self.qadr]
        ed = qd_ref - d.qvel[self.dofs]
        self.e_int = np.clip(self.e_int + e * dt, -INT_LIMIT, INT_LIMIT)
        v = qdd_ref + self.kp * e + self.ki * self.e_int + self.kd * ed
        if self.kind == "pid":
            return v
        self.dc.qpos[:] = d.qpos
        self.dc.qvel[:] = d.qvel
        mujoco.mj_forward(self.mc, self.dc)
        mujoco.mj_fullM(self.mc, self.dc, self.M)
        ix = np.ix_(self.dofs, self.dofs)
        return (self.M[ix] @ v + self.dc.qfrc_bias[self.dofs]
                - self.dc.qfrc_passive[self.dofs])


def _geoms_of(m, prefix):
    return [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(prefix)
            and m.geom_group[g] != 2]          # skip visual meshes: hulls/capsules + tools


def _min_dist(m, d, ga, gb, cap=1.0):
    ft = np.zeros(6)
    return min(mujoco.mj_geomDistance(m, d, a, b, cap, ft) for a in ga for b in gb)


# The operator stands on the far side of the shield (X > 0.76 m), reaches
# round its end (Y = 1.58 m) with ~10 cm clearance, then toward the PCB.
HAND_PTS = np.array([[1.35, 1.45, 0.30], [0.97, 1.70, 0.28], [0.64, 1.71, 0.26],
                     [0.50, 1.30, 0.22]])
HAND_PARK = np.array([3.0, 3.0, -3.0])        # out of sight until needed
FOREARM = 0.30                                # m: forearm trails this far along the path
HAND_OUT, HAND_HOLD, HAND_BACK = 2.8, 2.5, 2.4   # s: reach in, hold, withdraw


def _polyline(pts, n=400):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    u = np.linspace(0, s[-1], n)
    return u, np.column_stack([np.interp(u, s, pts[:, k]) for k in range(3)])


_HAND_S, _HAND_XYZ = _polyline(HAND_PTS)


def _along(s):
    """Point at arc length s on the hand path (extended straight before its start)."""
    if s < 0:
        back = HAND_PTS[0] - HAND_PTS[1]
        return HAND_PTS[0] + (-s) * back / np.linalg.norm(back)
    return np.array([np.interp(s, _HAND_S, _HAND_XYZ[:, k]) for k in range(3)])


def _hand_path(t, t0):
    """Hand pose (pos, quat): reaches in along the path, holds, withdraws the
    same way. The forearm points back along the path, so it bends round the
    shield's end with the hand instead of cutting through the shield."""
    L = _HAND_S[-1]
    tau = t - t0
    if tau < 0 or tau > HAND_OUT + HAND_HOLD + HAND_BACK:
        return HAND_PARK, np.array([1.0, 0, 0, 0])
    if tau < HAND_OUT:
        x = tau / HAND_OUT
    elif tau < HAND_OUT + HAND_HOLD:
        x = 1.0
    else:
        x = 1.0 - (tau - HAND_OUT - HAND_HOLD) / HAND_BACK
    s = L * (10 * x**3 - 15 * x**4 + 6 * x**5)
    p = _along(s)
    behind = _along(s - FOREARM) - p
    yaw = np.arctan2(-behind[1], -behind[0])       # +x (fingers) points away from the arm
    return p, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def _seg_time(plan, label, frac):
    for a, b, lab in plan["segments"]:
        if lab == label:
            return a + frac * (b - a)
    raise KeyError(label)


def run(controller="ct", gains="tuned", model_error=0.0, push=True, human=True,
        speed=1.0, view=False, log_every=10, frame_cb=None, quiet=False):
    m = build_model()
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    r3, r10 = Robot(m, "ur3"), Robot(m, "ur10e")
    ctrl = Controller((r3, r10), controller, gains, model_error)

    plan3 = plan_ur3(m, speed)
    ref3, ref10, plan10 = Reference(plan3), None, None
    d.qpos[r3.qadr] = plan3["q"][0]
    d.qpos[r10.qadr] = HOME_UR
    mujoco.mj_forward(m, d)

    tcp3, tcp10 = m.site("ur3_tcp").id, m.site("ur10e_tcp").id
    tool3, tool10 = m.body("ur3_tool").id, m.body("ur10e_tool").id
    pcb = m.body_mocapid[m.body("pcb").id]
    hand = m.body_mocapid[m.body("hand").id]
    pad_sites = [m.site(f"solder{i + 1}").id for i in range(4)]
    pad_geoms = [m.geom(f"pad{i + 1}").id for i in range(4)]
    g3, g10 = _geoms_of(m, "ur3_"), _geoms_of(m, "ur10e_")
    g_hand = [m.geom("hand").id, m.geom("forearm").id]
    fing = [m.actuator("ur3_finger_a").id, m.actuator("ur3_finger_b").id]

    push3_t = _seg_time(plan3, "carry PCB", 0.45)
    push10_t = human_t0 = start10 = end_t = None
    grasp, prev_tool3, s10, hand_d = None, 0.0, 1.0, np.nan
    closest = np.full(4, np.inf)
    ref_data = mujoco.MjData(m)
    log = {k: [] for k in ("t", "q", "q_ref", "tau", "tcp", "tcp_ref", "w", "hand_d",
                           "s10", "clear", "push3", "push10")}
    viewer = None
    if view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)
    wall0 = time.time()
    k = 0
    while True:
        t = k * dt
        # ── references ──
        q_ref, qd_ref, qdd_ref = np.zeros(12), np.zeros(12), np.zeros(12)
        q_ref[:6], qd_ref[:6], qdd_ref[:6], tool3_cmd, lab3 = ref3.sample()
        if ref10 is None and lab3 == "return home":
            # UR3 has released the PCB and withdrawn: measure the PCB, plan UR10e
            start10 = t
            plan10 = plan_ur10e(m, d.mocap_pos[pcb].copy(), d.mocap_quat[pcb].copy(), speed)
            ref10 = Reference(plan10)
            push10_t = _seg_time(plan10, "to pad 3", 0.5)
            human_t0 = start10 + _seg_time(plan10, "solder pad 2", 0.2) - HAND_OUT
        if ref10 is None:
            q_ref[6:], tool10_cmd, lab10 = HOME_UR, 0.0, "wait"
        else:
            q_ref[6:], qd10, qdd10, tool10_cmd, lab10 = ref10.sample()
            qd_ref[6:], qdd_ref[6:] = s10 * qd10, s10**2 * qdd10

        # ── operator's hand + speed and separation monitoring (every 10 ms) ──
        if human and human_t0 is not None:
            d.mocap_pos[hand], d.mocap_quat[hand] = _hand_path(t, human_t0)
            if t >= human_t0 and k % 10 == 0:
                mujoco.mj_kinematics(m, d)
                hand_d = _min_dist(m, d, g_hand, g10)
                s_target = float(np.clip((hand_d - SSM_STOP) / (SSM_SLOW - SSM_STOP), 0, 1))
                s10 += float(np.clip(s_target - s10, -0.025, 0.025))   # <= 2.5 /s ramp

        # ── control ──
        d.ctrl[np.r_[r3.act, r10.act]] = ctrl(d, q_ref, qd_ref, qdd_ref, dt)
        d.ctrl[fing] = FINGER_CLOSED if tool3_cmd > 0.5 else FINGER_OPEN
        d.xfrc_applied[:] = 0
        p3 = push and push3_t <= ref3.tau < push3_t + 0.05
        p10 = push and ref10 is not None and push10_t <= ref10.tau < push10_t + 0.05
        if p3:
            d.xfrc_applied[tool3, :3] = [0, 0, -30.0]      # 30 N downward knock
        if p10:
            d.xfrc_applied[tool10, :3] = [100.0, 0, 0]     # 100 N sideways knock

        mujoco.mj_step(m, d)

        # ── PCB carried while gripped ──
        R = d.site_xmat[tcp3].reshape(3, 3)
        p = d.site_xpos[tcp3]
        if tool3_cmd > 0.5 and prev_tool3 <= 0.5:
            Rp = np.zeros(9)
            mujoco.mju_quat2Mat(Rp, d.mocap_quat[pcb])
            grasp = (R.T @ (d.mocap_pos[pcb] - p), R.T @ Rp.reshape(3, 3))
        if tool3_cmd > 0.5 and grasp is not None:
            d.mocap_pos[pcb] = p + R @ grasp[0]
            qq = np.zeros(4)
            mujoco.mju_mat2Quat(qq, (R @ grasp[1]).flatten())
            d.mocap_quat[pcb] = qq
        prev_tool3 = tool3_cmd

        # ── soldering: hot tip within 1 mm of a pad ──
        if ref10 is not None and tool10_cmd > 0.5:
            tip = d.site_xpos[tcp10]
            for i, sid in enumerate(pad_sites):
                closest[i] = min(closest[i], np.linalg.norm(tip - d.site_xpos[sid]))
                if closest[i] < SOLDER_TOL:
                    m.geom_rgba[pad_geoms[i]] = [0.78, 0.78, 0.82, 1]

        # ── advance path clocks ──
        ref3.tau += dt
        if ref10 is not None:
            ref10.tau += s10 * dt

        if k % log_every == 0:
            ref_data.qpos[:] = d.qpos
            ref_data.qpos[r3.qadr], ref_data.qpos[r10.qadr] = q_ref[:6], q_ref[6:]
            mujoco.mj_kinematics(m, ref_data)
            log["t"].append(t)
            log["q"].append(np.r_[d.qpos[r3.qadr], d.qpos[r10.qadr]])
            log["q_ref"].append(q_ref.copy())
            log["tau"].append(d.ctrl[np.r_[r3.act, r10.act]].copy())
            log["tcp"].append(np.r_[d.site_xpos[tcp3], d.site_xpos[tcp10]])
            log["tcp_ref"].append(np.r_[ref_data.site_xpos[tcp3], ref_data.site_xpos[tcp10]])
            log["w"].append([r3.manipulability(d.qpos[r3.qadr]),
                             r10.manipulability(d.qpos[r10.qadr])])
            log["hand_d"].append(hand_d)
            log["s10"].append(s10)
            log["clear"].append(_min_dist(m, d, g3, g10) if k % 50 == 0 else np.nan)
            log["push3"].append(p3)
            log["push10"].append(p10)

        if frame_cb is not None:
            frame_cb(m, d, t, lab3 if ref10 is None else lab10, s10)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            lag = t - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)
        k += 1
        if ref10 is not None and ref10.done and end_t is None:
            end_t = t
        if (end_t is not None and t > end_t + 0.5) or t > 150:
            break
    if viewer is not None:
        viewer.close()

    out = {k: np.array(v) for k, v in log.items()}
    out.update(controller=controller, gains=gains, model_error=model_error, push=push,
               human=human, start10=start10, cycle=end_t, solder_closest=closest,
               pcb_final=d.mocap_pos[pcb].copy(), pcb_quat=d.mocap_quat[pcb].copy(),
               plan3=plan3, plan10=plan10,
               w_peak=(r3.w_peak, r10.w_peak), w_min=(r3.w_min, r10.w_min))
    if not quiet:
        summarise(out)
    return out


def summarise(o):
    e3 = 1000 * np.linalg.norm(o["tcp"][:, :3] - o["tcp_ref"][:, :3], axis=1)
    e10 = 1000 * np.linalg.norm(o["tcp"][:, 3:] - o["tcp_ref"][:, 3:], axis=1)
    w = o["w"] / np.array(o["w_peak"])
    print(f"controller={o['controller']} gains={o['gains']} model_error={o['model_error']:+.0%} "
          f"push={o['push']} human={o['human']}")
    if o["cycle"] is not None:
        print(f"  cycle {o['cycle']:.1f} s (UR10e starts {o['start10']:.1f} s)")
    print(f"  tool error UR3   mean {e3.mean():6.2f} mm  max {e3.max():7.2f} mm")
    print(f"  tool error UR10e mean {e10.mean():6.2f} mm  max {e10.max():7.2f} mm")
    print(f"  min manipulability / peak: UR3 {np.nanmin(w[:, 0]):.2f}  UR10e {np.nanmin(w[:, 1]):.2f}")
    print(f"  min UR3-UR10e clearance {1000 * np.nanmin(o['clear']):.0f} mm")
    if np.isfinite(o["hand_d"]).any():
        hold = (o["s10"] < 0.01).sum() * (o["t"][1] - o["t"][0])
        print(f"  operator hand: closest {1000 * np.nanmin(o['hand_d']):.0f} mm, "
              f"UR10e stopped for {hold:.1f} s")
    hits = [f"pad {i + 1} {1000 * c:.2f} mm" + ("" if c < SOLDER_TOL else " MISSED")
            for i, c in enumerate(o["solder_closest"])]
    print("  closest hot-tip approach: " + ", ".join(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["ct", "pid"], default="ct")
    ap.add_argument("--gains", choices=list(GAINS), default="tuned")
    ap.add_argument("--model-error", type=float, default=0.0)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-human", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="scale all segment speeds")
    ap.add_argument("--view", action="store_true")
    a = ap.parse_args()
    run(a.controller, a.gains, a.model_error, not a.no_push, not a.no_human, a.speed, a.view)


if __name__ == "__main__":
    main()
