"""Inverse kinematics and path planning for the soldering cell (brief, Part C).

IK: damped-least-squares on the MuJoCo model (position + orientation of the
tool centre point), multi-start, rejecting solutions near a singularity and
preferring the one closest to the previous pose (no wrist flips).

Paths are built from three segment types:
  joint   minimum-jerk quintic between two IK solutions (free-space moves)
  linear  straight Cartesian line with a minimum-jerk time law, IK per sample
          (approach / retreat along the tool axis)
  dwell   hold still (gripper closing, soldering)
and are checked for singularities with the Yoshikawa manipulability index.
"""

from __future__ import annotations

import mujoco
import numpy as np

from build_scene import (LAYOUT, MM, PADS, robot_indices)

DT_PLAN = 0.01
# Near-singular = manipulability below this fraction of the robot's own peak
# (the index scales with arm size, so a fixed number cannot serve both arms).
W_FRAC = 0.05


# ── orientation helpers ─────────────────────────────────────────────────────
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def tool_down(yaw, tilt=0.0, tilt_dir=0.0):
    """Tool frame with z pointing down (into the bench), x rotated by `yaw`
    about the vertical; optionally tilted by `tilt` so the tool's upper end leans
    toward horizontal direction `tilt_dir` (the tip then points away from it)."""
    R = rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])       # z down, x along yaw
    if tilt:
        ax = np.array([-np.sin(tilt_dir), np.cos(tilt_dir), 0.0])
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rt = np.eye(3) + np.sin(tilt) * K + (1 - np.cos(tilt)) * K @ K
        R = Rt @ R
    return R


def rot_error(R, Rt):
    return 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))


# ── IK ──────────────────────────────────────────────────────────────────────
MARGIN = np.deg2rad(12)
ENFORCE_BRANCH = True     # False reproduces the naive planner (for the figure)


def in_home_branch(q):
    """UR singularities: elbow (q3 = 0, pi), wrist (q5 = 0, pi) and shoulder
    (wrist centre on the base axis). Staying on one branch — shoulder lift in
    (-pi, 0), elbow in (0, pi), wrist 2 in (-pi, 0), each with a margin —
    means a joint-space move between two poses never crosses the elbow or
    wrist singularity; the shoulder case is checked along the path."""
    return (-np.pi + MARGIN < q[1] < -MARGIN and MARGIN < q[2] < np.pi - MARGIN
            and -np.pi + MARGIN < q[4] < -MARGIN)



class Robot:
    def __init__(self, m, prefix):
        self.m, self.prefix = m, prefix
        self.d = mujoco.MjData(m)
        self.qadr, self.dadr, self.act = robot_indices(m, prefix)
        self.site = m.site(f"{prefix}_tcp").id
        self.lo = m.jnt_range[[m.dof_jntid[a] for a in self.dadr], 0]
        self.hi = m.jnt_range[[m.dof_jntid[a] for a in self.dadr], 1]
        self.Jp = np.zeros((3, m.nv))
        self.Jr = np.zeros((3, m.nv))
        rng = np.random.default_rng(0)
        w = [self.manipulability(q) for q in rng.uniform(-np.pi, np.pi, (1000, 6))]
        self.w_peak = float(np.percentile(w, 99.5))
        self.w_min = W_FRAC * self.w_peak

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jacobian(self, q):
        self.fk(q)
        mujoco.mj_jacSite(self.m, self.d, self.Jp, self.Jr, self.site)
        return np.vstack([self.Jp[:, self.dadr], self.Jr[:, self.dadr]])

    def manipulability(self, q):
        J = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def ik(self, p, R, seed, iters=300, tol=1e-6):
        q = np.array(seed, float)
        for _ in range(iters):
            pc, Rc = self.fk(q)
            e = np.r_[p - pc, rot_error(Rc, R)]
            if np.linalg.norm(e) < tol:
                return q, True
            J = self.jacobian(q)
            lam = 1e-3 + 0.05 * np.linalg.norm(e)
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), self.lo, self.hi)
        pc, Rc = self.fk(q)
        return q, np.linalg.norm(np.r_[p - pc, rot_error(Rc, R)]) < 1e-4

    def ik_best(self, p, R, near, tries=60, rng=None):
        """Solutions from several seeds; keep non-singular ones, prefer closest."""
        rng = rng or np.random.default_rng(0)
        best = None
        for k in range(tries):
            seed = near if k == 0 else near + rng.normal(0, 0.6, 6)
            q, ok = self.ik(p, R, seed)
            if not ok or (ENFORCE_BRANCH and self.manipulability(q) < self.w_min):
                continue
            # joints 2, 3, 5 define the IK branch: wrap them to (-pi, pi] and
            # require the home branch (shoulder above, elbow up, wrist down);
            # joints 1, 4, 6 take the copy nearest the previous pose
            q = q.copy()
            q[[1, 2, 4]] = (q[[1, 2, 4]] + np.pi) % (2 * np.pi) - np.pi
            for j in (0, 3, 5):
                q[j] = near[j] + (q[j] - near[j] + np.pi) % (2 * np.pi) - np.pi
            if ENFORCE_BRANCH and not in_home_branch(q):
                continue
            cost = np.linalg.norm(q - near)
            if best is None or cost < best[0]:
                best = (cost, q)
        if best is None:
            raise RuntimeError(f"{self.prefix}: no non-singular IK solution for {np.round(p, 3)}")
        return best[1]


# ── path segments ───────────────────────────────────────────────────────────
def _min_jerk(T, dt=DT_PLAN):
    t = np.arange(0, T + 1e-9, dt)
    s = t / T
    return t, 10 * s**3 - 15 * s**4 + 6 * s**5


class Path:
    def __init__(self, robot, q0):
        self.r = robot
        self.q = [np.array(q0, float)]
        self.tool = [0.0]
        self.label = ["start"]
        self.segments = []      # (t_start, t_end, label)

    @property
    def q_now(self):
        return self.q[-1]

    def _append(self, qs, tool, label):
        t0 = (len(self.q) - 1) * DT_PLAN
        self.q += list(qs[1:])
        self.tool += [tool] * (len(qs) - 1)
        self.label += [label] * (len(qs) - 1)
        self.segments.append((t0, (len(self.q) - 1) * DT_PLAN, label))

    def joint(self, q_goal, T, tool, label):
        _, s = _min_jerk(T)
        q0 = self.q_now
        self._append([q0 + si * (q_goal - q0) for si in s], tool, label)

    def linear(self, p_goal, R_goal, T, tool, label):
        p0, R0 = self.r.fk(self.q_now)
        _, s = _min_jerk(T)
        qs, q = [self.q_now], self.q_now
        for si in s[1:]:
            p = p0 + si * (p_goal - p0)
            R = R_goal if np.allclose(R0, R_goal, atol=1e-6) else _slerp(R0, R_goal, si)
            q, ok = self.r.ik(p, R, q, iters=100)
            if not ok:
                raise RuntimeError(f"{self.r.prefix}: IK failed on linear segment '{label}'")
            qs.append(q)
        self._append(qs, tool, label)

    def dwell(self, T, tool, label):
        n = int(round(T / DT_PLAN))
        self._append([self.q_now] * (n + 1), tool, label)

    def finish(self):
        q = np.array(self.q)
        qd = np.gradient(q, DT_PLAN, axis=0)
        qdd = np.gradient(qd, DT_PLAN, axis=0)
        w = np.array([self.r.manipulability(qi) for qi in q])
        t = np.arange(len(q)) * DT_PLAN
        return dict(t=t, q=q, qd=qd, qdd=qdd, tool=np.array(self.tool), label=self.label,
                    w=w, w_peak=self.r.w_peak, w_min=self.r.w_min, segments=self.segments)


def _slerp(R0, R1, s):
    Rrel = R0.T @ R1
    ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return R1
    ax = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
    ax /= 2 * np.sin(ang)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    a = s * ang
    return R0 @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)


# ── task plans (brief, Part C) ──────────────────────────────────────────────
HOME_UR = np.deg2rad([0, -90, 90, -90, -90, 0])   # UR "home": elbow up, tool down


def pcb_pose(pos_xy, yaw):
    """Pose of the PCB centre on a holder (holder top at 140 mm)."""
    z = (LAYOUT["pcb_height"] + LAYOUT["pcb"][2] / 2) * MM
    return np.array([pos_xy[0] * MM, pos_xy[1] * MM, z]), yaw


def tool_horizontal(approach):
    """Tool z along a horizontal approach direction, fingers (tool x) vertical."""
    z = np.array([np.cos(approach), np.sin(approach), 0.0])
    x = np.array([0, 0, 1.0])
    return np.column_stack([x, np.cross(z, x), z])


def plan_ur3(m, speed=1.0):
    """Pick the PCB from Pos3 and place it on Pos4 (Figure 2).

    Workspace: the PCB centres are 561 mm (Pos3) and 685 mm (Pos4) from the
    UR3 base axis. With the gripper pointing down the UR3 can only reach
    ~517 mm at PCB height, so a top-down grasp is impossible at both; pointing
    horizontally away from the base it reaches ~726 mm (sampled from the
    model). The gripper therefore pinches the PCB's near edge (fingers on its
    top and bottom faces), passing under it through the slot in each holder."""
    r = Robot(m, "ur3")
    k = 1.0 / speed
    path = Path(r, HOME_UR)
    half_w = LAYOUT["pcb"][1] / 2 * MM          # 27.5 mm: centre to near edge
    grip_in = 0.012                             # fingers 12-13 mm over the board
    back, lift = 0.08, 0.03
    up = np.array([0, 0, lift])
    targets = {}
    for name, approach in (("pos3", 0.0), ("pos4", np.pi / 2)):   # +X, then +Y
        c, _ = pcb_pose(LAYOUT[name], 0.0)
        a = np.array([np.cos(approach), np.sin(approach), 0.0])
        targets[name] = (c - (half_w - grip_in) * a, tool_horizontal(approach), a)
    g3, R3, a3 = targets["pos3"]
    g4, R4, a4 = targets["pos4"]

    path.joint(r.ik_best(g3 - back * a3, R3, path.q_now), 3.0 * k, 0, "move to Pos3")
    path.linear(g3, R3, 2.0 * k, 0, "slide in")
    path.dwell(1.0 * k, 1, "close gripper")
    path.linear(g3 + up, R3, 1.0 * k, 1, "lift PCB")
    path.linear(g3 + up - back * a3, R3, 1.5 * k, 1, "withdraw")
    path.joint(r.ik_best(g4 + up - back * a4, R4, path.q_now), 4.5 * k, 1, "carry PCB")
    path.linear(g4 + up, R4, 2.0 * k, 1, "slide over Pos4")
    path.linear(g4, R4, 1.0 * k, 1, "lower PCB")
    path.dwell(1.0 * k, 0, "open gripper")
    path.linear(g4 - back * a4, R4, 1.5 * k, 0, "withdraw")
    path.joint(HOME_UR, 3.0 * k, 0, "return home")
    out = path.finish()
    out["grasp"] = {"pos3": g3, "pos4": g4}
    return out


def plan_ur10e(m, pcb_pos, pcb_quat, speed=1.0, tilt_deg=30.0):
    """Solder the four header pads of the PCB *as it was actually placed*.

    pcb_pos / pcb_quat are the PCB pose measured when the UR3 finishes (as the
    overhead camera would report it), so placement error is absorbed. The iron
    is tilted 30 deg from vertical, leaning back toward the UR10e base, and
    approaches each pad along its own axis."""
    r = Robot(m, "ur10e")
    k = 1.0 / speed
    path = Path(r, HOME_UR)
    Rp = np.zeros(9)
    mujoco.mju_quat2Mat(Rp, pcb_quat)
    Rp = Rp.reshape(3, 3)
    pads = [pcb_pos + Rp @ np.array([u * MM, v * MM, LAYOUT["pcb"][2] * MM / 2])
            for u, v in PADS]
    base = np.array([LAYOUT["ur10e_base"][0] * MM, LAYOUT["ur10e_base"][1] * MM])
    towards_base = np.arctan2(*(base - pads[0][:2])[::-1])
    R = tool_down(towards_base, np.deg2rad(tilt_deg), towards_base)
    axis = R[:, 2]                                  # tool z (pointing into the pad)
    standoff = 0.05
    q_pre = r.ik_best(pads[0] - standoff * axis, R, path.q_now)
    path.joint(q_pre, 3.5 * k, 0, "move to PCB")
    for i, pad in enumerate(pads):
        if i:
            path.linear(pad - standoff * axis, R, 1.0 * k, 0, f"to pad {i + 1}")
        path.linear(pad, R, 1.2 * k, 0, f"approach pad {i + 1}")
        path.dwell(2.0 * k, 1, f"solder pad {i + 1}")
        path.linear(pad - standoff * axis, R, 1.0 * k, 0, f"retract pad {i + 1}")
    path.joint(HOME_UR, 3.5 * k, 0, "return home")
    out = path.finish()
    out["pads"] = np.array(pads)
    return out


if __name__ == "__main__":
    from build_scene import build_model
    m = build_model()
    for name, plan in (("UR3", plan_ur3(m)), ):
        print(f"{name}: {plan['t'][-1]:.1f} s, min w/w_peak {plan['w'].min() / plan['w_peak']:.3f} "
              f"(threshold {W_FRAC})")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p4, yaw4 = pcb_pose(LAYOUT["pos4"], 0.0)
    pl = plan_ur10e(m, p4, np.array([1.0, 0, 0, 0]))
    print(f"UR10e: {pl['t'][-1]:.1f} s, min w/w_peak {pl['w'].min() / pl['w_peak']:.3f} "
          f"(threshold {W_FRAC})")
