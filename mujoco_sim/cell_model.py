"""Python port of the MATLAB workcell model (robots_define, workcell_define,
poses_waypoints, traj_build, fk_dh, jacobian_geometric).

Everything here mirrors the MATLAB files line for line so the MuJoCo
simulation runs exactly the same trajectories. Units: millimetres in the
MATLAB-style API (to match the .m files), metres for MuJoCo.

Coordinate note: the MATLAB code *stores* positions in [Y, X, Z] order; the
DH frame's local x is workcell X and local y is workcell Y (fk_dh.m adds
[p(2), p(1), p(3)] to a [Y, X, Z] base). MuJoCo uses the natural right-handed
world x = workcell X, y = workcell Y, z = Z, so each robot's base frame is
simply translated, not rotated.
"""

from __future__ import annotations

import numpy as np

# ── robots_define.m ─────────────────────────────────────────────────────────
ROBOTS = {
    "UR3": dict(
        alpha=[0, np.pi / 2, 0, 0, -np.pi / 2, np.pi / 2],
        a=[0, 0, 244, 213, 0, 0],
        d=[152, 0, 0, 112, 85, 234],       # d6 includes the gripper (~152 mm)
        theta_offset=[0, np.pi / 2, 0, -np.pi / 2, 0, 0],
        baseYXZ=[370, 400, 0],
        # Published UR3 link masses [kg] and joint torque limits [Nm]
        link_mass=[2.0, 3.42, 1.26, 0.8, 0.8, 0.35],
        torque_limit=[56, 56, 28, 12, 12, 12],
    ),
    "UR10E": dict(
        alpha=[0, np.pi / 2, 0, 0, -np.pi / 2, np.pi / 2],
        a=[0, 0, 612.7, 571.6, 0, 0],
        d=[180, 0, 0, 174.1, 119.8, 396.5],  # d6 includes the soldering iron
        theta_offset=[0, np.pi / 2, 0, -np.pi / 2, 0, 0],
        baseYXZ=[1695, 200, 0],
        link_mass=[7.369, 13.051, 3.989, 2.1, 1.98, 0.615],
        torque_limit=[330, 330, 150, 56, 56, 56],
    ),
}

# ── workcell_define.m ───────────────────────────────────────────────────────
CELL = dict(
    benchX=1250, benchY=1945,
    holder=dict(X=90, Y=150, Z=10, elev=140),
    pcb=dict(X=85.09, Y=56.134, Z=5),
    Pos3=np.array([505, 945, 140.0]),
    Pos4=np.array([1055, 405, 140.0]),
    P=np.array([[1016.237, 368.043, 140],
                [1002.775, 368.043, 140],
                [1002.775, 395.983, 140],
                [1002.775, 419.097, 140]], float),
    screen=dict(X=945, Y=1250, h=500, L=700, t=40),
)

# ── poses_waypoints.m ───────────────────────────────────────────────────────
_POSES_DEG = {
    "UR3": {
        "HOME": [0, 0, 0, 0, 0, 0],
        "POS3": [42.7555608615, -36.7046325452, -88.4431694647,
                 -54.8521979901, -47.2430256994, 0],
        "POS4": [103.7434853085, -67.2286817766, -28.5486608619,
                 -84.2226573615, -76.2565146915, 0],
    },
    "UR10E": {
        "HOME": [-45, -45, 90, -45, 90, 0],
        "P1": [-61.6773700249, -4.2703094749, -112.1683710003, 26.4386804752, 90, 0],
        "P2": [-62.2077830829, -5.4850116729, -110.8300949769, 26.3151066498, 90, 0],
        "P3": [-60.1874976551, -6.1401471190, -110.0956455397, 26.2357926516, 90, 0],
        "P4": [-58.5628387530, -6.7499621958, -109.4040587612, 26.1540209571, 90, 0],
    },
}


def pose(robot: str, name: str) -> np.ndarray:
    return np.deg2rad(np.array(_POSES_DEG[robot.upper()][name.upper()], float))


# ── fk_dh.m / jacobian_geometric.m ──────────────────────────────────────────
def dh_modified(theta, d, a, alpha):
    """Modified (Craig) DH: Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)."""
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0, 0, 0, 1]])


def frames(q, rob):
    """frames[0] = base, frames[i] = frame of joint i (i = 1..6)."""
    T = np.eye(4)
    out = [T]
    for i in range(6):
        T = T @ dh_modified(q[i] + rob["theta_offset"][i], rob["d"][i],
                            rob["a"][i], rob["alpha"][i])
        out.append(T)
    return out


def tool_position_yxz(q, rob):
    """End-effector position in workcell [Y, X, Z] mm, as fk_dh.m reports it."""
    p = frames(q, rob)[6][:3, 3]
    return np.asarray(rob["baseYXZ"], float) + np.array([p[1], p[0], p[2]])


def jacobian(q, rob):
    """Geometric Jacobian (6x6) in the robot's DH base frame.

    Modified DH: joint i rotates about z_i of frame i, i.e. frames[i]. The
    original jacobian_geometric.m used frames{i} in MATLAB's 1-based
    indexing — frame i-1, the *standard*-DH rule — which gives wrong columns.
    """
    F = frames(q, rob)
    on = F[6][:3, 3]
    Jv = np.array([np.cross(F[i][:3, 2], on - F[i][:3, 3]) for i in range(1, 7)]).T
    Jw = np.array([F[i][:3, 2] for i in range(1, 7)]).T
    return np.vstack([Jv, Jw])


def jacobian_original(q, rob):
    """The original (buggy) MATLAB indexing, kept for the before/after figure."""
    F = frames(q, rob)
    on = F[6][:3, 3]
    Jv = np.array([np.cross(F[i][:3, 2], on - F[i][:3, 3]) for i in range(6)]).T
    Jw = np.array([F[i][:3, 2] for i in range(6)]).T
    return np.vstack([Jv, Jw])


def manipulability(J):
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


# ── traj_build.m ────────────────────────────────────────────────────────────
def _mj_quintic(q0, qf, T, t):
    s = t / T
    dq = qf - q0
    q = q0 + dq * (10 * s**3 - 15 * s**4 + 6 * s**5)
    qd = dq * (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    qdd = dq * (60 * s - 180 * s**2 + 120 * s**3) / T**2
    return q, qd, qdd


def _quintic_chain(wp, seg_t, tool, dt):
    Q, QD, QDD, U, TT = [], [], [], [], []
    t0 = 0.0
    for s in range(len(wp) - 1):
        T = seg_t[s]
        tt = np.arange(0, T + 1e-9, dt)
        q, qd, qdd = (np.zeros((len(tt), 6)) for _ in range(3))
        for j in range(6):
            q[:, j], qd[:, j], qdd[:, j] = _mj_quintic(wp[s][j], wp[s + 1][j], T, tt)
        u = np.full(len(tt), tool[s], float)
        if tool[s] != tool[s + 1]:
            # MATLAB round() rounds halves away from zero; 1-based -> 0-based
            u[int(np.floor(len(tt) / 2 + 0.5)) - 1:] = tool[s + 1]
        k = 0 if s == 0 else 1
        Q.append(q[k:]); QD.append(qd[k:]); QDD.append(qdd[k:])
        U.append(u[k:]); TT.append(tt[k:] + t0)
        t0 += T
    return dict(q=np.vstack(Q), qd=np.vstack(QD), qdd=np.vstack(QDD),
                tool=np.concatenate(U), t=np.concatenate(TT))


def build_trajectory(robot: str, dt: float = 0.01):
    robot = robot.upper()
    if robot == "UR3":
        qH, q3, q4 = pose("UR3", "HOME"), pose("UR3", "POS3"), pose("UR3", "POS4")
        offs = np.deg2rad([0, 25, 40, -25, 0, 0])
        wp = [qH, q3 + offs, q3, q3, q3 + 0.3 * offs,
              q4 + offs, q4, q4, q4 + 0.3 * offs, qH]
        seg_t = [1.5, 0.8, 0.3, 0.6, 2.0, 0.8, 0.3, 0.6, 1.5]
        tool = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]          # gripper closed
    elif robot == "UR10E":
        qH = pose("UR10E", "HOME")
        ps = [pose("UR10E", f"P{i}") for i in range(1, 5)]
        offs = np.deg2rad([0, 15, 25, -15, 0, 0])
        wp = [qH]
        for p in ps:
            wp += [p + offs, p, p, p + offs]
        wp += [qH]
        seg_t = [1.5] + [0.5, 0.8, 0.4, 0.6] * 3 + [0.5, 0.8, 0.4, 1.5]
        tool = [0, 0, 1, 1, 0] + [0, 1, 1, 0] * 3 + [0]  # iron active
    else:
        raise ValueError(robot)
    return _quintic_chain(wp, seg_t, tool, dt)


# ── frame conversion for MuJoCo ─────────────────────────────────────────────
def yxz_mm_to_world_m(p_yxz):
    """Workcell [Y, X, Z] mm -> MuJoCo world [x, y, z] m (x = X, y = Y)."""
    p = np.asarray(p_yxz, float)
    return p[..., [1, 0, 2]] / 1000.0


def check_waypoints(verbose=True):
    """FK of each task pose should land on the workcell targets."""
    err = []
    for name, tgt in [("POS3", CELL["Pos3"]), ("POS4", CELL["Pos4"])]:
        err.append(np.linalg.norm(tool_position_yxz(pose("UR3", name), ROBOTS["UR3"]) - tgt))
    for i in range(4):
        p = tool_position_yxz(pose("UR10E", f"P{i + 1}"), ROBOTS["UR10E"])
        err.append(np.linalg.norm(p - CELL["P"][i]))
    if verbose:
        print(f"max FK error at task poses: {max(err):.3f} mm")
    return max(err)


if __name__ == "__main__":
    check_waypoints()
    rng = np.random.default_rng(0)
    for name, rob in ROBOTS.items():
        worst = 0.0
        for _ in range(100):
            q = rng.uniform(-np.pi, np.pi, 6)
            Jn = np.zeros((3, 6))
            p0 = frames(q, rob)[6][:3, 3]
            for i in range(6):
                dq = np.zeros(6); dq[i] = 1e-6
                Jn[:, i] = (frames(q + dq, rob)[6][:3, 3] - p0) / 1e-6
            worst = max(worst, np.abs(jacobian(q, rob)[:3] - Jn).max())
        print(f"{name}: Jacobian vs finite differences, max error {worst:.1e} mm/rad")
