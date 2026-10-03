"""Generate every figure and the GIF used in the README.

    python make_figures.py          # writes into ../figures/

Figures:
  sim_workcell.png          MuJoCo render of the cell mid-cycle
  sim_cell.gif              animated cycle (overlap schedule, tuned gains)
  sim_controllers.png       tool-position error: MATLAB law vs computed torque
  sim_disturbance.png       recovery after the push: MATLAB vs tuned gains
  sim_coordination.png      clearance-based scheduling of the UR10e start
  manipulability_fixed.png  Yoshikawa index before/after the Jacobian fix
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import simulate as S  # noqa: E402
from cell_model import (ROBOTS, build_trajectory, jacobian,  # noqa: E402
                        jacobian_original, manipulability)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RED = "#e34948"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlelocation": "left", "axes.titlesize": 11,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def tcp_err(o, robot):
    sl = slice(0, 3) if robot == "UR3" else slice(3, 6)
    return 1000 * np.linalg.norm(o["tcp"][:, sl] - o["tcp_ref"][:, sl], axis=1)


# ── controllers ─────────────────────────────────────────────────────────────
def fig_controllers(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    series = [("pid", "MATLAB PID+FF law used directly as torque", INK2),
              ("ct", "Computed torque, MATLAB gains", ORANGE),
              ("ct_tuned", "Computed torque, tuned gains", BLUE)]
    for ax, robot in zip(axes, ("UR3", "UR10E")):
        for key, label, col in series:
            o = runs[key]
            ax.plot(o["t"], np.maximum(tcp_err(o, robot), 1e-3), color=col, lw=1.6,
                    label=label)
        o = runs["ct"]
        push = o["push3"] if robot == "UR3" else o["push10"]
        tp = o["t"][np.argmax(push)]
        ax.axvline(tp, color=INK2, ls=":", lw=1)
        ax.text(tp, 2500, " push", color=INK2, fontsize=8.5, va="top")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 3e3)
        ax.set_xlabel("time (s)")
        ax.set_title("UR3 — pick & place" if robot == "UR3" else "UR10e — soldering")
        _style(ax)
    axes[0].set_ylabel("tool position error (mm, log scale)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.01, 0.01, "Sequential schedule, 50 ms tool push on each robot. Without a "
             "dynamics model the UR10e cannot hold itself up against gravity.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    _save(fig, "sim_controllers.png")


def fig_disturbance(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    for ax, me in zip(axes, (0.0, 0.15)):
        for key, label, col in (("matlab", "MATLAB gains (pole at s = −0.10)", ORANGE),
                                ("tuned", "Tuned gains (poles −28, −5.8 ± 6.1j)", BLUE)):
            o = runs[f"dist_{key}_{me}"]
            e = tcp_err(o, "UR10E")
            ax.plot(o["t"], e, color=col, lw=1.8, label=label)
        tp = o["t"][np.argmax(o["push10"])]
        ax.axvspan(tp, tp + 0.05, color=RED, alpha=0.25, lw=0)
        ax.set_xlim(tp - 0.5, tp + 7)
        ax.set_ylim(0, 40)
        ax.set_xlabel("time (s)")
        ax.set_title("UR10e, perfect dynamics model" if me == 0
                     else "UR10e, controller's model 15 % too heavy")
        _style(ax)
    axes[0].set_ylabel("tool position error (mm)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.01, 0.01, "Red band: 150 N, 50 ms push on the tool. The MATLAB integral "
             "gain leaves a ~10 s tail; the 5 mm solder tolerance is missed under model "
             "error.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    _save(fig, "sim_disturbance.png")


def fig_coordination(seq, ovl):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw=dict(width_ratios=[1, 1.3]))
    sw = ovl["sweep"]
    ax = axes[0]
    ok = ~np.isnan(sw[:, 1])
    ax.plot(sw[ok, 0], 1000 * sw[ok, 1], color=BLUE, lw=1.8, marker="o", ms=3)
    ax.axhline(1000 * S.SAFETY_MARGIN, color=RED, ls="--", lw=1.2)
    ax.text(7.45, 1000 * S.SAFETY_MARGIN + 12, "50 mm safety margin", color=RED,
            fontsize=8.5)
    ax.axvline(ovl["start10"], color=INK2, ls=":", lw=1)
    ax.text(ovl["start10"], ax.get_ylim()[1] * 0.95,
            f" chosen start {ovl['start10']:.1f} s", color=INK, fontsize=8.5, va="top")
    ax.set_xlabel("UR10e start time (s)")
    ax.set_ylabel("worst-case clearance (mm; < 0 = collision)")
    ax.set_title("Planning: earliest safe UR10e start")
    _style(ax)
    ax = axes[1]
    for o, label, col in ((seq, f"sequential (cycle {seq['ur3_T'] + seq['ur10_T']:.1f} s)",
                           INK2),
                          (ovl, f"overlapped (cycle {ovl['start10'] + ovl['ur10_T']:.1f} s)",
                           BLUE)):
        ax.plot(o["t"], 1000 * np.minimum(o["clear"], 0.6), color=col, lw=1.8, label=label)
    ax.axhline(1000 * S.SAFETY_MARGIN, color=RED, ls="--", lw=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("robot-robot clearance (mm, capped at 600)")
    ax.set_title("Executed: clearance during the cycle")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    _save(fig, "sim_coordination.png")


def fig_manipulability():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), sharex="col")
    for col, name in enumerate(("UR3", "UR10E")):
        tr = build_trajectory(name, 0.01)
        rob = ROBOTS[name]
        for row, (fn, lab) in enumerate(((jacobian_original, "original jacobian_geometric.m"),
                                         (jacobian, "fixed (modified-DH axis indexing)"))):
            w = np.array([manipulability(fn(q, rob)) for q in tr["q"]])
            thr = 0.05 * w.max()
            ax = axes[row, col]
            ax.plot(tr["t"], w / w.max(), color=BLUE if row else INK2, lw=1.4)
            ax.axhline(0.05, color=RED, ls="--", lw=1.1)
            ax.fill_between(tr["t"], 0, 1.02, where=w < thr, color=RED, alpha=0.08, lw=0)
            ax.text(0.99, 0.9, f"{100 * (w < thr).mean():.0f}% below threshold",
                    transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{'UR3' if name == 'UR3' else 'UR10e'} — {lab}", fontsize=10.5)
            _style(ax)
            if col == 0:
                ax.set_ylabel("w / max w")
            if row == 1:
                ax.set_xlabel("time (s)")
    fig.suptitle("Yoshikawa manipulability along the trajectory  (dashed: 5 % threshold, "
                 "shaded: near-singular)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "manipulability_fixed.png")


# ── renders ─────────────────────────────────────────────────────────────────
def _font(size):
    for n in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def renders():
    W, H = 960, 540
    frames, snapshot = [], {}
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.60, 1.05, 0.45]
    cam.distance, cam.azimuth, cam.elevation = 2.9, 130.0, -24.0
    state = {"r": None, "next": 0.0}

    def cb(m, d, t):
        if state["r"] is None:
            state["r"] = mujoco.Renderer(m, H, W)
        if t + 1e-9 < state["next"]:
            return
        state["next"] += 0.1                     # 10 Hz capture, played at 15 fps
        r = state["r"]
        r.update_scene(d, camera=cam)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, 34], fill=(252, 252, 251))
        dr.text((12, 8), f"t = {t:4.1f} s   UR3 (teal gripper): PCB Pos3 → Pos4   ·   "
                f"UR10e (orange iron): solder P1–P4   ·   1.5× speed",
                font=_font(16), fill=(11, 11, 11))
        frames.append(img)
        if abs(t - 6.6) < 0.05:
            snapshot["img"] = img

    S.run("ct", "overlap", disturb=False, gains="tuned", quiet=True, frame_cb=cb)
    snapshot.get("img", frames[len(frames) // 2]).save(os.path.join(OUT, "sim_workcell.png"))
    print("wrote figures/sim_workcell.png")
    small = [f.resize((720, 405), Image.LANCZOS).quantize(
                 colors=128, method=Image.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
             for f in frames]
    path = os.path.join(OUT, "sim_cell.gif")
    small[0].save(path, save_all=True, append_images=small[1:], duration=67, loop=0,
                  optimize=True)
    print(f"wrote figures/sim_cell.gif ({len(small)} frames, "
          f"{os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_manipulability()
    runs = {
        "pid": S.run("pid", "sequential", quiet=True),
        "ct": S.run("ct", "sequential", quiet=True),
        "ct_tuned": S.run("ct", "sequential", quiet=True, gains="tuned"),
    }
    fig_controllers(runs)
    for g in ("matlab", "tuned"):
        for me in (0.0, 0.15):
            runs[f"dist_{g}_{me}"] = S.run("ct", "sequential", model_error=me,
                                           gains=g, quiet=True)
    fig_disturbance(runs)
    seq = S.run("ct", "sequential", gains="tuned", quiet=True)
    ovl = S.run("ct", "overlap", gains="tuned", quiet=True)
    fig_coordination(seq, ovl)
    renders()


if __name__ == "__main__":
    main()
