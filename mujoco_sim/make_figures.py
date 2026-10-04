"""Generate every figure and the GIF used in the README.

    python make_figures.py          # writes into ../figures/

  sim_cell.gif              the full cycle (3x speed), with both induced errors
  sim_workcell.png          the cell while the UR3 carries the PCB
  sim_controllers.png       tool error: no dynamics model vs computed torque
  sim_disturbance.png       push recovery: coursework gains vs tuned gains
  sim_human.png             operator intrusion: camera vs true distance, UR10e speed
  sim_vision.png            overhead camera: PCB pose estimate and person detection
  sim_singularity.png       manipulability along the cycle, naive vs branch-locked IK
  manipulability_fixed.png  MATLAB Jacobian before / after the DH-convention fix
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import planner  # noqa: E402
import simulate as S  # noqa: E402
import vision as V  # noqa: E402
from build_scene import build_model  # noqa: E402
from cell_model import (ROBOTS, build_trajectory, jacobian,  # noqa: E402
                        jacobian_original, manipulability)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF, RED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#e34948"

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
    fig.savefig(os.path.join(OUT, name), dpi=140, facecolor=SURF)
    plt.close(fig)
    print("wrote figures/" + name)


def tcp_err(o, robot):
    sl = slice(0, 3) if robot == "UR3" else slice(3, 6)
    return 1000 * np.linalg.norm(o["tcp"][:, sl] - o["tcp_ref"][:, sl], axis=1)


def _mark_events(ax, o, ymax):
    for key, lab in (("push3", "UR3 push"), ("push10", "UR10e push")):
        if o[key].any():
            tp = o["t"][np.argmax(o[key])]
            ax.axvline(tp, color=INK2, ls=":", lw=1)
            ax.text(tp, ymax, f" {lab}", color=INK2, fontsize=8, va="top")
    ax.axvline(o["start10"], color=INK2, lw=0.8, alpha=0.5)


def fig_controllers(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    series = [("pid", "Coursework PID+FF law as raw torque (no dynamics model)", INK2),
              ("ct_matlab", "Computed torque, coursework gains", ORANGE),
              ("ct_tuned", "Computed torque, tuned gains", BLUE)]
    for ax, robot in zip(axes, ("UR3", "UR10E")):
        for key, label, col in series:
            o = runs[key]
            ax.plot(o["t"], np.maximum(tcp_err(o, robot), 1e-3), color=col, lw=1.4, label=label)
        _mark_events(ax, runs["ct_tuned"], 2500)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3e3)
        ax.set_xlabel("time (s)")
        ax.set_title("UR3 — pick and place" if robot == "UR3" else "UR10e — soldering")
        _style(ax)
    axes[0].set_ylabel("tool position error (mm, log scale)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.01, 0.01, "Full cycle with both induced errors. Grey line: UR10e start "
             "(after the UR3 has withdrawn from Pos4).", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    _save(fig, "sim_controllers.png")


def fig_disturbance(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    for ax, me in zip(axes, (0.0, 0.15)):
        for g, label, col in (("matlab", "Coursework gains (pole at s = −0.10)", ORANGE),
                              ("tuned", "Tuned gains (poles −28, −5.8 ± 6.1j)", BLUE)):
            o = runs[f"dist_{g}_{me}"]
            ax.plot(o["t"], tcp_err(o, "UR10E"), color=col, lw=1.8, label=label)
        tp = o["t"][np.argmax(o["push10"])]
        ax.axvspan(tp, tp + 0.05, color=RED, alpha=0.3, lw=0)
        ax.set_xlim(tp - 1, tp + 8)
        ax.set_ylim(0, 60)
        ax.set_xlabel("time (s)")
        ax.set_title("UR10e, perfect dynamics model" if me == 0
                     else "UR10e, controller's model 15 % too heavy")
        _style(ax)
    axes[0].set_ylabel("tool position error (mm)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.01, 0.01, "Red band: 100 N, 50 ms knock on the solder tool between pads 2 "
             "and 3.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    _save(fig, "sim_disturbance.png")


def fig_human(o):
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ok = np.isfinite(o["hand_true"]) & (o["hand_true"] < 0.9)
    t0 = o["t"][ok][0]
    ax.plot(o["t"][ok], 1000 * o["hand_true"][ok], color=INK2, lw=1.4, ls="--",
            label="true hand ↔ UR10e distance")
    cam = np.isfinite(o["hand_d"])
    ax.plot(o["t"][cam], 1000 * o["hand_d"][cam], color=ORANGE, lw=1.8,
            label="camera estimate (20 Hz, RGB-D)")
    ax.axhline(1000 * S.SSM_SLOW, color=INK2, ls=":", lw=1)
    ax.axhline(1000 * S.SSM_STOP, color=RED, ls=":", lw=1)
    ax.text(t0 + 9.2, 1000 * S.SSM_SLOW + 12, "slow below 500 mm (+100 mm camera margin)",
            color=INK2, fontsize=8.5)
    ax.text(t0 + 9.2, 1000 * S.SSM_STOP + 12, "stop below 250 mm (+100 mm camera margin)",
            color=RED, fontsize=8.5)
    ax.set_ylabel("distance (mm)")
    ax.set_ylim(0, 900)
    ax.set_xlim(t0 - 0.5, t0 + 13)
    ax2 = ax.twinx()
    ax2.plot(o["t"], o["s10"], color=BLUE, lw=1.8, label="UR10e speed scale")
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_ylabel("UR10e speed scale", color=BLUE)
    ax2.tick_params(length=0, colors=BLUE)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("time (s)")
    ax.set_title("Unexpected human movement: camera-based speed and separation monitoring")
    _style(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, "sim_human.png")


def fig_vision(o, person_frame):
    """Overhead camera: PCB measurement and person detection."""
    vis = o["vis"]
    cam_full, cam_small = V.Camera(V.W_FULL, V.H_FULL), V.Camera(V.W_HAND, V.H_HAND)
    est = vis["pcb"]
    full = V.overlay(vis["rgb_full"], cam_full, pcb=est, width=960)
    ul, vl, uh, vh = est["roi"]
    k = 960 / V.W_FULL
    zoom = full.crop((int(ul * k), int(vl * k), int(uh * k), int(vh * k))).resize((420, 420),
                                                                              Image.LANCZOS)
    rgb, mask, dist = person_frame
    person = V.overlay(rgb, cam_small, pcb=est, mask=mask, dist=dist, state="STOP", width=960)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw=dict(width_ratios=[1.6, 0.9, 1.6]))
    v = o["vision"]
    for ax, img, title in (
            (axes[0], full, "Overhead camera after the UR3 places the PCB"),
            (axes[1], zoom, f"PCB estimate: {v['pos_err']:.2f} mm, {v['yaw_err']:+.2f}°"),
            (axes[2], person, f"Operator detected: {1000 * dist:.0f} mm from the UR10e")):
        ax.imshow(img)
        ax.set_title(title, fontsize=10.5)
        ax.axis("off")
    fig.text(0.01, 0.02, "Green: estimated board outline; yellow: predicted pads; red: pixels "
             "classified as skin or hi-vis sleeve (after background subtraction).",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "sim_vision.png")


def fig_singularity(o):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, key, name in ((axes[0], "plan3", "UR3"), (axes[1], "plan10", "UR10e")):
        pl = o[key]
        r = pl["w"] / pl["w_peak"]
        ax.plot(pl["t"], r, color=BLUE, lw=1.8)
        ax.axhline(planner.W_FRAC, color=RED, ls="--", lw=1.1)
        ax.text(0.2, planner.W_FRAC + 0.025, "near-singular threshold (5 % of peak)",
                color=RED, fontsize=8.5)
        i = np.argmin(r)
        ax.annotate(f"minimum {r[i]:.2f} ('{pl['label'][i]}')", (pl["t"][i], r[i]),
                    (pl["t"][i], r[i] - 0.22), fontsize=8.5, color=INK, ha="center",
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("path time (s)")
        ax.set_title(f"{name}: manipulability along the planned path")
        _style(ax)
    axes[0].set_ylabel("w / peak w")
    fig.tight_layout()
    _save(fig, "sim_singularity.png")


def fig_manipulability_matlab():
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
    fig.suptitle("MATLAB model: Yoshikawa manipulability along the coursework trajectory",
                 x=0.01, ha="left", fontsize=11.5)
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
    frames, snap = [], {}
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.60, 1.00, 0.22]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 150.0, -32.0
    cam_small = V.Camera(V.W_HAND, V.H_HAND)
    st = {"r": None, "next": 0.0, "best": (np.inf, None)}

    def cb(m, d, t, label, s10, vis):
        if st["r"] is None:
            st["r"] = mujoco.Renderer(m, H, W)
        # keep the camera frame where the person is closest (for the figure)
        if vis["mask"] is not None and vis["mask"].any() and vis["dist"] < st["best"][0]:
            st["best"] = (vis["dist"], (vis["rgb"].copy(), vis["mask"].copy(), vis["dist"]))
        if t + 1e-9 < st["next"]:
            return
        st["next"] += 0.2                       # 5 Hz capture, played at 15 fps = 3x
        st["r"].update_scene(d, camera=cam)
        img = Image.fromarray(st["r"].render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, 36], fill=(252, 252, 251))
        state = ("UR10e STOPPED — operator too close" if s10 < 0.02 else
                 "UR10e slowed — operator nearby" if s10 < 0.98 else label)
        col = (227, 73, 72) if s10 < 0.98 else (11, 11, 11)
        dr.text((12, 8), f"t = {t:4.1f} s", font=_font(17), fill=(11, 11, 11))
        dr.text((130, 8), state, font=_font(17), fill=col)
        dr.text((W - 80, 9), "3× speed", font=_font(15), fill=(82, 81, 78))
        # camera inset: the latest monitoring frame, or a fresh render before the
        # UR10e starts (when monitoring is not running yet)
        if vis["rgb"] is not None and t - vis["t_frame"] < 0.1:
            rgb, mask, dist = vis["rgb"], vis["mask"], vis["dist"]
        else:
            rgb, mask, dist = vis["ov"]._render(vis["ov"].r_small, d), None, None
        inset = V.overlay(rgb, cam_small, pcb=vis["pcb"], mask=mask, dist=dist,
                          state=vis.get("state", ""), width=330)
        img.paste(inset, (W - inset.size[0] - 10, H - inset.size[1] - 10))
        frames.append(img)
        if abs(t - 11.0) < 0.05:
            snap["img"] = img

    o = S.run(quiet=True, frame_cb=cb)
    snap.get("img", frames[len(frames) // 3]).save(os.path.join(OUT, "sim_workcell.png"))
    print("wrote figures/sim_workcell.png")
    small = [f.resize((720, 405), Image.LANCZOS).quantize(colors=128, method=Image.MEDIANCUT)
             for f in frames]
    path = os.path.join(OUT, "sim_cell.gif")
    small[0].save(path, save_all=True, append_images=small[1:], duration=67, loop=0,
                  optimize=True)
    print(f"wrote figures/sim_cell.gif ({len(small)} frames, "
          f"{os.path.getsize(path) / 1e6:.1f} MB)")
    fig_vision(o, st["best"][1])


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_manipulability_matlab()
    runs = {"pid": S.run("pid", "matlab", quiet=True),
            "ct_matlab": S.run("ct", "matlab", quiet=True),
            "ct_tuned": S.run("ct", "tuned", quiet=True)}
    fig_controllers(runs)
    fig_human(runs["ct_tuned"])
    fig_singularity(runs["ct_tuned"])
    for g in ("matlab", "tuned"):
        runs[f"dist_{g}_0.0"] = runs[f"ct_{g}"]
        runs[f"dist_{g}_0.15"] = S.run("ct", g, model_error=0.15, quiet=True)
    fig_disturbance(runs)
    renders()


if __name__ == "__main__":
    main()
