"""Run the scenario matrix and print the README results table (Markdown).
All scenarios use the overhead camera for the PCB pose and operator detection."""

import numpy as np

import simulate as S
from build_scene import LAYOUT, MM

SCENARIOS = [
    ("Coursework PID+FF law as raw torque", dict(controller="pid", gains="matlab")),
    ("Computed torque, coursework gains", dict(gains="matlab")),
    ("Computed torque, tuned gains", dict(gains="tuned")),
    ("Computed torque, coursework gains, model +15 %", dict(gains="matlab", model_error=0.15)),
    ("Computed torque, tuned gains, model +15 %", dict(gains="tuned", model_error=0.15)),
]


def row(name, o):
    e3 = 1000 * np.linalg.norm(o["tcp"][:, :3] - o["tcp_ref"][:, :3], axis=1)
    e10 = 1000 * np.linalg.norm(o["tcp"][:, 3:] - o["tcp_ref"][:, 3:], axis=1)
    tgt = np.array([LAYOUT["pos4"][0] * MM, LAYOUT["pos4"][1] * MM,
                    (LAYOUT["pcb_height"] + LAYOUT["pcb"][2] / 2) * MM])
    pcb = 1000 * np.linalg.norm(o["pcb_final"] - tgt)
    sol = 1000 * o["solder_closest"]
    hit = int((sol < 1000 * S.SOLDER_TOL).sum())
    dt = o["t"][1] - o["t"][0]
    stop = (o["s10"] < 0.01).sum() * dt
    unsafe = (np.isfinite(o["hand_true"]) & (o["hand_true"] < S.SSM_STOP)
              & (o["s10"] > 0.02)).sum() * dt
    return (f"| {name} | {e3.mean():.2f} | {e10.mean():.2f} | {pcb:.1f} | "
            f"{hit}/4 (worst {sol.max():.2f} mm) | {stop:.1f} / {unsafe:.2f} | {o['cycle']:.1f} |")


if __name__ == "__main__":
    print("| scenario | UR3 mean tool error (mm) | UR10e mean tool error (mm) | "
          "PCB placement error (mm) | pads within 1 mm | UR10e stopped / moving inside 250 mm (s) | "
          "cycle (s) |")
    print("|---|---|---|---|---|---|---|")
    for name, kw in SCENARIOS:
        print(row(name, S.run(quiet=True, **kw)), flush=True)
    print()
    o = S.run(quiet=True)
    v = o["vision"]
    print(f"Camera (tuned gains run): PCB position error {v['pos_err']:.2f} mm, yaw error "
          f"{v['yaw_err']:+.2f} deg, worst predicted-pad error {v['pad_err']:.2f} mm")
    both = np.isfinite(o["hand_true"]) & np.isfinite(o["hand_d"]) & (o["hand_true"] < 0.6)
    err = 1000 * (o["hand_d"][both] - o["hand_true"][both])
    print(f"Camera operator distance vs truth (< 600 mm): mean {err.mean():+.0f} mm, "
          f"range {err.min():+.0f} to {err.max():+.0f} mm")
