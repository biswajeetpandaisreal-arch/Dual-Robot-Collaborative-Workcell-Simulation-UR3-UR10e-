"""Run the scenario matrix and print the README results table (Markdown)."""

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
    stop = (o["s10"] < 0.01).sum() * (o["t"][1] - o["t"][0])
    return (f"| {name} | {e3.mean():.2f} | {e10.mean():.2f} | {pcb:.1f} | "
            f"{hit}/4 (worst {sol.max():.2f} mm) | {stop:.1f} | {o['cycle']:.1f} |")


if __name__ == "__main__":
    print("| scenario | UR3 mean tool error (mm) | UR10e mean tool error (mm) | "
          "PCB placement error (mm) | pads within 1 mm | UR10e stopped for operator (s) | "
          "cycle (s) |")
    print("|---|---|---|---|---|---|---|")
    for name, kw in SCENARIOS:
        print(row(name, S.run(quiet=True, **kw)), flush=True)
