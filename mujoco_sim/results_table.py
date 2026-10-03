"""Run the scenario matrix and print the README results table (Markdown)."""

import numpy as np

import simulate as S
from cell_model import CELL, yxz_mm_to_world_m

SCENARIOS = [
    ("MATLAB PID+FF law as raw torque", dict(controller="pid")),
    ("Computed torque, MATLAB gains", dict()),
    ("Computed torque, tuned gains", dict(gains="tuned")),
    ("Computed torque, MATLAB gains, model +15 %", dict(model_error=0.15)),
    ("Computed torque, tuned gains, model +15 %", dict(gains="tuned", model_error=0.15)),
    ("Tuned gains, model +15 %, overlapped schedule", dict(gains="tuned", model_error=0.15, schedule="overlap")),
]


def row(name, o):
    e3 = 1000 * np.linalg.norm(o["tcp"][:, :3] - o["tcp_ref"][:, :3], axis=1)
    e10 = 1000 * np.linalg.norm(o["tcp"][:, 3:] - o["tcp_ref"][:, 3:], axis=1)
    tgt = yxz_mm_to_world_m(CELL["Pos4"]) + [0, 0, CELL["holder"]["Z"] / 1000]
    pcb = 1000 * np.linalg.norm(o["pcb_final"] - tgt)
    sol = 1000 * o["solder_closest"]
    hit = int((sol < 1000 * S.SOLDER_TOL).sum())
    cycle = max(o["ur3_T"], o["start10"] + o["ur10_T"])
    return (f"| {name} | {e3.mean():.2f} | {e10.mean():.2f} | {pcb:.1f} | "
            f"{hit}/4 (worst {sol.max():.2f} mm) | {1000 * o['clear'].min():.0f} | {cycle:.1f} |")


if __name__ == "__main__":
    print("| scenario | UR3 mean tool error (mm) | UR10e mean tool error (mm) | "
          "PCB placement error (mm) | solder points within 5 mm | min clearance (mm) | "
          "cycle (s) |")
    print("|---|---|---|---|---|---|---|")
    for name, kw in SCENARIOS:
        print(row(name, S.run(quiet=True, **{"schedule": "sequential", **kw})), flush=True)
