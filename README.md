# Dual-Robot Collaborative Workcell Simulation (UR3 + UR10e)

MSc Robotics coursework assignment (Cranfield University) simulating two UR robots sharing a workcell: a UR3 running a pick-and-place cycle and a UR10e running a soldering cycle, with joint-space trajectory generation, a PID+feedforward controller with disturbance recovery, and a manipulability-based singularity analysis.

## What's in here

- **UR3** moves Home → Pos3 → Pos4 → Home (pick & place).
- **UR10e** moves Home → P1 → P2 → P3 → P4 → Home (soldering).
- Minimum-jerk quintic trajectories are generated per joint and chained across waypoints.
- A discrete-time PID + acceleration-feedforward controller tracks each trajectory, with a one-shot disturbance injected mid-motion to test recovery.
- Forward kinematics and the geometric Jacobian are computed from first principles (modified/Craig DH convention), used both for 3D visualisation and for a Yoshikawa manipulability-index singularity check along each trajectory.
- A 3D stick-figure animation plays both robots through their respective phases in the shared cell.

## Repo structure

```
.
├── main_partC.m              # entry point — run this
├── cfg_partC.m                # all tunable parameters (gains, dt, thresholds, output paths)
├── robots_define.m            # UR3 / UR10e DH parameters
├── workcell_define.m          # bench, holder, PCB, solder-point geometry
├── poses_waypoints.m          # named joint-space poses (Home, Pos3/4, P1–P4)
├── traj_build.m                # minimum-jerk trajectory chaining
├── controller_pidff.m         # PID + feedforward joint-space controller
├── fk_dh.m                     # forward kinematics (modified DH)
├── jacobian_geometric.m       # geometric Jacobian
├── draw_workcell.m            # static 3D workcell + robot rendering
├── animate_cell.m             # 3D animation of both robots
├── plot_trajectories.m        # joint-space trajectory plots
├── plot_control_results.m     # tracking error / control effort / recovery plots
├── plot_manipulability_graph.m # manipulability-index singularity plot
├── figures/                    # generated output figures (see Results below)
└── archive/                    # superseded scripts, kept for reference (see archive/README.md)
```

## How to run

No toolboxes required — kinematics, Jacobians, and trajectory generation are all implemented from scratch.

1. Open MATLAB and `cd` into this folder.
2. Run:
   ```matlab
   main_partC
   ```
3. Figures 1–5a are generated automatically into `figures/`. The script then pauses ("Press any key to start") before running the 3D animation — press any key in the Command Window to continue.

All gains, thresholds, and the simulation timestep live in `cfg_partC.m` if you want to retune anything.

## Method summary

**Trajectory generation** (`traj_build.m`) chains minimum-jerk quintic segments between waypoints, with small joint offsets inserted before/after each task pose to give the motion a visible approach/retreat rather than a sharp stop.

**Control** (`controller_pidff.m`) is a discrete-time PID controller with acceleration feedforward and anti-windup clamping, run once per robot under normal conditions and once with a single disturbance pulse injected at a fixed time, to compare tracking error and recovery time.

**Kinematics** (`fk_dh.m`, `jacobian_geometric.m`) use the modified (Craig) DH convention. The geometric Jacobian is computed in the DH-local frame and permuted into the workcell's [Y, X, Z] axis convention used everywhere else in the project.

**Singularity analysis** (`plot_manipulability_graph.m`) computes the Yoshikawa manipulability index w(q) = √det(J·Jᵗ) along each trajectory and flags points below a threshold (5% of the trajectory's peak manipulability by default) as near-singular.

## Results

**Workcell layout** — UR3 (cyan) and UR10e (orange) in the shared cell, with Pos3/Pos4 pick-and-place fixtures and P1–P4 solder points on the screen mount.

![Workcell layout](figures/fig1_workcell_layout.png)

**Joint-space trajectories** — position, velocity, and acceleration for joints 1–3 of both robots across the full motion cycle.

![Trajectory planning](figures/fig2_trajectory_planning.png)

**Controller tracking error and control effort**, normal vs. disturbed runs, with the disturbance injection point marked.

![Controller error handling](figures/fig3_controller_error_handling.png)

**Recovery detail** — Joint 2 desired vs. actual position zoomed around the disturbance, showing the controller pulling each robot back onto its reference trajectory.

![Error recovery detail](figures/fig4_error_recovery_detail.png)

**UR3 manipulability index** along its full trajectory, with the singularity threshold and near-singular points marked.

![UR3 manipulability](figures/fig5a_ur3_manipulability.png)

**UR10e manipulability index** along its full soldering cycle (five segments: Home → P1 → P2 → P3 → P4 → Home).

![UR10e manipulability](figures/fig5b_ur10e_manipulability.png)

## Known limitations / next steps

- Both manipulability traces are noticeably jagged compared to the smoothness of the underlying joint trajectories, and both spend a large fraction of the motion below the singularity threshold. Worth plotting joints 4–6 (not currently shown in the trajectory figure) to check whether a wrist joint is behaving non-smoothly near a kinematic flip, before citing this as "frequent near-singular encounters" in any write-up — the pattern repeating across both robots suggests it's systematic rather than a one-off artifact.
- `archive/` contains an earlier SVD-based singularity approach and a standalone IK scratch script, kept for reference rather than deleted — see `archive/README.md`.

## Author

Biswajeet Panda — MSc Robotics, Cranfield University
