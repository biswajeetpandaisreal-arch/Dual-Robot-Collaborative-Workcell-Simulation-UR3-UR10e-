function main_partC()
% ==============================================================
% PART C – Dual Robot Collaborative Simulation (UR3 + UR10e)
% - UR3: Pick & Place (Home -> Pos3 -> Pos4 -> Home)
% - UR10e: Soldering (Home -> P1 -> P2 -> P3 -> P4 -> Home)
%
% Outputs:
%  fig1_workcell_layout.png
%  fig2_trajectory_planning.png
%  fig3_controller_error_handling.png
%  fig4_error_recovery_detail.png
%  fig5a_ur3_manipulability.png
%  fig5b_ur10e_manipulability.png

% Biswajeet Panda - 484137 - Cranfield University - Msc in Robotics.
% ==============================================================
clc; close all;

cfg   = cfg_partC();
robot = robots_define();
cell  = workcell_define();

if ~exist(cfg.outDir, 'dir')
    mkdir(cfg.outDir);
end

fprintf('\n=============================================================\n');
fprintf(' PART C: COLLABORATIVE ROBOTICS SIMULATION\n');
fprintf(' UR3 Pick&Place + UR10e Soldering\n');
fprintf('=============================================================\n\n');

%% 1) Workcell layout (NO robot static figs)
fprintf('[1] Workcell layout figure...\n');
fig = figure('Name','Workcell Layout','Position',[60 60 1200 900]);
draw_workcell(cell, robot, true);             % doFormat=true for a nice static output
saveas(fig, cfg.files.fig_workcell);
fprintf('Saved: %s\n\n', cfg.files.fig_workcell);

%% 2) Trajectories (joint space)
fprintf('[2] Trajectory generation...\n');
[tr3,  t3]  = traj_build('UR3',   cfg.dt);
[tr10, t10] = traj_build('UR10E', cfg.dt);

fprintf(' UR3  : duration %.2f s | N=%d\n',  t3(end),  numel(t3));
fprintf(' UR10e: duration %.2f s | N=%d\n\n', t10(end), numel(t10));

plot_trajectories(t3, tr3, t10, tr10, cfg.files.fig_traj);

%% 3) Control simulations (PID + FF)
fprintf('[3] PID+FF control (normal + disturbed)...\n');
rng(0);   % the disturbance uses randn: fix the seed so runs are reproducible

d_none = struct('enable',false,'t0',0,'mag',0);
d_ur3  = struct('enable',true,'t0',2.5,'mag',0.10);
d_ur10 = struct('enable',true,'t0',3.0,'mag',0.08);

r3_norm  = controller_pidff(tr3,  t3,  cfg.ctrl, d_none);
r3_err   = controller_pidff(tr3,  t3,  cfg.ctrl, d_ur3);
r10_norm = controller_pidff(tr10, t10, cfg.ctrl, d_none);
r10_err  = controller_pidff(tr10, t10, cfg.ctrl, d_ur10);

plot_control_results(t3, r3_norm, r3_err, 2.5, ...
                     t10, r10_norm, r10_err, 3.0, ...
                     cfg.files.fig_ctrl, cfg.files.fig_recovery);

%% 4) Singularity (Manipulability graphs like your screenshot)
fprintf('[4] Manipulability-based singularity graphs (UR3 + UR10e)...\n');

plot_manipulability_graph('UR3',   tr3,  t3,  robot.UR3,   cfg.sing, cfg.files.fig_manip_ur3);
plot_manipulability_graph('UR10e', tr10, t10, robot.UR10E, cfg.sing, cfg.files.fig_manip_ur10);

%% 5) Summary print
fprintf('\n==================== SUMMARY ====================\n');
fprintf('Controller: PID + FF (joint-space)\n');
fprintf('Kp = %s\n', mat2str(cfg.ctrl.Kp));
fprintf('Ki = %s\n', mat2str(cfg.ctrl.Ki));
fprintf('Kd = %s\n', mat2str(cfg.ctrl.Kd));
fprintf('FF = %s\n\n', ternary(cfg.ctrl.useFF,'ON','OFF'));

fprintf('UR3 normal : max err = %.3f deg | mean err = %.3f deg\n', ...
    rad2deg(r3_norm.stats.maxErr), rad2deg(r3_norm.stats.meanErr));
fprintf('UR3 error  : max err = %.3f deg | recovery ~ %.2f s\n', ...
    rad2deg(r3_err.stats.maxErr), max(0, r3_err.stats.settleTime - 2.5));

fprintf('UR10 normal: max err = %.3f deg | mean err = %.3f deg\n', ...
    rad2deg(r10_norm.stats.maxErr), rad2deg(r10_norm.stats.meanErr));
fprintf('UR10 error : max err = %.3f deg | recovery ~ %.2f s\n', ...
    rad2deg(r10_err.stats.maxErr), max(0, r10_err.stats.settleTime - 3.0));

fprintf('\nFigures generated:\n');
fprintf(' %s\n %s\n %s\n %s\n %s\n %s\n', ...
    cfg.files.fig_workcell, cfg.files.fig_traj, ...
    cfg.files.fig_ctrl, cfg.files.fig_recovery, ...
    cfg.files.fig_manip_ur3, cfg.files.fig_manip_ur10);
fprintf('=================================================\n\n');

%% 6) Animation (UR3 phase then UR10e phase)
fprintf('[5] Animation...\nPress any key to start.\n');
pause;

animate_cell(cell, robot, tr3, t3, tr10, t10, cfg.anim.fps);
fprintf('\nDone.\n');

end

function s = ternary(cond,a,b)
if cond, s=a; else, s=b; end
end
