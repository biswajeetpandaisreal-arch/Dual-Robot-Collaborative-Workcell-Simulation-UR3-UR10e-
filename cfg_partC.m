function cfg = cfg_partC()

cfg.dt = 0.01;

cfg.outDir = 'figures';

cfg.files.fig_workcell   = fullfile(cfg.outDir, 'fig1_workcell_layout.png');
cfg.files.fig_traj       = fullfile(cfg.outDir, 'fig2_trajectory_planning.png');
cfg.files.fig_ctrl       = fullfile(cfg.outDir, 'fig3_controller_error_handling.png');
cfg.files.fig_recovery   = fullfile(cfg.outDir, 'fig4_error_recovery_detail.png');
cfg.files.fig_manip_ur3  = fullfile(cfg.outDir, 'fig5a_ur3_manipulability.png');
cfg.files.fig_manip_ur10 = fullfile(cfg.outDir, 'fig5b_ur10e_manipulability.png');

% PID + Feedforward (joint space)
cfg.ctrl.Kp = [50 50 50 30 30 30];
cfg.ctrl.Ki = [5  5  5  3  3  3];
cfg.ctrl.Kd = [10 10 10  6  6  6];
cfg.ctrl.intLimit   = 1.0;     % anti-windup clamp
cfg.ctrl.useFF      = true;
cfg.ctrl.settleFrac = 0.01;    % 1% of max error threshold for settling

% Manipulability threshold: w < manipFrac*max(w) -> near singular
cfg.sing.manipFrac   = 0.05;
cfg.sing.manipThresh = [];     % keep empty -> auto uses manipFrac

% Animation
cfg.anim.fps = 15;

end
