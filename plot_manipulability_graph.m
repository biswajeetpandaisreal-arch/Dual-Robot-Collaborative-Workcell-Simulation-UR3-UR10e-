function M = plot_manipulability_graph(robotName, traj, t, rob, cfgSing, outFile)
% Manipulability plot (Yoshikawa): w(q) = sqrt(det(J*J'))
% Plots w(t), threshold, and highlights near-singular points.

robotName = upper(robotName);

N = size(traj.q,1);
w = zeros(N,1);

for k = 1:N
    J = jacobian_geometric(traj.q(k,:), rob);     % expects 6x6
    w(k) = sqrt(max(det(J*J'), 0));
end

if isfield(cfgSing,'manipThresh') && ~isempty(cfgSing.manipThresh)
    thresh = cfgSing.manipThresh;
else
    frac = 0.05;
    if isfield(cfgSing,'manipFrac') && ~isempty(cfgSing.manipFrac)
        frac = cfgSing.manipFrac;
    end
    thresh = frac * max(w);
end

near = (w < thresh);

fig = figure('Name',sprintf('%s Manipulability',robotName),'Position',[320 120 900 520]);
plot(t, w, 'LineWidth', 1.6); hold on; grid on;

yline(thresh, 'r--', 'Singularity Threshold', 'LineWidth', 2);

if any(near)
    idx = find(near);
    plot(t(idx), w(idx), 'ro', 'MarkerSize', 6, 'LineWidth', 1.2);
    legend('Manipulability', 'Threshold', 'Near-Singular', 'Location','best');
else
    legend('Manipulability', 'Threshold', 'Location','best');
end

xlabel('Time (s)');
ylabel('Manipulability Index');
title(sprintf('%s: Manipulability Along Trajectory', robotName), 'FontWeight','bold');

if nargin >= 6 && ~isempty(outFile)
    saveas(fig, outFile);
    fprintf('Saved: %s\n', outFile);
end

M.w = w; M.thresh = thresh; M.near = near;
M.minW = min(w); M.maxW = max(w);

fprintf('%s manipulability: min=%.6e | max=%.6e | thresh=%.6e | nearPts=%d\n', ...
    robotName, M.minW, M.maxW, thresh, sum(near));
end
