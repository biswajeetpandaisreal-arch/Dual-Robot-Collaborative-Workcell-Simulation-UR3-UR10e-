function animate_cell(cell, robot, tr3, t3, tr10, t10, fps)
% ==============================================================
% Part C Animation (Stick Model + Screen)
% Phase 1: UR3 moves, UR10e stays home
% Phase 2: UR10e moves, UR3 stays home
% ==============================================================

if nargin < 7 || isempty(fps)
    fps = 15;
end

fig = figure('Name','Animation','Position',[60 60 1200 900]);
frame_dt = 1/fps;

T1 = t3(end);
T2 = t10(end);
T  = T1 + T2;

q3_home  = poses_waypoints('UR3','HOME');
q10_home = poses_waypoints('UR10E','HOME');

t = 0;
while t <= T && ishandle(fig)
    clf(fig);
    axes(fig); hold on;

    % --- Draw workcell (table, PCB holders, etc.)
    draw_workcell(cell, robot, false);

    % --- Draw screen (ADDED BACK)
    draw_screen();

    % --- Select phase
    if t <= T1
        q3  = sampleQ(tr3.q,  t3,  t);
        q10 = q10_home;
        titleStr = sprintf('Phase 1 (UR3 Pick & Place) | t = %.2f s', t);
    else
        q3  = q3_home;
        q10 = sampleQ(tr10.q, t10, t - T1);
        titleStr = sprintf('Phase 2 (UR10e Soldering) | t = %.2f s', t);
    end

    % --- Forward kinematics
    [~, j3]  = fk_dh(q3,  robot.UR3);
    [~, j10] = fk_dh(q10, robot.UR10E);

    % --- Draw robots (stick model)
    draw_robot_stick(j3,  [0 0.7 0.7], 'UR3');
    draw_robot_stick(j10, [1 0.6 0],   'UR10e');

    % --- View settings
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    axis equal; grid on;
    view(45, 30);
    title(titleStr,'FontWeight','bold');

    drawnow;
    pause(frame_dt);
    t = t + frame_dt;
end

fprintf('Animation complete.\n');
end

% ========================= HELPERS =========================

function q = sampleQ(Q, T, t)
t = min(max(t, T(1)), T(end));
q = zeros(1, size(Q,2));
for j = 1:size(Q,2)
    q(j) = interp1(T, Q(:,j), t, 'linear');
end
end

function draw_robot_stick(jointsYXZ, col, name)
% jointsYXZ rows are [Y X Z]

% Links
for i = 1:6
    p1 = jointsYXZ(i,:);
    p2 = jointsYXZ(i+1,:);
    plot3([p1(2) p2(2)], [p1(1) p2(1)], [p1(3) p2(3)], '-', ...
        'LineWidth', 5, 'Color', col);
end

% Joints
plot3(jointsYXZ(:,2), jointsYXZ(:,1), jointsYXZ(:,3), 'o', ...
    'MarkerSize', 7, 'MarkerFaceColor', col, 'MarkerEdgeColor', col*0.7);

% Base label
b = jointsYXZ(1,:);
text(b(2), b(1), b(3)-40, name, ...
    'FontWeight','bold', 'HorizontalAlignment','center', 'Color', col);
end

function draw_screen()
% Screen geometry (same logic as your earlier code)

sc = [1250, 945];     % [Y X] position
h  = 500;             % height
l  = 700;             % length
t  = 40;              % thickness

v = [ ...
    sc(2)-t/2, sc(1)-l/2, 0;
    sc(2)+t/2, sc(1)-l/2, 0;
    sc(2)+t/2, sc(1)+l/2, 0;
    sc(2)-t/2, sc(1)+l/2, 0;
    sc(2)-t/2, sc(1)-l/2, h;
    sc(2)+t/2, sc(1)-l/2, h;
    sc(2)+t/2, sc(1)+l/2, h;
    sc(2)-t/2, sc(1)+l/2, h ];

f = [ ...
    1 2 3 4;
    5 6 7 8;
    1 2 6 5;
    2 3 7 6;
    3 4 8 7;
    4 1 5 8 ];

patch('Vertices', v, 'Faces', f, ...
      'FaceColor', [1 0.9 0], ...
      'FaceAlpha', 0.35, ...
      'EdgeColor', [0.8 0.7 0]);

text(sc(2), sc(1), h+30, 'SCREEN', ...
    'FontSize', 10, 'HorizontalAlignment', 'center', 'FontWeight','bold');
end
