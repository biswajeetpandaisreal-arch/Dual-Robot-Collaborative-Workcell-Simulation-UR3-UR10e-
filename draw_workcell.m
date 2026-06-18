function draw_workcell(cell, robot, doFormat)
% ==============================================================
% Workcell layout drawing (bench + holders + PCB + solder points)
% + Screen model
% + Physical UR3 & UR10e models at HOME pose
% ==============================================================
if nargin < 3, doFormat = true; end

hold on;

%% ------------------ BENCH ------------------
patch([0 cell.benchX cell.benchX 0], [0 0 cell.benchY cell.benchY], [0 0 0 0], ...
    [0.7 0.5 0.3], 'FaceAlpha',0.28, 'EdgeColor',[0.4 0.2 0.1], 'LineWidth',2);

%% ------------- HOLDERS + PCBs --------------
draw_boxYXZ(cell.Pos3, cell.holder.Y, cell.holder.X, cell.holder.Z, [0.6 0.6 0.6], 0.6);
draw_pcb(cell.Pos3+[0 0 cell.holder.Z], cell.pcb.X, cell.pcb.Y, cell.pcb.Z, 'horizontal');
text(cell.Pos3(2),cell.Pos3(1),cell.Pos3(3)+70,'Pos3','FontWeight','bold','HorizontalAlignment','center');

draw_boxYXZ(cell.Pos4, cell.holder.X, cell.holder.Y, cell.holder.Z, [0.6 0.6 0.6], 0.6);
draw_pcb(cell.Pos4+[0 0 cell.holder.Z], cell.pcb.X, cell.pcb.Y, cell.pcb.Z, 'vertical');
text(cell.Pos4(2),cell.Pos4(1),cell.Pos4(3)+70,'Pos4','FontWeight','bold','HorizontalAlignment','center');

%% --------------- SOLDER POINTS -------------
for i=1:4
    p = cell.P(i,:);
    plot3(p(2),p(1),p(3)+5,'o','MarkerSize',8, ...
        'MarkerFaceColor',[1 0.8 0],'MarkerEdgeColor','k');
    text(p(2)+15,p(1),p(3)+12,sprintf('P%d',i));
end

%% ------------------- SCREEN ----------------
draw_screen();   % Added to the workcell layout figure

%% --------- ROBOTS (PHYSICAL MODEL @ HOME) --
q3_home  = poses_waypoints('UR3','HOME');
q10_home = poses_waypoints('UR10E','HOME');

[~, j3]  = fk_dh(q3_home,  robot.UR3);
[~, j10] = fk_dh(q10_home, robot.UR10E);

draw_robot_solid(j3,  [0 0.7 0.7], 'UR3');
draw_robot_solid(j10, [1 0.6 0],   'UR10e');

%% ------------------- CAMERA ----------------
draw_boxXYZ([cell.camera.x cell.camera.y cell.camera.z], 120, 120, 120, [1 1 1], 0.95);
text(cell.camera.x,cell.camera.y,cell.camera.z+140,'CAMERA', ...
    'FontWeight','bold','HorizontalAlignment','center');

%% ------------------ FORMAT -----------------
if doFormat
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    axis equal; grid on; view(45,30);
    xlim([-100 cell.benchX+100]);
    ylim([-100 cell.benchY+100]);
    zlim([0 2200]);
    title('Workcell Layout','FontWeight','bold');
end

end  % <<< IMPORTANT: end of main function


% ==============================================================
% LOCAL HELPER FUNCTIONS
% ==============================================================

function draw_pcb(center, X, Y, Z, orient)
col = [0.2 0.65 0.2];
if strcmpi(orient,'horizontal')
    W = X; L = Y;
else
    W = Y; L = X;
end
draw_boxYXZ(center, L, W, Z, col, 0.9);
end

function draw_boxYXZ(center, L, W, H, col, a)
% center is [Y X Z]
cy=center(1); cx=center(2); cz=center(3);
l=L/2; w=W/2; h=H/2;

V = [cx-w,cy-l,cz-h; cx+w,cy-l,cz-h; cx+w,cy+l,cz-h; cx-w,cy+l,cz-h;
     cx-w,cy-l,cz+h; cx+w,cy-l,cz+h; cx+w,cy+l,cz+h; cx-w,cy+l,cz+h];
F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
patch('Vertices',V,'Faces',F,'FaceColor',col,'FaceAlpha',a,'EdgeColor','k');
end

function draw_boxXYZ(center, W, L, H, col, a)
% center is [X Y Z]
cx=center(1); cy=center(2); cz=center(3);
w=W/2; l=L/2; h=H/2;

V = [cx-w,cy-l,cz-h; cx+w,cy-l,cz-h; cx+w,cy+l,cz-h; cx-w,cy+l,cz-h;
     cx-w,cy-l,cz+h; cx+w,cy-l,cz+h; cx+w,cy+l,cz+h; cx-w,cy+l,cz+h];
F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
patch('Vertices',V,'Faces',F,'FaceColor',col,'FaceAlpha',a,'EdgeColor','k');
end

function draw_screen()
% Screen centered at (X=945, Y=1250) in mm (same as your earlier model)
screenX = 945;
screenY = 1250;
h  = 500;
L  = 700;
t  = 40;

V = [screenX-t/2, screenY-L/2, 0;
     screenX+t/2, screenY-L/2, 0;
     screenX+t/2, screenY+L/2, 0;
     screenX-t/2, screenY+L/2, 0;
     screenX-t/2, screenY-L/2, h;
     screenX+t/2, screenY-L/2, h;
     screenX+t/2, screenY+L/2, h;
     screenX-t/2, screenY+L/2, h];

F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];

patch('Vertices', V, 'Faces', F, ...
      'FaceColor', [1 0.9 0], 'FaceAlpha', 0.35, ...
      'EdgeColor', [0.8 0.7 0]);

text(screenX, screenY, h+30, 'SCREEN', ...
     'FontSize', 10, 'HorizontalAlignment','center', 'FontWeight','bold');
end

function draw_robot_solid(jointsYXZ, col, name)
% jointsYXZ rows are [Y X Z]; convert to XYZ for correct geometry
jXYZ = [jointsYXZ(:,2), jointsYXZ(:,1), jointsYXZ(:,3)];

linkRadius  = 18;   % mm
jointRadius = 22;   % mm
nCirc       = 16;

% links (cylinders)
for i = 1:6
    draw_cylinder_XYZ(jXYZ(i,:), jXYZ(i+1,:), linkRadius, nCirc, col);
end

% joints (spheres)
for i = 1:7
    draw_sphere_XYZ(jXYZ(i,:), jointRadius, col*0.85);
end

% label
b = jXYZ(1,:);
text(b(1), b(2), b(3)-60, name, 'FontWeight','bold', ...
    'HorizontalAlignment','center', 'Color', col);
end

function draw_sphere_XYZ(pXYZ, r, col)
[xs,ys,zs] = sphere(10);
surf(pXYZ(1)+r*xs, pXYZ(2)+r*ys, pXYZ(3)+r*zs, ...
    'FaceColor', col, 'EdgeColor','none', 'FaceAlpha', 0.95);
end

function draw_cylinder_XYZ(p1, p2, r, n, col)
v = (p2 - p1);
L = norm(v);
if L < 1e-9, return; end
v = v / L;

% cylinder along +Z
[XC,YC,ZC] = cylinder(r, n);
ZC = ZC * L;
P = [XC(:), YC(:), ZC(:)]';

% rotate +Z to direction v
R = rot_from_a_to_b([0;0;1], v(:));
P2 = R * P;

X = reshape(P2(1,:), size(XC)) + p1(1);
Y = reshape(P2(2,:), size(YC)) + p1(2);
Z = reshape(P2(3,:), size(ZC)) + p1(3);

surf(X, Y, Z, 'FaceColor', col, 'EdgeColor','none', 'FaceAlpha', 0.95);
end

function R = rot_from_a_to_b(a, b)
a = a / norm(a); b = b / norm(b);
v = cross(a,b); c = dot(a,b); s = norm(v);

if s < 1e-9
    if c > 0
        R = eye(3);
    else
        tmp = [1;0;0];
        if abs(dot(tmp,a)) > 0.9, tmp = [0;1;0]; end
        u = cross(a,tmp); u = u/norm(u);
        R = axis_angle(u, pi);
    end
    return;
end

vx = [  0   -v(3)  v(2);
      v(3)   0   -v(1);
     -v(2)  v(1)   0 ];

R = eye(3) + vx + vx*vx*((1-c)/(s^2));
end

function R = axis_angle(u, th)
u = u / norm(u);
ux = [0 -u(3) u(2);
      u(3) 0 -u(1);
     -u(2) u(1) 0];
R = eye(3) + sin(th)*ux + (1-cos(th))*(ux*ux);
end
