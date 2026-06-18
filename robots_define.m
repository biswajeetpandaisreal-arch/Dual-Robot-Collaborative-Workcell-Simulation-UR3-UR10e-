function robot = robots_define()

% ---------------- UR3 ----------------
UR3.name  = 'UR3';
UR3.alpha = [0, pi/2, 0, 0, -pi/2, pi/2];
UR3.a     = [0, 0, 244, 213, 0, 0];
UR3.d     = [152, 0, 0, 112, 85, 234];
UR3.theta_offset = [0, pi/2, 0, -pi/2, 0, 0];
UR3.baseYXZ = [370, 400, 0];      % [Y, X, Z] as in your original script

% ---------------- UR10e ----------------
UR10E.name  = 'UR10e';
UR10E.alpha = [0, pi/2, 0, 0, -pi/2, pi/2];
UR10E.a     = [0, 0, 612.7, 571.6, 0, 0];
UR10E.d     = [180, 0, 0, 174.1, 119.8, 396.5];
UR10E.theta_offset = [0, pi/2, 0, -pi/2, 0, 0];
UR10E.baseYXZ = [1695, 200, 0];

robot.UR3   = UR3;
robot.UR10E = UR10E;

end
