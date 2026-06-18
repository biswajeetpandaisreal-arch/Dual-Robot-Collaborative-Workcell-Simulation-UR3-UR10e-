function [traj, t] = traj_build(robotName, dt)

robotName = upper(robotName);

switch robotName
    case 'UR3'
        qH = poses_waypoints('UR3','HOME');
        q3 = poses_waypoints('UR3','POS3');
        q4 = poses_waypoints('UR3','POS4');

        % motion shaping (kept from your intent, but implemented differently)
        offs = deg2rad([0 25 40 -25 0 0]);

        wp = [
            qH;
            q3 + offs;
            q3;
            q3;
            q3 + 0.3*offs;
            q4 + offs;
            q4;
            q4;
            q4 + 0.3*offs;
            qH
        ];

        segT = [1.5;0.8;0.3;0.6;2.0;0.8;0.3;0.6;1.5];

        tool = [0;0;0;1;1;1;1;0;0;0]; % "gripper" state
        [traj,t] = time_scaled_quintic_chain(wp, segT, tool, dt);

    case 'UR10E'
        qH = poses_waypoints('UR10E','HOME');
        q1 = poses_waypoints('UR10E','P1');
        q2 = poses_waypoints('UR10E','P2');
        q3 = poses_waypoints('UR10E','P3');
        q4 = poses_waypoints('UR10E','P4');

        offs = deg2rad([0 15 25 -15 0 0]);

        wp = [
            qH;
            q1+offs; q1; q1; q1+offs;
            q2+offs; q2; q2; q2+offs;
            q3+offs; q3; q3; q3+offs;
            q4+offs; q4; q4; q4+offs;
            qH
        ];

        segT = [1.5;0.5;0.8;0.4;0.6; 0.5;0.8;0.4;0.6; 0.5;0.8;0.4;0.6; 0.5;0.8;0.4;1.5];
        tool = [0;0;1;1;0; 0;1;1;0; 0;1;1;0; 0;1;1;0;0]; % "iron active"
        [traj,t] = time_scaled_quintic_chain(wp, segT, tool, dt);

    otherwise
        error('Unknown robot trajectory request.');
end

end

function [traj, t] = time_scaled_quintic_chain(wp, segT, toolState, dt)
nj = size(wp,2);
qAll=[]; qdAll=[]; qddAll=[]; uAll=[]; tAll=[];
t0=0;

for s = 1:(size(wp,1)-1)
    T = segT(s);
    tt = (0:dt:T)'; n=numel(tt);

    q   = zeros(n,nj);
    qd  = zeros(n,nj);
    qdd = zeros(n,nj);

    % minimum-jerk quintic per joint
    for j=1:nj
        [q(:,j),qd(:,j),qdd(:,j)] = mj_quintic(wp(s,j), wp(s+1,j), T, tt);
    end

    u = ones(n,1)*toolState(s);
    if toolState(s)~=toolState(s+1)
        u(round(n/2):end) = toolState(s+1);
    end

    if s==1
        qAll=q; qdAll=qd; qddAll=qdd; uAll=u; tAll=tt+t0;
    else
        qAll=[qAll; q(2:end,:)]; %#ok<AGROW>
        qdAll=[qdAll; qd(2:end,:)]; %#ok<AGROW>
        qddAll=[qddAll; qdd(2:end,:)]; %#ok<AGROW>
        uAll=[uAll; u(2:end)]; %#ok<AGROW>
        tAll=[tAll; tt(2:end)+t0]; %#ok<AGROW>
    end

    t0 = t0 + T;
end

traj.q   = qAll;
traj.qd  = qdAll;
traj.qdd = qddAll;
traj.tool = uAll;

t = tAll;
end

function [q,qd,qdd] = mj_quintic(q0,qf,T,t)
% Minimum-jerk quintic with zero vel/acc endpoints
a0=q0; a1=0; a2=0;
a3= 10*(qf-q0)/T^3;
a4=-15*(qf-q0)/T^4;
a5=  6*(qf-q0)/T^5;

q   = a0 + a1*t + a2*t.^2 + a3*t.^3 + a4*t.^4 + a5*t.^5;
qd  = a1 + 2*a2*t + 3*a3*t.^2 + 4*a4*t.^3 + 5*a5*t.^4;
qdd = 2*a2 + 6*a3*t + 12*a4*t.^2 + 20*a5*t.^3;
end
