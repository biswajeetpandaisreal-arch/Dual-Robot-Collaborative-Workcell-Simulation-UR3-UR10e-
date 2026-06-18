function out = controller_pidff(traj, t, ctrl, disturb)
% Discrete-time PID + acceleration feedforward controller (joint space)
% Includes anti-windup on integral term.

dt = t(2) - t(1);
N  = numel(t);

qRef   = traj.q;
qdRef  = traj.qd;
qddRef = traj.qdd;

q  = zeros(N,6);
qd = zeros(N,6);
u  = zeros(N,6);
eN = zeros(N,1);

eInt = zeros(6,1);

q(1,:)  = qRef(1,:);
qd(1,:) = qdRef(1,:);

for k = 2:N
    qc  = q(k-1,:)';
    qdc = qd(k-1,:)' ;

    if disturb.enable && abs(t(k) - disturb.t0) < dt
        qc = qc + disturb.mag * randn(6,1);
    end

    e  = qRef(k,:)'  - qc;
    ed = qdRef(k,:)' - qdc;

    eInt = eInt + e * dt;
    lim = ctrl.intLimit;
    eInt = max(min(eInt, lim), -lim);

    if ctrl.useFF
        ff = qddRef(k,:)';
    else
        ff = zeros(6,1);
    end

    tau = ctrl.Kp(:).*e + ctrl.Ki(:).*eInt + ctrl.Kd(:).*ed + ff;

    qdd = tau;
    qd(k,:) = (qdc + qdd*dt)';
    q(k,:)  = (qc  + qdc*dt + 0.5*qdd*dt^2)';

    u(k,:) = tau';
    eN(k)  = norm(e);
end

mx  = max(eN);
mn  = mean(eN);
thr = ctrl.settleFrac * mx;

idx = find(eN > thr, 1, 'last');
if isempty(idx), st = 0; else, st = t(idx); end

out.q = q; out.qd = qd; out.u = u;
out.errN = eN; out.t = t; out.q_ref = qRef;

out.stats.maxErr     = mx;
out.stats.meanErr    = mn;
out.stats.finalErr   = eN(end);
out.stats.settleTime = st;
end
