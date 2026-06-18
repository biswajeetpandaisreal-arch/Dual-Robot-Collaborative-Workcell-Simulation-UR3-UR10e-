function [T, jointsYXZ, frames] = fk_dh(q, rob)

q = q(:)';

T = eye(4);
frames = cell(1,7);
frames{1} = T;

jointsYXZ = zeros(7,3);
jointsYXZ(1,:) = rob.baseYXZ;

for i = 1:6
    th = q(i) + rob.theta_offset(i);
    A  = dh(th, rob.d(i), rob.a(i), rob.alpha(i));
    T  = T*A;
    frames{i+1} = T;

    p = T(1:3,4); % local DH xyz
    jointsYXZ(i+1,:) = rob.baseYXZ + [p(2), p(1), p(3)];
end

end

function A = dh(theta,d,a,alpha)
% Modified (Craig) DH transform: Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)
ct=cos(theta); st=sin(theta);
ca=cos(alpha); sa=sin(alpha);

A = [ ct,    -st,    0,   a;
      st*ca, ct*ca, -sa, -sa*d;
      st*sa, ct*sa,  ca,  ca*d;
      0,      0,     0,   1 ];
end
