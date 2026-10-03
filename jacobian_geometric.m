function J = jacobian_geometric(q, rob)


[~, ~, frames] = fk_dh(q, rob);

% Origins and z-axes in DH-local.
% fk_dh uses the MODIFIED (Craig) DH convention, where joint i rotates about
% z_i of frame i. frames{1} is the base and frames{i+1} is frame i, so
% column i uses frames{i+1}. (Using frames{i} - frame i-1 - is the
% standard-DH rule and gives wrong columns; verified against a numerical
% finite-difference Jacobian.)
o = zeros(3,7);
z = zeros(3,6);

for i=1:6
    Ti = frames{i+1};
    o(:,i) = Ti(1:3,4);
    z(:,i) = Ti(1:3,3);
end
o(:,7) = frames{7}(1:3,4);

on = o(:,7);

Jv = zeros(3,6);
Jw = zeros(3,6);
for i=1:6
    Jv(:,i) = cross(z(:,i), (on - o(:,i)));
    Jw(:,i) = z(:,i);
end

J_local = [Jv; Jw];

% Apply orthonormal permutation to map local [x;y;z] -> your [y;x;z]
P = [0 1 0;
     1 0 0;
     0 0 1];
P6 = blkdiag(P,P);

J = P6 * J_local;
end
