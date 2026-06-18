function plot_trajectories(t3,tr3,t10,tr10,outFile)

fig = figure('Name','Trajectories','Position',[200 120 1400 650]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
title(tiledlayout(2,3),'Joint-Space Trajectories','FontWeight','bold');

nexttile; plot(t3,rad2deg(tr3.q(:,1:3)),'LineWidth',1.3); grid on;
xlabel('t (s)'); ylabel('deg'); title('UR3 q (J1-3)'); legend('J1','J2','J3');

nexttile; plot(t3,rad2deg(tr3.qd(:,1:3)),'LineWidth',1.3); grid on;
xlabel('t (s)'); ylabel('deg/s'); title('UR3 qdot (J1-3)'); legend('J1','J2','J3');

nexttile; plot(t3,rad2deg(tr3.qdd(:,1:3)),'LineWidth',1.3); grid on;
xlabel('t (s)'); ylabel('deg/s^2'); title('UR3 qddot (J1-3)'); legend('J1','J2','J3');

nexttile; plot(t10,rad2deg(tr10.q(:,1:3)),'LineWidth',1.3); grid on;
xlabel('t (s)'); ylabel('deg'); title('UR10e q (J1-3)'); legend('J1','J2','J3');

nexttile; plot(t10,rad2deg(tr10.qd(:,1:3)),'LineWidth',1.3); grid on;
xlabel('t (s)'); ylabel('deg/s'); title('UR10e qdot (J1-3)'); legend('J1','J2','J3');

nexttile; plot(t10,rad2deg(tr10.qdd(:,1:3)),'LineWidth',1.3); grid on;
xlabel('t (s)'); ylabel('deg/s^2'); title('UR10e qddot (J1-3)'); legend('J1','J2','J3');

saveas(fig,outFile);
fprintf('Saved: %s\n\n', outFile);

end
