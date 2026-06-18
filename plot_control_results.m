function plot_control_results(t3,r3n,r3e,inj3,t10,r10n,r10e,inj10,outA,outB)

fig = figure('Name','Controller Results','Position',[240 100 1400 820]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(t3,rad2deg(r3n.errN),'LineWidth',1.4); hold on;
plot(t3,rad2deg(r3e.errN),'LineWidth',1.4);
xline(inj3,'--','Injected','LineWidth',1.4);
grid on; xlabel('t (s)'); ylabel('deg'); title('UR3 Tracking Error');
legend('Normal','Disturbed','Location','best');

nexttile;
plot(t3,r3e.u(:,1:3),'LineWidth',1.1); hold on;
xline(inj3,'--','Injected','LineWidth',1.4);
grid on; xlabel('t (s)'); ylabel('u (arb)'); title('UR3 Control Effort (J1-3)');
legend('J1','J2','J3','Location','best');

nexttile;
plot(t10,rad2deg(r10n.errN),'LineWidth',1.4); hold on;
plot(t10,rad2deg(r10e.errN),'LineWidth',1.4);
xline(inj10,'--','Injected','LineWidth',1.4);
grid on; xlabel('t (s)'); ylabel('deg'); title('UR10e Tracking Error');
legend('Normal','Disturbed','Location','best');

nexttile;
plot(t10,r10e.u(:,1:3),'LineWidth',1.1); hold on;
xline(inj10,'--','Injected','LineWidth',1.4);
grid on; xlabel('t (s)'); ylabel('u (arb)'); title('UR10e Control Effort (J1-3)');
legend('J1','J2','J3','Location','best');

saveas(fig,outA);
fprintf('Saved: %s\n', outA);

% Recovery zoom (Joint 2 as an example)
fig2 = figure('Name','Recovery Detail','Position',[320 140 1200 520]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
mask = (t3>=2.0 & t3<=4.5);
plot(t3(mask),rad2deg(r3e.q_ref(mask,2)),'LineWidth',1.8); hold on;
plot(t3(mask),rad2deg(r3e.q(mask,2)),'--','LineWidth',1.6);
xline(inj3,'--','Injected','LineWidth',1.4);
grid on; xlabel('t (s)'); ylabel('deg'); title('UR3 Joint 2 Recovery');
legend('Desired','Actual','Location','best');

nexttile;
mask = (t10>=2.5 & t10<=5.0);
plot(t10(mask),rad2deg(r10e.q_ref(mask,2)),'LineWidth',1.8); hold on;
plot(t10(mask),rad2deg(r10e.q(mask,2)),'--','LineWidth',1.6);
xline(inj10,'--','Injected','LineWidth',1.4);
grid on; xlabel('t (s)'); ylabel('deg'); title('UR10e Joint 2 Recovery');
legend('Desired','Actual','Location','best');

saveas(fig2,outB);
fprintf('Saved: %s\n\n', outB);

end
