%% AA222: Blown Wing Aircraft Parameters
% This script keeps the AA222 control model synchronized with the frozen
% AA146 capstone vehicle configuration.

capstoneRoot = fullfile(pwd, 'AA146-Capstone');
params = aa222_load_capstone_config(capstoneRoot);

%% Initial States for 6DOF / trim studies
initial_pos_ned = [0, 0, -500];
initial_vel_body = [params.cruise_speed_mps, 0, 0];
initial_euler = [0, 0, 0];
initial_rates = [0, 0, 0];

% Legacy scalar aliases still referenced by some Simulink blocks.
lt = params.lt;
D_prop = params.D_prop;
prop_rpm_grid = params.prop_rpm_grid;
prop_thrust_grid = params.prop_thrust_grid;
alpha_deg_grid = params.alpha_deg_grid;
cl_clean_baseline = params.cl_clean_baseline;
cl_blow_only = params.cl_blow_only;
cl_flap_only = params.cl_flap_only;
cl_all_high_lift = params.cl_all_high_lift;
cd_clean_baseline = params.cd_clean_baseline;
cd_blow_only = params.cd_blow_only;
cd_flap_only = params.cd_flap_only;
cd_all_high_lift = params.cd_all_high_lift;
cm_clean_baseline = params.cm_clean_baseline;
cm_blow_only = params.cm_blow_only;
cm_flap_only = params.cm_flap_only;
cm_all_high_lift = params.cm_all_high_lift;
y_dist = params.y_dist;
b = params.b;
c = params.c;
cg_x_m = params.cg_x_m;
prop_axial_x_m = params.prop_axial_x_m;
flap_span_fraction = params.flap_span_fraction;
flap_deflection_slow = params.flap_deflection_slow;
k_span_expansion = params.k_span_expansion;
S_ht = params.S_ht;
c_ht = params.c_ht;
b_ht = params.b_ht;
CL_alpha_ht = params.CL_alpha_ht;
CL_delta_e = params.CL_delta_e;

%% Optimizer Variables
% x = [RPM1...RPM10, Elevator_rad, Flap_rad, Rudder_rad]
% x = [RPM1...RPM10, Elevator_rad, Flap_rad]
x0 = [params.cruise_rpm * ones(1, params.n_props), 0, 0];
lb = [zeros(1, params.n_props), -params.elevator_max, 0];
ub = [14000 * ones(1, params.n_props), params.elevator_max, params.flap_deflection_slow];

disp('AA222 parameters loaded from the frozen AA146 capstone vehicle configuration.');
