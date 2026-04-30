%% AA222: Nonlinear Static Trim and Linearization for Blown-Wing Aircraft
clear; clc;

init_params;

target = struct();
target.speed_mps = params.cruise_speed_mps;
target.altitude_m = 500;
target.flight_path_rad = 0;
target.turn_rate_rps = 0;

% Decision vector:
% z = [alpha, beta, phi, theta, rpm_1...rpm_N, elevator, flap]
nProps = params.n_props;
z0 = [deg2rad(2.0), 0, 0, deg2rad(2.0), params.cruise_rpm * ones(1, nProps), 0, 0];
lb = [deg2rad(-6), deg2rad(-8), deg2rad(-20), deg2rad(-15), zeros(1, nProps), ...
      -params.elevator_max, 0];
ub = [deg2rad(18), deg2rad(8), deg2rad(20), deg2rad(20), 14000 * ones(1, nProps), ...
      params.elevator_max, params.flap_deflection_slow];

options = optimoptions('fmincon', ...
    'Display', 'iter-detailed', ...
    'Algorithm', 'interior-point', ...
    'ConstraintTolerance', 1e-6, ...
    'StepTolerance', 1e-8, ...
    'OptimalityTolerance', 1e-6, ...
    'MaxIterations', 250, ...
    'MaxFunctionEvaluations', 40000, ...
    'ScaleProblem', true, ...
    'UseParallel', false);

fprintf('Starting AA222 nonlinear trim solve using fmincon interior-point NLP.\n');
fprintf('Method basis: constrained nonlinear programming with KKT residuals; the interior-point barrier strategy matches Chapter 10 of optimization-1e.pdf for smooth bound/equality constrained problems.\n\n');

[zOpt, Jmin, exitflag, output] = fmincon( ...
    @(z) trim_objective(z, params), ...
    z0, [], [], [], [], lb, ub, ...
    @(z) trim_constraints(z, params, target), ...
    options);

[xTrim, uTrim] = unpack_trim(zOpt, params, target);
[xdotTrim, auxTrim] = aa222_trim_dynamics(xTrim, uTrim, params);
[A, B] = aa222_linearize_trim(xTrim, uTrim, params);

trimResult = struct();
trimResult.z = zOpt;
trimResult.x = xTrim;
trimResult.u = uTrim;
trimResult.cost = Jmin;
trimResult.exitflag = exitflag;
trimResult.output = output;
trimResult.xdot = xdotTrim;
trimResult.aux = auxTrim;
trimResult.A = A;
trimResult.B = B;
trimResult.stateNames = {'u','v','w','p','q','r','phi','theta','psi'};
trimResult.inputNames = [arrayfun(@(k) sprintf('rpm_%02d', k), 1:nProps, 'UniformOutput', false), ...
    {'elevator','flap'}];

save('aa222_trim_linearization.mat', 'params', 'target', 'trimResult');

fprintf('\n--- AA222 Trim Result ---\n');
fprintf('Exit flag:       %d\n', exitflag);
fprintf('Cost:            %.6g\n', Jmin);
fprintf('Alpha/Beta:      %.3f / %.3f deg\n', rad2deg(zOpt(1)), rad2deg(zOpt(2)));
fprintf('Phi/Theta:       %.3f / %.3f deg\n', rad2deg(zOpt(3)), rad2deg(zOpt(4)));
fprintf('RPM range:       %.1f to %.1f rpm\n', min(uTrim(1:nProps)), max(uTrim(1:nProps)));
fprintf('Elev/Flap:       %.3f / %.3f deg\n', rad2deg(uTrim(nProps+1:nProps+2)));
fprintf('Force residual:  [%.3e %.3e %.3e] N\n', auxTrim.force_body_n);
fprintf('Moment residual: [%.3e %.3e %.3e] N-m\n', auxTrim.moment_body_nm);
fprintf('Saved A/B matrices to aa222_trim_linearization.mat\n');

function J = trim_objective(z, params)
    n = params.n_props;
    rpm = z(5:4+n);
    elevator = z(5+n);
    flap = z(6+n);
    rpmMean = mean(rpm);
    rpmSpread = rpm - rpmMean;
    rpmRef = max(params.cruise_rpm, 1);

    energy = mean((rpm / rpmRef).^2);
    differentialThrust = mean((rpmSpread / rpmRef).^2);
    surfaceEffort = (elevator / params.elevator_max)^2 + ...
        0.15 * (flap / max(params.flap_deflection_slow, 1e-6))^2;
    attitudeEffort = 0.05 * (z(3)^2 + z(4)^2);

    J = energy + 0.35 * differentialThrust + 0.08 * surfaceEffort + attitudeEffort;
end

function [c, ceq] = trim_constraints(z, params, target)
    [x, u] = unpack_trim(z, params, target);
    [xdot, aux] = aa222_trim_dynamics(x, u, params);

    accelScale = max(params.g, 1);
    rateScale = 1;
    ceq = [xdot(1:3) / accelScale; xdot(4:6) / rateScale; xdot(7:9)];

    % Keep a small positive local-CL margin by limiting to the capstone
    % flaps-down design local CLmax with the same 1.20 stall factor.
    qS = max(aux.q_eff_pa * params.S, 1e-9);
    clReqProxy = abs(aux.force_aero_body_n(3)) / qS;
    c = clReqProxy - 0.98 * max(params.cl_all_high_lift);
end

function [x, u] = unpack_trim(z, params, target)
    n = params.n_props;
    alpha = z(1);
    beta = z(2);
    phi = z(3);
    theta = z(4);
    speed = target.speed_mps;

    ub = speed * cos(alpha) * cos(beta);
    vb = speed * sin(beta);
    wb = speed * sin(alpha) * cos(beta);

    x = [ub; vb; wb; 0; 0; target.turn_rate_rps; phi; theta; 0];
    u = z(5:6+n).';
end
