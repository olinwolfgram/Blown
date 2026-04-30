function J = compute_cost(control)
%COMPUTE_COST Trim residual cost for a candidate AA222 control vector.
%   control = [RPM1...RPM10, elevator_rad, flap_rad, rudder_rad]

if evalin('base', 'exist(''params'', ''var'')')
    params = evalin('base', 'params');
else
    params = aa222_load_capstone_config(fullfile(pwd, 'AA146-Capstone'));
end

targetSpeed = params.cruise_speed_mps;
alpha0 = deg2rad(2.0);
state = [targetSpeed * cos(alpha0); 0; targetSpeed * sin(alpha0); 0; 0; 0; 0; alpha0; 0];

[xdot, aux] = aa222_trim_dynamics(state, control(:), params);

accelCost = sum((xdot(1:3) / max(params.g, 1)).^2);
rateCost = sum(xdot(4:6).^2);
attitudeCost = sum(xdot(7:9).^2);
rpmCost = mean((control(1:params.n_props) / max(params.cruise_rpm, 1)).^2);
surfaceCost = sum(control(params.n_props+1:end).^2);

J = 1e4 * accelCost + 1e4 * rateCost + 1e3 * attitudeCost + rpmCost + surfaceCost;

if any(~isfinite(xdot)) || any(~isfinite(aux.force_body_n)) || any(~isfinite(aux.moment_body_nm))
    J = 1e12;
end
end
