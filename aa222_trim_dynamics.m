function [xdot, aux] = aa222_trim_dynamics(x, u, params)
%AA222_TRIM_DYNAMICS Rigid-body body-axis dynamics for trim/linearization.
%   x = [u v w p q r phi theta psi]

aux = aa222_blown_wing_forces(params, x, u);

V = x(1:3);
omega = x(4:6);
phi = x(7);
theta = x(8);

vdot = aux.force_body_n / params.mass - cross(omega, V);
omegaDot = params.InertiaMatrix \ (aux.moment_body_nm - cross(omega, params.InertiaMatrix * omega));

T = [1, sin(phi) * tan(theta), cos(phi) * tan(theta); ...
     0, cos(phi), -sin(phi); ...
     0, sin(phi) / cos(theta), cos(phi) / cos(theta)];
eulerDot = T * omega;

xdot = [vdot; omegaDot; eulerDot];
end
