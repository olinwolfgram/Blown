function [A, B, f0] = aa222_linearize_trim(xTrim, uTrim, params, dx, du)
%AA222_LINEARIZE_TRIM Central finite-difference linearization about trim.

if nargin < 4 || isempty(dx)
    dx = [0.05 0.05 0.05 deg2rad([0.2 0.2 0.2 0.2 0.2 0.2])];
end
if nargin < 5 || isempty(du)
    du = [25 * ones(1, params.n_props), deg2rad([0.1 0.1])];
end

xTrim = xTrim(:);
uTrim = uTrim(:);
[f0, ~] = aa222_trim_dynamics(xTrim, uTrim, params);

nx = numel(xTrim);
nu = numel(uTrim);
A = zeros(numel(f0), nx);
B = zeros(numel(f0), nu);

for i = 1:nx
    step = dx(i);
    xp = xTrim; xm = xTrim;
    xp(i) = xp(i) + step;
    xm(i) = xm(i) - step;
    fp = aa222_trim_dynamics(xp, uTrim, params);
    fm = aa222_trim_dynamics(xm, uTrim, params);
    A(:, i) = (fp - fm) / (2 * step);
end

for i = 1:nu
    step = du(i);
    up = uTrim; um = uTrim;
    up(i) = up(i) + step;
    um(i) = um(i) - step;
    fp = aa222_trim_dynamics(xTrim, up, params);
    fm = aa222_trim_dynamics(xTrim, um, params);
    B(:, i) = (fp - fm) / (2 * step);
end
end
