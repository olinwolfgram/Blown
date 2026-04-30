function out = aa222_blown_wing_forces(params, state, control)
%AA222_BLOWN_WING_FORCES Sectioned blown-wing force and moment surrogate.
%   state = [u v w p q r phi theta psi]
%   control = [rpm_1 ... rpm_N elevator_rad flap_rad]

u = state(1); v = state(2); w = state(3);
p = state(4); q = state(5); r = state(6);
phi = state(7); theta = state(8);

nProps = params.n_props;
rpm = reshape(control(1:nProps), [], 1);
deltaE = control(nProps + 1);
deltaF = control(nProps + 2);

rho = params.rho0;
Vb = [u; v; w];
Va = max(norm(Vb), 0.1);
alpha = atan2(w, max(u, 1e-6));
beta = asin(max(min(v / Va, 1), -1));
qInf = 0.5 * rho * Va^2;

thrusts = prop_thrust_from_table(params, rpm);
Vslip = slipstream_velocity(params, thrusts, Va);
[Fwing, Mwing, wingDiag] = wing_section_sums(params, Vb, alpha, rho, Vslip, deltaF);
[Ftail, Mtail] = tail_section_sums(params, Vb, [p; q; r], rho, deltaE);

Fprop = [sum(thrusts); 0; 0];
Fg = params.mass * params.g * [-sin(theta); sin(phi) * cos(theta); cos(phi) * cos(theta)];

out = struct();
out.force_body_n = Fwing + Ftail + Fprop + Fg;
out.moment_body_nm = Mwing + Mtail;
out.force_aero_body_n = Fwing + Ftail;
out.force_prop_body_n = Fprop;
out.force_gravity_body_n = Fg;
out.force_wing_body_n = Fwing;
out.force_tail_body_n = Ftail;
out.moment_wing_body_nm = Mwing;
out.moment_tail_body_nm = Mtail;
out.thrusts_n = thrusts;
out.v_slip_mps = Vslip;
out.q_inf_pa = qInf;
out.q_blown_pa = wingDiag.q_blown_pa;
out.q_eff_pa = wingDiag.q_eff_pa;
out.alpha_rad = alpha;
out.beta_rad = beta;
out.section_areas_m2 = wingDiag.section_areas_m2;
end

function thrusts = prop_thrust_from_table(params, rpm)
rpmGrid = params.prop_rpm_grid(:);
thrustGrid = params.prop_thrust_grid(:);
rpmClamped = min(max(rpm, rpmGrid(1)), rpmGrid(end));
thrusts = interp1(rpmGrid, thrustGrid, rpmClamped, 'pchip');
thrusts = max(thrusts, 0);
end

function Vslip = slipstream_velocity(params, thrusts, Va)
area = pi * (params.D_prop / 2)^2;
term = (2 .* max(thrusts, 0)) ./ max(params.rho0 * area, 1e-9);
Vslip = sqrt(max(Va, 0)^2 + term);
end

function [Fwing, Mwing, diag] = wing_section_sums(params, Vb, alpha, rho, Vslip, deltaF)
Fwing = zeros(3, 1);
Mwing = zeros(3, 1);

posMask = params.y_dist(:) > 0;
posCenters = params.y_dist(posMask);
posSlip = Vslip(posMask);
semiSpan = 0.5 * params.b;
flapEnd = params.flap_span_fraction * semiSpan;
blownHalfWidth = 0.5 * params.k_span_expansion * params.D_prop;
edgeList = [0; flapEnd; semiSpan; ...
    max(posCenters - blownHalfWidth, 0); ...
    min(posCenters + blownHalfWidth, semiSpan)];
edgeList = unique(sort(edgeList));

qInf = 0.5 * rho * max(norm(Vb), 0.1)^2;
flapMix = min(max(abs(deltaF) / max(params.flap_deflection_slow, 1e-6), 0), 1);

blownArea = 0;
qBlownWeighted = 0;
allArea = 0;

for i = 1:(numel(edgeList) - 1)
    y0 = edgeList(i);
    y1 = edgeList(i + 1);
    width = y1 - y0;
    if width <= 1e-6
        continue;
    end

    yMid = 0.5 * (y0 + y1);
    isFlapRegion = yMid <= flapEnd + 1e-9;
    overlapMask = (posCenters + blownHalfWidth > y0) & (posCenters - blownHalfWidth < y1);
    isBlownRegion = any(overlapMask);

    qLocal = qInf;
    if isBlownRegion
        vLocal = mean(posSlip(overlapMask));
        qLocal = 0.5 * rho * vLocal^2;
        blownArea = blownArea + 2 * params.c * width;
        qBlownWeighted = qBlownWeighted + qLocal * (2 * params.c * width);
    end
    allArea = allArea + 2 * params.c * width;

    [CL, CD, Cm] = select_section_coefficients(params, alpha, flapMix, isBlownRegion, isFlapRegion);
    areaStrip = params.c * width;
    [Fplus, Mplus] = strip_force_moment(params, alpha, qLocal, areaStrip, yMid, CL, CD, Cm);
    [Fminus, Mminus] = strip_force_moment(params, alpha, qLocal, areaStrip, -yMid, CL, CD, Cm);

    Fwing = Fwing + Fplus + Fminus;
    Mwing = Mwing + Mplus + Mminus;
end

diag = struct();
diag.q_blown_pa = qBlownWeighted / max(blownArea, 1e-9);
diag.q_eff_pa = ((max(allArea - blownArea, 0) * qInf) + qBlownWeighted) / max(allArea, 1e-9);
diag.section_areas_m2 = [max(allArea - blownArea, 0); blownArea];
end

function [Fbody, Mbody] = strip_force_moment(params, alpha, qLocal, areaStrip, y, CL, CD, Cm)
ca = cos(alpha);
sa = sin(alpha);
Rsb = [ca, 0, -sa; 0, 1, 0; sa, 0, ca];
Fstab = qLocal * areaStrip * [-CD; 0; -CL];
Fbody = Rsb * Fstab;
Maero = qLocal * areaStrip * [0; Cm * params.c; 0];
rVec = [params.prop_axial_x_m - params.cg_x_m; y; 0];
Mbody = Maero + cross(rVec, Fbody);
end

function [CL, CD, Cm] = select_section_coefficients(params, alpha, flapMix, isBlownRegion, isFlapRegion)
alphaDeg = rad2deg(alpha);
if isBlownRegion
    clNoFlap = interp1(params.alpha_deg_grid, params.cl_blow_only, alphaDeg, 'pchip', 'extrap');
    cdNoFlap = interp1(params.alpha_deg_grid, params.cd_blow_only, alphaDeg, 'pchip', 'extrap');
    cmNoFlap = interp1(params.alpha_deg_grid, params.cm_blow_only, alphaDeg, 'pchip', 'extrap');
    clFlap = interp1(params.alpha_deg_grid, params.cl_all_high_lift, alphaDeg, 'pchip', 'extrap');
    cdFlap = interp1(params.alpha_deg_grid, params.cd_all_high_lift, alphaDeg, 'pchip', 'extrap');
    cmFlap = interp1(params.alpha_deg_grid, params.cm_all_high_lift, alphaDeg, 'pchip', 'extrap');
else
    clNoFlap = interp1(params.alpha_deg_grid, params.cl_clean_baseline, alphaDeg, 'pchip', 'extrap');
    cdNoFlap = interp1(params.alpha_deg_grid, params.cd_clean_baseline, alphaDeg, 'pchip', 'extrap');
    cmNoFlap = interp1(params.alpha_deg_grid, params.cm_clean_baseline, alphaDeg, 'pchip', 'extrap');
    clFlap = interp1(params.alpha_deg_grid, params.cl_flap_only, alphaDeg, 'pchip', 'extrap');
    cdFlap = interp1(params.alpha_deg_grid, params.cd_flap_only, alphaDeg, 'pchip', 'extrap');
    cmFlap = interp1(params.alpha_deg_grid, params.cm_flap_only, alphaDeg, 'pchip', 'extrap');
end

if isFlapRegion
    CL = clNoFlap + flapMix * (clFlap - clNoFlap);
    CD = cdNoFlap + flapMix * (cdFlap - cdNoFlap);
    Cm = cmNoFlap + flapMix * (cmFlap - cmNoFlap);
else
    CL = clNoFlap;
    CD = cdNoFlap;
    Cm = cmNoFlap;
end
end

function [Ftail, Mtail] = tail_section_sums(params, Vb, Omega, rho, deltaE)
rTail = [-params.lt; 0; 0];
Vtail = Vb + cross(Omega, rTail);
uT = Vtail(1);
wT = Vtail(3);
VaT = sqrt(sum(Vtail.^2)) + 0.1;
alphaT = atan2(wT, uT);
qTail = 0.5 * rho * VaT^2;

CL = params.CL_alpha_ht * alphaT + params.CL_delta_e * deltaE;
CD = 0.02 + 0.06 * CL^2;
Cm = -0.15 * deltaE;

ca = cos(alphaT);
sa = sin(alphaT);
Rsb = [ca, 0, -sa; 0, 1, 0; sa, 0, ca];
Fstab = qTail * params.S_ht * [-CD; 0; -CL];
Ftail = Rsb * Fstab;
Mtail = qTail * params.S_ht * [0; Cm * params.c_ht; 0] + cross(rTail, Ftail);
end
