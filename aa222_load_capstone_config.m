function params = aa222_load_capstone_config(capstoneRoot)
%AA222_LOAD_CAPSTONE_CONFIG Load the frozen AA146 vehicle for controls work.
%   The AA146 optimizer remains the source of truth for geometry and
%   propulsion architecture; this function translates its CSV artifacts into
%   the params struct used by the AA222 control/trim scripts.

if nargin < 1 || isempty(capstoneRoot)
    capstoneRoot = fullfile(pwd, 'AA146-Capstone');
end

outputsRoot = fullfile(capstoneRoot, 'outputs');
topDesignCsv = fullfile(outputsRoot, 'stage3_aerosandbox_top_designs.csv');
stage2Csv = fullfile(outputsRoot, 'stage2_prop_span_report.csv');

assert(isfile(topDesignCsv), 'Missing capstone Stage 3 CSV: %s', topDesignCsv);
assert(isfile(stage2Csv), 'Missing capstone Stage 2 CSV: %s', stage2Csv);

stage3 = readtable(topDesignCsv, 'TextType', 'string');
stage3 = stage3(stage3.status == "SUCCESS", :);
assert(height(stage3) >= 1, 'No successful Stage 3 design found in %s.', topDesignCsv);
design = stage3(1, :);

stage2 = readtable(stage2Csv, 'TextType', 'string');
match = stage2.n_props == design.n_props & ...
    abs(stage2.prop_diameter_in - design.prop_diameter_in) < 1e-9 & ...
    abs(stage2.prop_pitch_ratio - design.prop_pitch_ratio) < 1e-9 & ...
    stage2.prop_family == design.prop_family;
assert(any(match), 'No matching Stage 2 propulsion row for selected Stage 3 design.');
propRow = stage2(find(match, 1), :);

params = struct();

params.rho0 = 1.225;
params.g = 9.80665;

params.mass = design.gross_flight_mass_kg;
params.S = design.wing_area_m2;
params.b = design.wing_span_m;
params.c = design.wing_chord_m;
params.AR = design.wing_aspect_ratio;
params.oswald_e = 0.6591;
params.cg_x_m = design.cg_x_m;
params.static_margin_mac = design.static_margin_mac;

params.S_ht = design.htail_area_m2;
params.b_ht = design.htail_span_m;
params.c_ht = 0.5 * (design.htail_root_chord_m + design.htail_tip_chord_m);
params.lt = design.tail_arm_m;
params.CL_alpha_ht = 3.5;
params.CL_delta_e = -1.5;
params.elevator_max = deg2rad(design.elevator_max_deflection_deg);

params.S_vt = design.vtail_area_m2;
params.b_vt = design.vtail_span_m;
params.c_vt = 0.5 * (design.vtail_root_chord_m + design.vtail_tip_chord_m);
params.lv = design.tail_arm_m;
params.zv = 0.15;
params.CL_delta_r = 0.8;
params.rudder_max = deg2rad(design.rudder_max_deflection_deg);

params.n_props = design.n_props;
params.D_prop = design.prop_diameter_in * 0.0254;
params.prop_pitch_ratio = design.prop_pitch_ratio;
params.prop_family = char(design.prop_family);
params.y_dist = parse_semicolon_vector(propRow.prop_centers_m);
params.blown_span_fraction = propRow.blown_span_fraction;
params.S_blown = design.slow_flight_blown_area_m2;
params.S_unblown = design.slow_flight_unblown_area_m2;
params.S_blown_seg = params.S_blown / max(params.n_props, 1);
params.k_span_expansion = 0.8;
params.prop_axial_x_m = design.prop_axial_x_m;
params.prop_drop_m = 0.12 * params.c;
params.prop_drop_fraction_of_chord = 0.12;

params.low_speed_rpm = design.low_speed_rpm;
params.cruise_rpm = design.cruise_rpm;
params.low_speed_veff_mps = design.low_speed_actual_veff_mps;
params.cruise_veff_mps = design.cruise_blown_effective_velocity_mps;
params.cruise_speed_mps = design.cruise_blown_effective_velocity_mps * 0 + 10.0;
params.slow_speed_mps = design.stage3_slow_flight_speed_mps;
params.flap_span_fraction = design.flap_span_fraction;
params.flap_chord_fraction = design.flap_chord_fraction;
params.flap_deflection_slow = deg2rad(design.flap_deflection_slow_deg);
params.aileron_span_fraction = design.aileron_span_fraction;
params.aileron_chord_fraction = design.aileron_chord_fraction;

params.Ixx = 0.5;
params.Iyy = 0.8;
params.Izz = 1.2;
params.Ixz = 0.05;
params.InertiaMatrix = [params.Ixx, 0, -params.Ixz; ...
                        0, params.Iyy, 0; ...
                        -params.Ixz, 0, params.Izz];

propTable = load_prop_table(outputsRoot, design);
params.prop_rpm_grid = propTable.rpm(:);
params.prop_thrust_grid = propTable.thrust_n(:);
params.prop_ct_grid = propTable.ct_static(:);
params.prop_cp_grid = propTable.cp_static(:);

aeroTable = load_aero_table(outputsRoot);
params.alpha_deg_grid = aeroTable.alpha_deg(:);
params.cl_clean_baseline = aeroTable.cl_clean_baseline(:);
params.cl_blow_only = aeroTable.cl_blow_only(:);
params.cl_flap_only = aeroTable.cl_flap_only(:);
params.cl_all_high_lift = aeroTable.cl_all_high_lift(:);
params.cd_clean_baseline = aeroTable.cd_clean_baseline(:);
params.cd_blow_only = aeroTable.cd_blow_only(:);
params.cd_flap_only = aeroTable.cd_flap_only(:);
params.cd_all_high_lift = aeroTable.cd_all_high_lift(:);
params.cm_clean_baseline = aeroTable.cm_clean_baseline(:);
params.cm_blow_only = aeroTable.cm_blow_only(:);
params.cm_flap_only = aeroTable.cm_flap_only(:);
params.cm_all_high_lift = aeroTable.cm_all_high_lift(:);

params.alpha_range = deg2rad(params.alpha_deg_grid(:)');
params.CL_data = params.cl_clean_baseline(:)';
params.CD_data = params.cd_clean_baseline(:)';
params.CM_data = params.cm_clean_baseline(:)';
params.CL_flap = 1.2;
params.Cm_elev = -1.5;
params.Cm_q = -18.0;
params.CY_beta = -0.6;
params.Cl_beta = -0.08;
params.Cn_beta = 0.08;
params.Cl_p = -0.45;
params.Cn_r = -0.20;
end

function v = parse_semicolon_vector(s)
parts = split(string(s), ';');
v = str2double(parts(:));
end

function propTable = load_prop_table(outputsRoot, design)
rel = string(design.ecalc_static_csv);
candidate = fullfile(outputsRoot, erase(rel, "outputs/"));
if ~isfile(candidate)
    files = dir(fullfile(outputsRoot, 'ecalc_prop_analysis', '**', 'ecalc_static_partial_load.csv'));
    assert(~isempty(files), 'Could not find eCalc static prop table.');
    candidate = fullfile(files(1).folder, files(1).name);
end
propTable = readtable(candidate);
end

function aeroTable = load_aero_table(outputsRoot)
files = dir(fullfile(outputsRoot, 'control_surface_sizing', '**', 'dae51', 'total_cl_curve.csv'));
if isempty(files)
    files = dir(fullfile(outputsRoot, 'wing_workflow', '**', 'dae51', 'total_cl_curve.csv'));
end
assert(~isempty(files), 'Could not find DAE51 total_cl_curve.csv.');
aeroTable = readtable(fullfile(files(1).folder, files(1).name));
end
