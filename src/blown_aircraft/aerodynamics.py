from __future__ import annotations

import math

import numpy as np

from .geometry import Vehicle
from .propulsion import prop_positions_body, thrust_per_prop_n


def dynamic_pressure(rho_kgpm3: float, speed_mps: float) -> float:
    return 0.5 * rho_kgpm3 * speed_mps**2


def airdata_from_body_velocity(u: float, v: float, w: float) -> dict[str, float]:
    speed = float(np.sqrt(u**2 + v**2 + w**2))
    alpha = float(np.arctan2(w, max(u, 1e-6)))
    beta = float(np.arcsin(np.clip(v / max(speed, 1e-6), -1.0, 1.0)))
    return {"V": speed, "alpha": alpha, "beta": beta}


def semispan_strip_centers(vehicle: Vehicle) -> np.ndarray:
    b = float(vehicle.geometry["span_m"])
    n_half = int(vehicle.blown["n_strips_per_semispan"])
    ys = np.linspace(0.0, 0.5 * b, n_half + 1)
    centers = 0.5 * (ys[:-1] + ys[1:])
    return np.concatenate([-centers[::-1], centers])


def strip_width_m(vehicle: Vehicle) -> float:
    b = float(vehicle.geometry["span_m"])
    n_half = int(vehicle.blown["n_strips_per_semispan"])
    return 0.5 * b / n_half


def strip_blown_halfwidth_m(vehicle: Vehicle) -> float:
    return 0.5 * float(vehicle.blown["wake_span_expansion_factor"]) * float(vehicle.propulsion["diameter_m"])


def strip_is_blown(y_strip: float, vehicle: Vehicle) -> bool:
    halfwidth = strip_blown_halfwidth_m(vehicle)
    prop_y = np.asarray(vehicle.propulsion["centers_y_m"], dtype=float)
    return bool(np.any(np.abs(prop_y - y_strip) <= halfwidth))


def strip_family_name(blown_active: bool, flap_active: bool) -> str:
    if blown_active and flap_active:
        return "blown_flap"
    if blown_active and not flap_active:
        return "blown_clean"
    if (not blown_active) and flap_active:
        return "clean_flap"
    return "clean_clean"


def blown_wake_ratio_at_strip(y_strip: float, u_axial: float, rpm_vec: np.ndarray, vehicle: Vehicle) -> float:
    prop_positions = prop_positions_body(vehicle)
    rho = vehicle.rho_kgpm3
    d_prop = float(vehicle.propulsion["diameter_m"])
    disk_area_single = math.pi * (0.5 * d_prop) ** 2
    span_sigma = 0.5 * float(vehicle.blown["wake_span_expansion_factor"]) * d_prop
    x_decay = math.exp(
        -float(vehicle.blown["prop_to_wing_leading_edge_m"]) / max(float(vehicle.blown["wake_decay_length_m"]), 1e-6)
    )

    wake_q_increment = 0.0
    for idx, pos in enumerate(prop_positions):
        thrust = thrust_per_prop_n(float(rpm_vec[idx]), vehicle)
        v_induced = math.sqrt(max(thrust, 0.0) / (2.0 * rho * disk_area_single))
        v_wake = u_axial + 2.0 * v_induced
        ratio = max((v_wake / max(u_axial, 1e-3)) ** 2 - 1.0, 0.0)
        span_weight = math.exp(-0.5 * ((y_strip - pos[1]) / max(span_sigma, 1e-6)) ** 2)
        wake_q_increment += ratio * span_weight * x_decay
    return float(wake_q_increment)


def wing_strip_forces_and_moments(
    state: np.ndarray,
    control: np.ndarray,
    vehicle: Vehicle,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute wing aerodynamic forces and moments from spanwise strips.

    This is a compact blown-wing strip model:
    - body velocity defines section alpha,
    - each strip sees a local wake ratio based on nearby prop disks,
    - flapped strips receive additional blown-lift and blown-pitch increments,
    - forces are summed in body axes and moments are accumulated about the CG.
    """

    _, _, _, u, v, w, phi, theta, psi, p, q, r = state
    n_props = int(vehicle.propulsion["n_props"])
    rpm_vec = np.asarray(control[:n_props], dtype=float)
    de, da, dr, df = np.asarray(control[n_props : n_props + 4], dtype=float)

    air = airdata_from_body_velocity(u, v, w)
    V = air["V"]
    alpha = air["alpha"]
    beta = air["beta"]
    qbar = dynamic_pressure(vehicle.rho_kgpm3, V)
    strip_y = semispan_strip_centers(vehicle)
    dy = strip_width_m(vehicle)
    c = float(vehicle.geometry["chord_m"])
    x_ac = float(vehicle.geometry["wing_ac_x_m"]) - float(vehicle.geometry["cg_x_m"])
    flap_halfspan = 0.5 * float(vehicle.geometry["span_m"]) * float(vehicle.geometry["flap_span_fraction"])

    aero = vehicle.aero
    total_force = np.zeros(3, dtype=float)
    total_moment = np.zeros(3, dtype=float)
    strip_records = []
    family_totals = {
        "clean_clean": {
            "area_m2": 0.0,
            "force_body_n": np.zeros(3, dtype=float),
            "moment_body_nm": np.zeros(3, dtype=float),
            "q_local_pa": [],
            "cl": [],
            "cd": [],
            "cm": [],
        },
        "clean_flap": {
            "area_m2": 0.0,
            "force_body_n": np.zeros(3, dtype=float),
            "moment_body_nm": np.zeros(3, dtype=float),
            "q_local_pa": [],
            "cl": [],
            "cd": [],
            "cm": [],
        },
        "blown_clean": {
            "area_m2": 0.0,
            "force_body_n": np.zeros(3, dtype=float),
            "moment_body_nm": np.zeros(3, dtype=float),
            "q_local_pa": [],
            "cl": [],
            "cd": [],
            "cm": [],
        },
        "blown_flap": {
            "area_m2": 0.0,
            "force_body_n": np.zeros(3, dtype=float),
            "moment_body_nm": np.zeros(3, dtype=float),
            "q_local_pa": [],
            "cl": [],
            "cd": [],
            "cm": [],
        },
    }
    flap_pitch_total = 0.0
    blown_pitch_total = 0.0
    arm_pitch_total = 0.0
    section_pitch_total = 0.0
    cm0_pitch_total = 0.0
    cm_alpha_pitch_total = 0.0
    cm_q_pitch_total = 0.0
    parasite_profile_drag_total = 0.0
    induced_drag_total = 0.0
    blown_drag_total = 0.0
    flap_drag_total = 0.0

    for y_i in strip_y:
        area_strip = c * dy
        flap_active = abs(y_i) <= flap_halfspan
        blown_active = strip_is_blown(y_i, vehicle)
        family_name = strip_family_name(blown_active, flap_active)
        u_axial = max(V * math.cos(alpha), 0.5)
        wake_ratio_raw = blown_wake_ratio_at_strip(y_i, u_axial, rpm_vec, vehicle)
        wake_ratio = wake_ratio_raw if blown_active else 0.0
        q_local = qbar * (1.0 + wake_ratio)

        qhat = q * c / max(2.0 * V, 1e-6)
        alpha_eff = alpha + math.radians(float(vehicle.geometry["wing_incidence_deg"]))

        cl_clean = aero["cl0"] + aero["cl_alpha_per_rad"] * alpha_eff + aero["cl_q"] * qhat
        cl_flap = aero["cl_flap_per_rad"] * df if flap_active else 0.0
        cl_blown = float(vehicle.blown["lift_gain"]) * wake_ratio
        cl_blown_flap = float(vehicle.blown["flap_lift_gain"]) * df * wake_ratio if flap_active else 0.0
        cl = cl_clean + cl_flap + cl_blown + cl_blown_flap

        cd_parasite_coeff = aero["cd0"]
        cd_induced_coeff = (1.0 / (math.pi * float(vehicle.geometry["oswald_e"]) * float(vehicle.geometry["aspect_ratio"]))) * cl**2
        cd_blown_coeff = float(vehicle.blown["drag_gain"]) * wake_ratio
        cd_flap_coeff = (aero["cd_flap_per_rad2"] * df**2 if flap_active else 0.0) + (
            float(vehicle.blown.get("flap_drag_gain", 0.0)) * df * wake_ratio if flap_active else 0.0
        )
        cd = cd_parasite_coeff + cd_induced_coeff + cd_blown_coeff + cd_flap_coeff

        cm0_coeff = aero["cm0"]
        cm_alpha_coeff = aero["cm_alpha_per_rad"] * alpha
        cm_q_coeff = aero["cm_q"] * qhat
        cm_flap_coeff = aero["cm_flap_per_rad"] * df if flap_active else 0.0
        cm_blown_coeff = float(vehicle.blown["pitch_gain"]) * wake_ratio
        cm_blown_flap_coeff = float(vehicle.blown["flap_pitch_gain"]) * df * wake_ratio if flap_active else 0.0
        cm = cm0_coeff + cm_alpha_coeff + cm_q_coeff + cm_flap_coeff + cm_blown_coeff + cm_blown_flap_coeff

        lift = q_local * area_strip * cl
        parasite_profile_drag = q_local * area_strip * cd_parasite_coeff
        induced_drag = q_local * area_strip * cd_induced_coeff
        blown_drag = q_local * area_strip * cd_blown_coeff
        flap_drag = q_local * area_strip * cd_flap_coeff
        drag = parasite_profile_drag + induced_drag + blown_drag + flap_drag
        fx = -drag * math.cos(alpha) + lift * math.sin(alpha)
        fz = -drag * math.sin(alpha) - lift * math.cos(alpha)
        force_i = np.array([fx, 0.0, fz], dtype=float)

        cm0_pitch_i = q_local * area_strip * c * cm0_coeff
        cm_alpha_pitch_i = q_local * area_strip * c * cm_alpha_coeff
        cm_q_pitch_i = q_local * area_strip * c * cm_q_coeff
        flap_pitch_i = q_local * area_strip * c * cm_flap_coeff
        blown_pitch_i = q_local * area_strip * c * (cm_blown_coeff + cm_blown_flap_coeff)
        pitch_ac = cm0_pitch_i + cm_alpha_pitch_i + cm_q_pitch_i + flap_pitch_i + blown_pitch_i
        arm_moment_i = np.cross(np.array([x_ac, y_i, 0.0], dtype=float), force_i)
        moment_i = arm_moment_i + np.array([0.0, pitch_ac, 0.0])

        total_force += force_i
        total_moment += moment_i
        flap_pitch_total += flap_pitch_i
        blown_pitch_total += blown_pitch_i
        arm_pitch_total += arm_moment_i[1]
        section_pitch_total += pitch_ac
        cm0_pitch_total += cm0_pitch_i
        cm_alpha_pitch_total += cm_alpha_pitch_i
        cm_q_pitch_total += cm_q_pitch_i
        parasite_profile_drag_total += parasite_profile_drag
        induced_drag_total += induced_drag
        blown_drag_total += blown_drag
        flap_drag_total += flap_drag
        family_totals[family_name]["area_m2"] += float(area_strip)
        family_totals[family_name]["force_body_n"] += force_i
        family_totals[family_name]["moment_body_nm"] += moment_i
        family_totals[family_name]["q_local_pa"].append(float(q_local))
        family_totals[family_name]["cl"].append(float(cl))
        family_totals[family_name]["cd"].append(float(cd))
        family_totals[family_name]["cm"].append(float(cm))
        strip_records.append(
            {
                "y_m": float(y_i),
                "family": family_name,
                "blown": bool(blown_active),
                "wake_ratio_raw": float(wake_ratio_raw),
                "wake_ratio": float(wake_ratio),
                "q_local_pa": float(q_local),
                "area_m2": float(area_strip),
                "cl": float(cl),
                "cd": float(cd),
                "cm": float(cm),
                "flap": bool(flap_active),
                "lift_n": float(lift),
                "drag_n": float(drag),
                "drag_components_n": {
                    "parasite_profile": float(parasite_profile_drag),
                    "induced": float(induced_drag),
                    "blown": float(blown_drag),
                    "flap": float(flap_drag),
                },
                "force_body_n": force_i.tolist(),
                "arm_pitch_moment_nm": float(arm_moment_i[1]),
                "section_pitch_moment_nm": float(pitch_ac),
                "section_pitch_components_nm": {
                    "cm0": float(cm0_pitch_i),
                    "cm_alpha": float(cm_alpha_pitch_i),
                    "cm_q": float(cm_q_pitch_i),
                    "flap": float(flap_pitch_i),
                    "blown": float(blown_pitch_i),
                },
                "moment_body_nm": moment_i.tolist(),
            }
        )

    family_diag = {}
    for name, accum in family_totals.items():
        q_vals = np.asarray(accum["q_local_pa"], dtype=float)
        cl_vals = np.asarray(accum["cl"], dtype=float)
        cd_vals = np.asarray(accum["cd"], dtype=float)
        cm_vals = np.asarray(accum["cm"], dtype=float)
        family_diag[name] = {
            "area_m2": float(accum["area_m2"]),
            "force_body_n": accum["force_body_n"].tolist(),
            "moment_body_nm": accum["moment_body_nm"].tolist(),
            "n_strips": int(len(q_vals)),
            "q_local_mean_pa": float(np.mean(q_vals)) if len(q_vals) else 0.0,
            "q_local_min_pa": float(np.min(q_vals)) if len(q_vals) else 0.0,
            "q_local_max_pa": float(np.max(q_vals)) if len(q_vals) else 0.0,
            "cl_mean": float(np.mean(cl_vals)) if len(cl_vals) else 0.0,
            "cd_mean": float(np.mean(cd_vals)) if len(cd_vals) else 0.0,
            "cm_mean": float(np.mean(cm_vals)) if len(cm_vals) else 0.0,
        }

    return total_force, total_moment, {
        "strips": strip_records,
        "families": family_diag,
        "alpha_rad": alpha,
        "beta_rad": beta,
        "qbar_pa": qbar,
        "force_body_n": total_force.tolist(),
        "moment_body_nm": total_moment.tolist(),
        "arm_pitch_moment_nm": float(arm_pitch_total),
        "section_pitch_moment_nm": float(section_pitch_total),
        "section_pitch_components_nm": {
            "cm0": float(cm0_pitch_total),
            "cm_alpha": float(cm_alpha_pitch_total),
            "cm_q": float(cm_q_pitch_total),
            "flap": float(flap_pitch_total),
            "blown": float(blown_pitch_total),
            "total": float(cm0_pitch_total + cm_alpha_pitch_total + cm_q_pitch_total + flap_pitch_total + blown_pitch_total),
        },
        "flap_pitch_moment_nm": float(flap_pitch_total),
        "blown_pitch_moment_nm": float(blown_pitch_total),
        "drag_components_n": {
            "parasite_profile": float(parasite_profile_drag_total),
            "induced": float(induced_drag_total),
            "blown": float(blown_drag_total),
            "flap": float(flap_drag_total),
            "total": float(parasite_profile_drag_total + induced_drag_total + blown_drag_total + flap_drag_total),
        },
    }


def tail_forces_and_moments(
    state: np.ndarray,
    control: np.ndarray,
    vehicle: Vehicle,
) -> tuple[np.ndarray, np.ndarray, dict]:
    _, _, _, u, v, w, phi, theta, psi, p, q, r = state
    n_props = int(vehicle.propulsion["n_props"])
    de = float(control[n_props])

    air = airdata_from_body_velocity(u, v, w)
    V = air["V"]
    alpha = air["alpha"]
    aero = vehicle.aero

    downwash = aero["downwash_gradient"] * alpha
    alpha_tail = alpha + math.radians(float(vehicle.geometry["htail_incidence_deg"])) - downwash
    cl_tail = aero["tail_cl_alpha_per_rad"] * alpha_tail + aero["tail_cl_de_per_rad"] * de
    qbar = dynamic_pressure(vehicle.rho_kgpm3, V)
    lift_tail = qbar * float(vehicle.geometry["htail_area_m2"]) * cl_tail
    drag_tail = qbar * float(vehicle.geometry["htail_area_m2"]) * aero["tail_cd0"]

    fx = -drag_tail * math.cos(alpha_tail) + lift_tail * math.sin(alpha_tail)
    fz = -drag_tail * math.sin(alpha_tail) - lift_tail * math.cos(alpha_tail)
    force = np.array([fx, 0.0, fz], dtype=float)

    x_tail = float(vehicle.geometry["tail_arm_m"])
    moment = np.cross(np.array([x_tail, 0.0, 0.0], dtype=float), force)
    return force, moment, {
        "alpha_tail_rad": float(alpha_tail),
        "cl_tail": float(cl_tail),
        "drag_tail_n": float(drag_tail),
        "lift_tail_n": float(lift_tail),
        "force_body_n": force.tolist(),
        "moment_body_nm": moment.tolist(),
    }


def aerodynamic_forces_and_moments(
    state: np.ndarray,
    control: np.ndarray,
    vehicle: Vehicle,
) -> tuple[np.ndarray, np.ndarray, dict]:
    _, _, _, u, v, w, _, _, _, p, q, r = np.asarray(state, dtype=float)
    n_props = int(vehicle.propulsion["n_props"])
    _, da, dr, _ = np.asarray(control[n_props : n_props + 4], dtype=float)

    wing_force, wing_moment, wing_diag = wing_strip_forces_and_moments(state, control, vehicle)
    tail_force, tail_moment, tail_diag = tail_forces_and_moments(state, control, vehicle)
    air = airdata_from_body_velocity(u, v, w)
    V = air["V"]
    beta = air["beta"]
    qbar = dynamic_pressure(vehicle.rho_kgpm3, V)
    b = float(vehicle.geometry["span_m"])
    s = float(vehicle.geometry["area_m2"])
    p_hat = p * b / max(2.0 * V, 1e-6)
    r_hat = r * b / max(2.0 * V, 1e-6)

    aero = vehicle.aero
    cy = (
        aero.get("cy_beta_per_rad", 0.0) * beta
        + aero.get("cy_p", 0.0) * p_hat
        + aero.get("cy_r", 0.0) * r_hat
        + aero.get("cy_da_per_rad", 0.0) * da
        + aero.get("cy_dr_per_rad", 0.0) * dr
    )
    cl = (
        aero.get("cl_beta_per_rad", 0.0) * beta
        + aero.get("cl_p", 0.0) * p_hat
        + aero.get("cl_r", 0.0) * r_hat
        + aero.get("cl_da_per_rad", 0.0) * da
        + aero.get("cl_dr_per_rad", 0.0) * dr
    )
    cn = (
        aero.get("cn_beta_per_rad", 0.0) * beta
        + aero.get("cn_p", 0.0) * p_hat
        + aero.get("cn_r", 0.0) * r_hat
        + aero.get("cn_da_per_rad", 0.0) * da
        + aero.get("cn_dr_per_rad", 0.0) * dr
    )

    lateral_force = np.array([0.0, qbar * s * cy, 0.0], dtype=float)
    lateral_moment = np.array([qbar * s * b * cl, 0.0, qbar * s * b * cn], dtype=float)

    total_force = wing_force + tail_force + lateral_force
    total_moment = wing_moment + tail_moment + lateral_moment
    return total_force, total_moment, {
        "wing": wing_diag,
        "tail": tail_diag,
        "lateral_coefficients": {
            "beta_rad": float(beta),
            "p_hat": float(p_hat),
            "r_hat": float(r_hat),
            "cy": float(cy),
            "cl": float(cl),
            "cn": float(cn),
            "force_body_n": lateral_force.tolist(),
            "moment_body_nm": lateral_moment.tolist(),
        },
        "force_body_n": total_force.tolist(),
        "moment_body_nm": total_moment.tolist(),
    }
