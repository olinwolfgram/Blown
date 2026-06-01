from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import Vehicle

jax.config.update("jax_enable_x64", True)


def _build_longitudinal_params(vehicle: Vehicle) -> dict[str, jnp.ndarray | float]:
    geom = vehicle.geometry
    aero = vehicle.aero
    prop = vehicle.propulsion
    blown = vehicle.blown

    span = float(geom["span_m"])
    chord = float(geom["chord_m"])
    area = float(geom["area_m2"])
    aspect_ratio = float(geom["aspect_ratio"])
    oswald_e = float(geom["oswald_e"])
    wing_inc = math.radians(float(geom["wing_incidence_deg"]))
    x_ac = float(geom["wing_ac_x_m"]) - float(geom["cg_x_m"])
    tail_arm = float(geom["tail_arm_m"])
    htail_area = float(geom["htail_area_m2"])
    htail_inc = math.radians(float(geom["htail_incidence_deg"]))
    flap_halfspan = 0.5 * span * float(geom["flap_span_fraction"])

    n_half = int(blown["n_strips_per_semispan"])
    ys = np.linspace(0.0, 0.5 * span, n_half + 1)
    strip_y = np.concatenate([-(0.5 * (ys[:-1] + ys[1:]))[::-1], 0.5 * (ys[:-1] + ys[1:])])
    strip_y = jnp.asarray(strip_y, dtype=jnp.float64)
    dy = 0.5 * span / n_half
    area_strip = chord * dy
    prop_y = jnp.asarray(prop["centers_y_m"], dtype=jnp.float64)
    halfwidth = 0.5 * float(blown["wake_span_expansion_factor"]) * float(prop["diameter_m"])
    blown_mask = jnp.any(jnp.abs(prop_y[None, :] - strip_y[:, None]) <= halfwidth, axis=1).astype(jnp.float64)
    flap_mask = (jnp.abs(strip_y) <= flap_halfspan).astype(jnp.float64)

    return {
        "mass": float(vehicle.mass_kg),
        "gravity": float(vehicle.gravity_mps2),
        "rho": float(vehicle.rho_kgpm3),
        "iyy": float(vehicle.inertia[1, 1]),
        "span": span,
        "chord": chord,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "oswald_e": oswald_e,
        "wing_inc": wing_inc,
        "x_ac": x_ac,
        "tail_arm": tail_arm,
        "htail_area": htail_area,
        "htail_inc": htail_inc,
        "flap_halfspan": flap_halfspan,
        "n_props": int(prop["n_props"]),
        "prop_y": prop_y,
        "diameter": float(prop["diameter_m"]),
        "rpm_grid": jnp.asarray(prop["rpm_grid"], dtype=jnp.float64),
        "thrust_grid": jnp.asarray(prop["thrust_n_grid"], dtype=jnp.float64),
        "wake_span_expansion_factor": float(blown["wake_span_expansion_factor"]),
        "wake_decay_length": float(blown["wake_decay_length_m"]),
        "prop_to_wing_leading_edge": float(blown["prop_to_wing_leading_edge_m"]),
        "lift_gain": float(blown["lift_gain"]),
        "drag_gain": float(blown["drag_gain"]),
        "pitch_gain": float(blown["pitch_gain"]),
        "flap_lift_gain": float(blown["flap_lift_gain"]),
        "flap_drag_gain": float(blown["flap_drag_gain"]),
        "flap_pitch_gain": float(blown["flap_pitch_gain"]),
        "strip_y": strip_y,
        "area_strip": area_strip,
        "blown_mask": blown_mask,
        "flap_mask": flap_mask,
        "cl0": float(aero["cl0"]),
        "cl_alpha": float(aero["cl_alpha_per_rad"]),
        "cd0": float(aero["cd0"]),
        "cm0": float(aero["cm0"]),
        "cm_alpha": float(aero["cm_alpha_per_rad"]),
        "cl_q": float(aero["cl_q"]),
        "cm_q": float(aero["cm_q"]),
        "cl_flap": float(aero["cl_flap_per_rad"]),
        "cd_flap2": float(aero["cd_flap_per_rad2"]),
        "cm_flap": float(aero["cm_flap_per_rad"]),
        "tail_cl_alpha": float(aero["tail_cl_alpha_per_rad"]),
        "tail_cl_de": float(aero["tail_cl_de_per_rad"]),
        "tail_cd0": float(aero["tail_cd0"]),
        "downwash_gradient": float(aero["downwash_gradient"]),
    }


def build_jax_longitudinal_dynamics(vehicle: Vehicle, dt: float):
    params = _build_longitudinal_params(vehicle)

    def thrust_per_prop(rpm: jnp.ndarray) -> jnp.ndarray:
        rpm_clamped = jnp.clip(rpm, params["rpm_grid"][0], params["rpm_grid"][-1])
        return jnp.interp(rpm_clamped, params["rpm_grid"], params["thrust_grid"])

    def wing_force_moment(u: jnp.ndarray, w: jnp.ndarray, q: jnp.ndarray, df: jnp.ndarray, rpm_coll: jnp.ndarray):
        speed = jnp.sqrt(u * u + w * w + 1e-12)
        alpha = jnp.arctan2(w, jnp.maximum(u, 1e-6))
        qbar = 0.5 * params["rho"] * speed * speed
        qhat = q * params["chord"] / jnp.maximum(2.0 * speed, 1e-6)
        alpha_eff = alpha + params["wing_inc"]

        thrust_single = thrust_per_prop(rpm_coll)
        disk_area = math.pi * (0.5 * params["diameter"]) ** 2
        v_induced = jnp.sqrt(jnp.maximum(thrust_single, 0.0) / (2.0 * params["rho"] * disk_area))
        u_axial = jnp.maximum(speed * jnp.cos(alpha), 0.5)
        v_wake = u_axial + 2.0 * v_induced
        base_ratio = jnp.maximum((v_wake / jnp.maximum(u_axial, 1e-3)) ** 2 - 1.0, 0.0)
        span_sigma = 0.5 * params["wake_span_expansion_factor"] * params["diameter"]
        x_decay = jnp.exp(-params["prop_to_wing_leading_edge"] / jnp.maximum(params["wake_decay_length"], 1e-6))

        total_fx = 0.0
        total_fz = 0.0
        total_my = 0.0
        for i in range(params["strip_y"].shape[0]):
            y_i = params["strip_y"][i]
            span_weights = jnp.exp(-0.5 * ((y_i - params["prop_y"]) / jnp.maximum(span_sigma, 1e-6)) ** 2)
            wake_ratio_raw = base_ratio * jnp.sum(span_weights) * x_decay
            wake_ratio = wake_ratio_raw * params["blown_mask"][i]
            flap_active = params["flap_mask"][i]
            q_local = qbar * (1.0 + wake_ratio)

            cl_clean = params["cl0"] + params["cl_alpha"] * alpha_eff + params["cl_q"] * qhat
            cl_flap = params["cl_flap"] * df * flap_active
            cl_blown = params["lift_gain"] * wake_ratio
            cl_blown_flap = params["flap_lift_gain"] * df * wake_ratio * flap_active
            cl = cl_clean + cl_flap + cl_blown + cl_blown_flap

            cd_parasite = params["cd0"]
            cd_induced = (1.0 / (math.pi * params["oswald_e"] * params["aspect_ratio"])) * cl * cl
            cd_blown = params["drag_gain"] * wake_ratio
            cd_flap = params["cd_flap2"] * df * df * flap_active + params["flap_drag_gain"] * df * wake_ratio * flap_active
            cd = cd_parasite + cd_induced + cd_blown + cd_flap

            cm = (
                params["cm0"]
                + params["cm_alpha"] * alpha
                + params["cm_q"] * qhat
                + params["cm_flap"] * df * flap_active
                + params["pitch_gain"] * wake_ratio
                + params["flap_pitch_gain"] * df * wake_ratio * flap_active
            )

            lift = q_local * params["area_strip"] * cl
            drag = q_local * params["area_strip"] * cd
            fx = -drag * jnp.cos(alpha) + lift * jnp.sin(alpha)
            fz = -drag * jnp.sin(alpha) - lift * jnp.cos(alpha)
            section_pitch = q_local * params["area_strip"] * params["chord"] * cm
            arm_pitch = -params["x_ac"] * fz

            total_fx = total_fx + fx
            total_fz = total_fz + fz
            total_my = total_my + arm_pitch + section_pitch
        return total_fx, total_fz, total_my

    def tail_force_moment(u: jnp.ndarray, w: jnp.ndarray, de: jnp.ndarray):
        speed = jnp.sqrt(u * u + w * w + 1e-12)
        alpha = jnp.arctan2(w, jnp.maximum(u, 1e-6))
        qbar = 0.5 * params["rho"] * speed * speed
        downwash = params["downwash_gradient"] * alpha
        alpha_tail = alpha + params["htail_inc"] - downwash
        cl_tail = params["tail_cl_alpha"] * alpha_tail + params["tail_cl_de"] * de
        lift_tail = qbar * params["htail_area"] * cl_tail
        drag_tail = qbar * params["htail_area"] * params["tail_cd0"]
        fx = -drag_tail * jnp.cos(alpha_tail) + lift_tail * jnp.sin(alpha_tail)
        fz = -drag_tail * jnp.sin(alpha_tail) - lift_tail * jnp.cos(alpha_tail)
        my = -params["tail_arm"] * fz
        return fx, fz, my

    def continuous_dynamics(x_lon: jnp.ndarray, u_lon: jnp.ndarray) -> jnp.ndarray:
        x_fwd, h, u, w, theta, q = x_lon
        rpm_coll, de = u_lon
        df = 0.0

        wing_fx, wing_fz, wing_my = wing_force_moment(u, w, q, df, rpm_coll)
        tail_fx, tail_fz, tail_my = tail_force_moment(u, w, de)
        thrust_total = params["n_props"] * thrust_per_prop(rpm_coll)

        fx = wing_fx + tail_fx + thrust_total - params["mass"] * params["gravity"] * jnp.sin(theta)
        fz = wing_fz + tail_fz + params["mass"] * params["gravity"] * jnp.cos(theta)
        my = wing_my + tail_my

        u_dot = fx / params["mass"] - q * w
        w_dot = fz / params["mass"] + q * u
        theta_dot = q
        q_dot = my / params["iyy"]
        x_dot = u * jnp.cos(theta) + w * jnp.sin(theta)
        h_dot = u * jnp.sin(theta) - w * jnp.cos(theta)
        return jnp.array([x_dot, h_dot, u_dot, w_dot, theta_dot, q_dot], dtype=jnp.float64)

    def rk4_step(x_lon: jnp.ndarray, u_lon: jnp.ndarray) -> jnp.ndarray:
        k1 = continuous_dynamics(x_lon, u_lon)
        k2 = continuous_dynamics(x_lon + 0.5 * dt * k1, u_lon)
        k3 = continuous_dynamics(x_lon + 0.5 * dt * k2, u_lon)
        k4 = continuous_dynamics(x_lon + dt * k3, u_lon)
        return x_lon + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    rk4_step_jit = jax.jit(rk4_step)
    jacobian_fn = jax.jit(jax.jacrev(rk4_step, argnums=(0, 1)))

    def dynamics_np(x_lon: np.ndarray, u_lon: np.ndarray) -> np.ndarray:
        return np.asarray(rk4_step_jit(jnp.asarray(x_lon, dtype=jnp.float64), jnp.asarray(u_lon, dtype=jnp.float64)))

    def jacobian_np(x_lon: np.ndarray, u_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A, B = jacobian_fn(jnp.asarray(x_lon, dtype=jnp.float64), jnp.asarray(u_lon, dtype=jnp.float64))
        return np.asarray(A), np.asarray(B)

    return dynamics_np, jacobian_np
