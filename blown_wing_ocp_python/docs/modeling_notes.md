# Modeling Notes

## Why This Restart Exists

The earlier project accumulated too many layers at once: Simulink plant editing, strip bookkeeping, wake calibration, linearization, tracking control, and optional 6DOF playback. This restart narrows the problem back to a clean optimization pipeline.

The new model is intentionally:

- coefficient-based,
- first-principles in force/moment form,
- small enough to run in Colab,
- documented enough to defend in a class report.

## Core Equations

### Dynamic pressure

\[
q = \frac{1}{2}\rho V^2
\]

Reference:
- Anderson, *Fundamentals of Aerodynamics*, 5th ed.

### Aerodynamic force and moment coefficients

The aerodynamic model is written in standard low-order aircraft form:

\[
C_L = C_{L0} + C_{L_\alpha}\alpha + C_{L_q}\hat{q} + C_{L_{\delta_e}}\delta_e + \Delta C_{L,\text{blown}}
\]

\[
C_D = C_{D0} + k C_L^2 + \Delta C_{D,\text{blown}} + \Delta C_{D,\text{flap}}
\]

\[
C_m = C_{m0} + C_{m_\alpha}\alpha + C_{m_q}\hat{q} + C_{m_{\delta_e}}\delta_e + \Delta C_{m,\text{blown}}
\]

with

\[
\hat{q} = \frac{q c}{2V}
\]

Reference:
- Nelson, *Flight Stability and Automatic Control*
- Anderson, *Fundamentals of Aerodynamics*

### Propulsion

Per-prop static thrust is obtained by interpolation from the frozen capstone eCalc table.

Total thrust is modeled as:

\[
T = n_{\text{prop}} T_{\text{per-prop}}(\mathrm{RPM})
\]

and applied along the body \(+x\) axis.

### Blown-wing surrogate

The wake model uses a simple actuator-disk-inspired velocity increment:

\[
v_i = \sqrt{\frac{T}{2 \rho A_{\text{disk}}}}
\]

\[
V_{\text{wake}} \approx V_x + 2v_i
\]

\[
\mu_q = \left(\frac{V_{\text{wake}}}{\max(V_x,\varepsilon)}\right)^2 - 1
\]

The blown-wing lift increment is then modeled as

\[
\Delta C_{L,\text{blown}}
=
\eta_b
k_{L,b}
\mu_q
\left(1 + k_{L,f}\max(\delta_f,0)\right)
\]

where \(\eta_b\) accounts for blown span fraction and streamwise wake decay from prop disk to wing.

This is not claimed as a closed-form result from the references. It is a compact surrogate motivated by:

- momentum-theory slipstream scaling,
- Agrawal et al. (2019) wind-tunnel evidence that blown flaps alter section lift behavior strongly,
- Long (2021) discussion of wake delivery and flap/slot interaction.

## Why Keep 6DOF If the First OCP Is Longitudinal?

Because it helps keep the dynamics definitions honest. The first optimization problem is still reduced-order longitudinal, but the full rigid-body equations are already in the codebase for future extension.

## Frozen vs Tunable Quantities

Frozen:
- geometry,
- prop layout,
- tail arm,
- control limits,
- propeller thrust table.

Tunable:
- low-order aerodynamic coefficients,
- blown-wing gains,
- OCP weights,
- trajectory definition.

That split is deliberate and matches the “simple but defensible” goal of the restart.
