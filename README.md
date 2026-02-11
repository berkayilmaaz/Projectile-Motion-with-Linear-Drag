#Projectile Motion with Linear Drag
<p align="center">
<img src="assets/brky-logo.png" height="50">
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="assets/siths-logo.png" height="50">
</p>

<p align="center">
🚀 <b>For a more detailed analysis and interactive visualizations, visit our project website:</b>



<a href="https://projectile.brky.ai"><b>projectile.brky.ai</b></a>
</p>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

A numerical simulation of 2D projectile motion under gravity with linear (viscous) air resistance. Compares drag vs. drag-free trajectories and lets you sweep launch angle, drag coefficient, initial velocity, and gravitational acceleration.

Built as a university physics project by **Schrödinger's Siths** — originally a Jupyter notebook, now refactored into a clean standalone module.

---

## Overview

The simulator uses forward Euler integration to solve the equations of motion for a point mass launched at an angle. It produces six-panel static plots, animated GIFs, and parameter-sweep comparisons across angles, drag coefficients, planets, and velocities.

**What you get:**

- Side-by-side comparison of drag vs. no-drag trajectories
- Animated 6-panel dashboard (saved as `.gif`)
- Parameter sweeps with clean dark-themed plots
- Drag coefficient vs. landing distance analysis

---

## The Physics Behind It

We model a projectile of mass $m$ subject to gravity $\vec{g}$ and a **linear drag force** proportional to velocity:

$$
\vec{F}_{\text{drag}} = -k\,\vec{v}
$$

where $k$ is the drag coefficient in $\text{kg/s}$. Applying Newton's second law:

$$
m\,\vec{a} = m\,\vec{g} - k\,\vec{v}
$$

Splitting into components with $\gamma = k/m$:

$$
\ddot{x} = -\gamma\,\dot{x}
$$

$$
\ddot{y} = -g - \gamma\,\dot{y}
$$

These are first-order-reducible ODEs. The horizontal component has an analytical solution:

$$
v_x(t) = v_{x0}\,e^{-\gamma t}
\quad\Rightarrow\quad
x(t) = \frac{m\,v_{x0}}{k}\left(1 - e^{-\gamma t}\right)
$$

The vertical component doesn't separate as cleanly due to gravity, so we integrate both numerically using the Euler method:

$$
v_{x}^{n+1} = v_{x}^{n} - \gamma\,v_{x}^{n}\,\Delta t
$$

$$
v_{y}^{n+1} = v_{y}^{n} - (g + \gamma\,v_{y}^{n})\,\Delta t
$$

$$
x^{n+1} = x^{n} + v_{x}^{n}\,\Delta t, \qquad
y^{n+1} = y^{n} + v_{y}^{n}\,\Delta t
$$

**Note on drag models:** This project uses *linear* drag ($F \propto v$), appropriate for low Reynolds number regimes (slow, small objects in viscous media). For high-speed projectiles in air, *quadratic* drag ($F \propto v^2$) would be more realistic — but the linear model keeps the math tractable and still demonstrates the core physics clearly.

### Default parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| $m$ | 5.0 kg | Projectile mass |
| $k$ | 0.1 kg/s | Linear drag coefficient |
| $g$ | 9.81 m/s² | Gravitational acceleration |
| $v_0$ | 50.0 m/s | Initial speed |
| $\theta$ | 15° | Launch angle |
| $\Delta t$ | 0.01 s | Integration time step |

---

## Installation & Usage

```bash
git clone https://github.com/berkayilmaaz/Projectile-Motion-with-Linear-Drag.git
cd Projectile-Motion-with-Linear-Drag
pip install -r requirements.txt
```

**Run the full simulation:**

```bash
python projectile_motion.py
```

This generates all static plots (`.png`), an animated GIF (`trajectory_animation.gif`), and opens the matplotlib viewer.

**Use as a library:**

```python
from projectile_motion import SimulationParams, calculate_trajectory, plot_static

params = SimulationParams(v0=60.0, angle_deg=30.0, k=0.15)
drag, no_drag = calculate_trajectory(params)
fig = plot_static(drag, no_drag)
fig.savefig("my_plot.png", dpi=150)
```

---

## Project Structure

```
.
├── assets/                # Logos and visual assets (brky-logo.png, siths-logo.png)
├── projectile_motion.py   # simulation, plotting, animation
├── requirements.txt
└── README.md
```

---

## Team

| Name | Email |
|------|-------|
| Berkay Yılmaz | contact@brky.ai |
| Ahmet Ali Akkurt  | ahmetakkurt@marun.edu.tr |
| Livanur Çelik | livanurcelik@marun.edu.tr |
