"""
Projectile motion simulation with linear drag.
Compares trajectories with and without air resistance using Euler integration.

Team: Schrödinger's Siths
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class SimulationParams:
    v0: float = 50.0        # initial velocity [m/s]
    angle_deg: float = 15.0 # launch angle [degrees]
    k: float = 0.1          # linear drag coefficient [kg/s]
    m: float = 5.0          # mass [kg]
    g: float = 9.81         # gravitational acceleration [m/s^2]
    dt: float = 0.01        # time step [s]
    t_max: float = 30.0     # max simulation time [s]


@dataclass
class TrajectoryData:
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    vx: np.ndarray = field(default_factory=lambda: np.array([]))
    vy: np.ndarray = field(default_factory=lambda: np.array([]))
    v: np.ndarray = field(default_factory=lambda: np.array([]))
    ke: np.ndarray = field(default_factory=lambda: np.array([]))
    pe: np.ndarray = field(default_factory=lambda: np.array([]))
    te: np.ndarray = field(default_factory=lambda: np.array([]))


# ──────────────────────────────────────────────
# Physics engine
# ──────────────────────────────────────────────

def calculate_trajectory(params: SimulationParams) -> tuple[TrajectoryData, TrajectoryData]:
    """
    Euler integration for projectile motion.
    Returns (with_drag, no_drag) trajectory data.

    Drag model (LINEAR):
        a_x = -(k/m) * v_x
        a_y = -g - (k/m) * v_y
    """
    angle_rad = np.radians(params.angle_deg)
    vx0 = params.v0 * np.cos(angle_rad)
    vy0 = params.v0 * np.sin(angle_rad)

    gamma = params.k / params.m  # drag-to-mass ratio [1/s]

    # pre-allocate with generous upper bound
    max_steps = int(params.t_max / params.dt) + 1

    # with drag
    t_arr = np.zeros(max_steps)
    x_arr = np.zeros(max_steps)
    y_arr = np.zeros(max_steps)
    vx_arr = np.zeros(max_steps)
    vy_arr = np.zeros(max_steps)

    # no drag
    x_nd = np.zeros(max_steps)
    y_nd = np.zeros(max_steps)
    vx_nd = np.zeros(max_steps)
    vy_nd = np.zeros(max_steps)

    # initial conditions
    vx_arr[0], vy_arr[0] = vx0, vy0
    vx_nd[0], vy_nd[0] = vx0, vy0

    n = 1
    for i in range(1, max_steps):
        t_arr[i] = i * params.dt

        # --- with drag (LINEAR: F_drag = -k * v) ---
        ax = -gamma * vx_arr[i - 1]
        ay = -params.g - gamma * vy_arr[i - 1]

        vx_arr[i] = vx_arr[i - 1] + ax * params.dt
        vy_arr[i] = vy_arr[i - 1] + ay * params.dt
        x_arr[i] = x_arr[i - 1] + vx_arr[i - 1] * params.dt
        y_arr[i] = y_arr[i - 1] + vy_arr[i - 1] * params.dt

        # --- no drag ---
        vx_nd[i] = vx0
        vy_nd[i] = vy_nd[i - 1] - params.g * params.dt
        x_nd[i] = x_nd[i - 1] + vx_nd[i - 1] * params.dt
        y_nd[i] = y_nd[i - 1] + vy_nd[i - 1] * params.dt

        n = i + 1
        if y_arr[i] < 0 or y_nd[i] < 0:
            break

    # trim to actual length
    sl = slice(0, n)
    t_out = t_arr[sl]

    def _build(x, y, vx, vy):
        v = np.sqrt(vx ** 2 + vy ** 2)
        ke = 0.5 * params.m * v ** 2
        pe = params.m * params.g * np.maximum(y, 0.0)
        te = ke + pe
        return TrajectoryData(t=t_out, x=x, y=y, vx=vx, vy=vy, v=v, ke=ke, pe=pe, te=te)

    drag = _build(x_arr[sl], y_arr[sl], vx_arr[sl], vy_arr[sl])
    no_drag = _build(x_nd[sl], y_nd[sl], vx_nd[sl], vy_nd[sl])

    return drag, no_drag


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────

STYLE_CONFIG = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.8,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 7,
    "font.size": 9,
}

PALETTE = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#f0883e", "#79c0ff"]
NODRAG_STYLE = dict(linestyle="--", color="#484f58", alpha=0.6, linewidth=1)


def _apply_style():
    plt.rcParams.update(STYLE_CONFIG)


def plot_static(drag: TrajectoryData, no_drag: TrajectoryData, title_suffix: str = ""):
    """Six-panel overview: trajectory, velocities, positions, speed, speed-vs-x, energies."""
    _apply_style()
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"Projectile Motion — Linear Drag{title_suffix}", fontsize=13, color="#e6edf3")

    c = PALETTE

    # (0,0) trajectory
    ax = axs[0, 0]
    ax.plot(drag.x, drag.y, color=c[0], label="With drag")
    ax.plot(no_drag.x, no_drag.y, label="No drag", **NODRAG_STYLE)
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Trajectory")
    ax.legend()
    ax.grid(True)

    # (0,1) velocity components vs time
    ax = axs[0, 1]
    ax.plot(drag.t, drag.vx, color=c[0], label="vx")
    ax.plot(drag.t, drag.vy, color=c[1], label="vy")
    ax.plot(no_drag.t, no_drag.vx, label="vx (no drag)", **NODRAG_STYLE)
    ax.plot(no_drag.t, no_drag.vy, label="vy (no drag)", **{**NODRAG_STYLE, "linestyle": ":"})
    ax.set(xlabel="t [s]", ylabel="v [m/s]", title="Velocity Components")
    ax.legend()
    ax.grid(True)

    # (1,0) position components vs time
    ax = axs[1, 0]
    ax.plot(drag.t, drag.x, color=c[0], label="x")
    ax.plot(drag.t, drag.y, color=c[1], label="y")
    ax.plot(no_drag.t, no_drag.x, label="x (no drag)", **NODRAG_STYLE)
    ax.plot(no_drag.t, no_drag.y, label="y (no drag)", **{**NODRAG_STYLE, "linestyle": ":"})
    ax.set(xlabel="t [s]", ylabel="pos [m]", title="Position Components")
    ax.legend()
    ax.grid(True)

    # (1,1) total speed vs time
    ax = axs[1, 1]
    ax.plot(drag.t, drag.v, color=c[2], label="With drag")
    ax.plot(no_drag.t, no_drag.v, label="No drag", **NODRAG_STYLE)
    ax.set(xlabel="t [s]", ylabel="|v| [m/s]", title="Speed")
    ax.legend()
    ax.grid(True)

    # (2,0) speed vs horizontal distance
    ax = axs[2, 0]
    ax.plot(drag.x, drag.v, color=c[3], label="With drag")
    ax.plot(no_drag.x, no_drag.v, label="No drag", **NODRAG_STYLE)
    ax.set(xlabel="x [m]", ylabel="|v| [m/s]", title="Speed vs. Distance")
    ax.legend()
    ax.grid(True)

    # (2,1) energies
    ax = axs[2, 1]
    ax.plot(drag.t, drag.ke, color=c[0], label="KE")
    ax.plot(drag.t, drag.pe, color=c[1], label="PE")
    ax.plot(drag.t, drag.te, color=c[2], label="Total")
    ax.set(xlabel="t [s]", ylabel="E [J]", title="Energy")
    ax.legend()
    ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_animation(drag: TrajectoryData, no_drag: TrajectoryData,
                     fps: int = 30, skip: int = 3) -> FuncAnimation:
    """Smooth animated version of the six-panel plot."""
    _apply_style()
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Projectile Motion — Linear Drag (animated)", fontsize=13, color="#e6edf3")

    c = PALETTE
    pad = 1.05  # axis limit padding

    def _lim(arr, lo=None):
        mn = lo if lo is not None else float(np.min(arr))
        mx = float(np.max(arr))
        margin = (mx - mn) * 0.08 if mx != mn else 1.0
        return (mn - margin, mx + margin)

    # --- set up axes limits from full data ---
    all_x = np.concatenate([drag.x, no_drag.x])
    all_y = np.concatenate([drag.y, no_drag.y])
    all_t = drag.t

    axs[0, 0].set(xlim=_lim(all_x, 0), ylim=_lim(all_y, 0), xlabel="x [m]", ylabel="y [m]", title="Trajectory")
    axs[0, 1].set(xlim=_lim(all_t, 0), ylim=_lim(np.concatenate([drag.vx, drag.vy, no_drag.vx, no_drag.vy])),
                  xlabel="t [s]", ylabel="v [m/s]", title="Velocity Components")
    axs[1, 0].set(xlim=_lim(all_t, 0), ylim=_lim(np.concatenate([drag.x, drag.y, no_drag.x, no_drag.y]), 0),
                  xlabel="t [s]", ylabel="pos [m]", title="Position Components")
    axs[1, 1].set(xlim=_lim(all_t, 0), ylim=_lim(np.concatenate([drag.v, no_drag.v]), 0),
                  xlabel="t [s]", ylabel="|v| [m/s]", title="Speed")
    axs[2, 0].set(xlim=_lim(all_x, 0), ylim=_lim(np.concatenate([drag.v, no_drag.v]), 0),
                  xlabel="x [m]", ylabel="|v| [m/s]", title="Speed vs. Distance")
    axs[2, 1].set(xlim=_lim(all_t, 0), ylim=_lim(np.concatenate([drag.ke, drag.pe, drag.te]), 0),
                  xlabel="t [s]", ylabel="E [J]", title="Energy")

    for ax in axs.flat:
        ax.grid(True)

    # create line objects
    lines = {}
    # trajectory
    lines["traj"], = axs[0, 0].plot([], [], color=c[0], lw=1.5)
    lines["traj_nd"], = axs[0, 0].plot([], [], **NODRAG_STYLE)
    lines["traj_pt"], = axs[0, 0].plot([], [], "o", color=c[0], ms=5, zorder=5)
    lines["traj_pt_nd"], = axs[0, 0].plot([], [], "o", color="#484f58", ms=4, zorder=5)

    # velocity components
    lines["vx"], = axs[0, 1].plot([], [], color=c[0], lw=1.5, label="vx")
    lines["vy"], = axs[0, 1].plot([], [], color=c[1], lw=1.5, label="vy")
    lines["vx_nd"], = axs[0, 1].plot([], [], **NODRAG_STYLE)
    lines["vy_nd"], = axs[0, 1].plot([], [], **{**NODRAG_STYLE, "linestyle": ":"})

    # position components
    lines["px"], = axs[1, 0].plot([], [], color=c[0], lw=1.5, label="x")
    lines["py"], = axs[1, 0].plot([], [], color=c[1], lw=1.5, label="y")
    lines["px_nd"], = axs[1, 0].plot([], [], **NODRAG_STYLE)
    lines["py_nd"], = axs[1, 0].plot([], [], **{**NODRAG_STYLE, "linestyle": ":"})

    # speed
    lines["spd"], = axs[1, 1].plot([], [], color=c[2], lw=1.5)
    lines["spd_nd"], = axs[1, 1].plot([], [], **NODRAG_STYLE)

    # speed vs x
    lines["svx"], = axs[2, 0].plot([], [], color=c[3], lw=1.5)
    lines["svx_nd"], = axs[2, 0].plot([], [], **NODRAG_STYLE)

    # energy
    lines["ke"], = axs[2, 1].plot([], [], color=c[0], lw=1.5, label="KE")
    lines["pe"], = axs[2, 1].plot([], [], color=c[1], lw=1.5, label="PE")
    lines["te"], = axs[2, 1].plot([], [], color=c[2], lw=1.5, label="Total")

    # legends
    for ax in axs.flat:
        if ax.get_legend_handles_labels()[1]:
            ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    n_frames = len(drag.t)
    frame_indices = list(range(0, n_frames, skip))

    def _update(idx):
        i = frame_indices[idx]
        s = slice(0, i + 1)

        lines["traj"].set_data(drag.x[s], drag.y[s])
        lines["traj_nd"].set_data(no_drag.x[s], no_drag.y[s])
        lines["traj_pt"].set_data([drag.x[i]], [drag.y[i]])
        lines["traj_pt_nd"].set_data([no_drag.x[i]], [no_drag.y[i]])

        lines["vx"].set_data(drag.t[s], drag.vx[s])
        lines["vy"].set_data(drag.t[s], drag.vy[s])
        lines["vx_nd"].set_data(no_drag.t[s], no_drag.vx[s])
        lines["vy_nd"].set_data(no_drag.t[s], no_drag.vy[s])

        lines["px"].set_data(drag.t[s], drag.x[s])
        lines["py"].set_data(drag.t[s], drag.y[s])
        lines["px_nd"].set_data(no_drag.t[s], no_drag.x[s])
        lines["py_nd"].set_data(no_drag.t[s], no_drag.y[s])

        lines["spd"].set_data(drag.t[s], drag.v[s])
        lines["spd_nd"].set_data(no_drag.t[s], no_drag.v[s])

        lines["svx"].set_data(drag.x[s], drag.v[s])
        lines["svx_nd"].set_data(no_drag.x[s], no_drag.v[s])

        lines["ke"].set_data(drag.t[s], drag.ke[s])
        lines["pe"].set_data(drag.t[s], drag.pe[s])
        lines["te"].set_data(drag.t[s], drag.te[s])

        return list(lines.values())

    ani = FuncAnimation(fig, _update, frames=len(frame_indices),
                        interval=1000 // fps, blit=True)
    return ani


# ──────────────────────────────────────────────
# Parameter sweep plots
# ──────────────────────────────────────────────

def compare_parameter(base_params: SimulationParams, param_name: str,
                      values: list, labels: Optional[list] = None):
    """
    Sweep one parameter while keeping the rest fixed.
    Produces a 2x2 panel: trajectory, speed, KE, PE.
    """
    _apply_style()
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    if labels is None:
        labels = [f"{param_name}={v}" for v in values]

    titles = ["Trajectory", "Speed vs. Time", "Kinetic Energy", "Potential Energy"]
    ylabels = ["y [m]", "|v| [m/s]", "KE [J]", "PE [J]"]
    xlabels = ["x [m]", "t [s]", "t [s]", "t [s]"]

    for idx, (val, label) in enumerate(zip(values, labels)):
        p = SimulationParams(**{**base_params.__dict__, param_name: val})
        drag, no_drag = calculate_trajectory(p)
        color = PALETTE[idx % len(PALETTE)]

        axs[0, 0].plot(drag.x, drag.y, color=color, label=label, lw=1.5)
        axs[0, 1].plot(drag.t, drag.v, color=color, label=label, lw=1.5)
        axs[1, 0].plot(drag.t, drag.ke, color=color, label=label, lw=1.5)
        axs[1, 1].plot(drag.t, drag.pe, color=color, label=label, lw=1.5)

        if idx == 0:
            for ax_pair, xd, yd in [
                (axs[0, 0], no_drag.x, no_drag.y),
                (axs[0, 1], no_drag.t, no_drag.v),
            ]:
                ax_pair.plot(xd, yd, label="No drag", **NODRAG_STYLE)

    for ax, title, xl, yl in zip(axs.flat, titles, xlabels, ylabels):
        ax.set(title=title, xlabel=xl, ylabel=yl)
        ax.legend()
        ax.grid(True)

    swept_display = param_name.replace("_", " ").title()
    fig.suptitle(f"Parameter Sweep: {swept_display}", fontsize=13, color="#e6edf3")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def compare_drag_vs_distance(base_params: SimulationParams, k_values: list):
    """Shows how landing distance shrinks with increasing drag coefficient."""
    _apply_style()
    fig, (ax_traj, ax_bar) = plt.subplots(1, 2, figsize=(16, 5),
                                           gridspec_kw={"width_ratios": [2, 1]})

    landing_distances = []
    for idx, k_val in enumerate(k_values):
        p = SimulationParams(**{**base_params.__dict__, "k": k_val})
        drag, no_drag = calculate_trajectory(p)
        color = PALETTE[idx % len(PALETTE)]
        ax_traj.plot(drag.x, drag.y, color=color, label=f"k={k_val:.1f}", lw=1.3)
        landing_distances.append(drag.x[-1])

        if idx == 0:
            ax_traj.plot(no_drag.x, no_drag.y, label="No drag", **NODRAG_STYLE)

    ax_traj.set(xlabel="x [m]", ylabel="y [m]", title="Trajectories")
    ax_traj.legend(fontsize=7)
    ax_traj.grid(True)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(k_values))]
    ax_bar.barh([f"k={k:.1f}" for k in k_values], landing_distances, color=colors, height=0.6)
    ax_bar.set(xlabel="Landing distance [m]", title="Range vs. Drag Coefficient")
    ax_bar.grid(True, axis="x")

    fig.suptitle("Drag Coefficient → Range Relationship", fontsize=13, color="#e6edf3")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    params = SimulationParams(v0=50.0, angle_deg=15.0, k=0.1, m=5.0, g=9.81)

    # 1) main experiment
    drag, no_drag = calculate_trajectory(params)

    fig_static = plot_static(drag, no_drag)
    fig_static.savefig("static_overview.png", dpi=150, bbox_inches="tight")

    ani = create_animation(drag, no_drag, fps=30, skip=3)
    ani.save("trajectory_animation.gif", writer=PillowWriter(fps=30))

    # 2) angle sweep
    fig_angles = compare_parameter(params, "angle_deg", [15, 45, 75],
                                   labels=["15°", "45°", "75°"])
    fig_angles.savefig("sweep_angles.png", dpi=150, bbox_inches="tight")

    # 3) drag coefficient sweep
    fig_drag = compare_parameter(params, "k", [0.05, 0.1, 0.2],
                                 labels=["k=0.05", "k=0.1", "k=0.2"])
    fig_drag.savefig("sweep_drag.png", dpi=150, bbox_inches="tight")

    # 4) planet comparison
    planet_params = {"Earth": 9.81, "Mars": 3.71, "Jupiter": 24.79, "Moon": 1.62}
    fig_planets = compare_parameter(
        SimulationParams(**{**params.__dict__, "angle_deg": 45.0}),
        "g", list(planet_params.values()), labels=list(planet_params.keys())
    )
    fig_planets.savefig("sweep_planets.png", dpi=150, bbox_inches="tight")

    # 5) velocity sweep
    fig_vel = compare_parameter(
        SimulationParams(**{**params.__dict__, "angle_deg": 45.0}),
        "v0", [20, 50, 80],
        labels=["20 m/s", "50 m/s", "80 m/s"]
    )
    fig_vel.savefig("sweep_velocity.png", dpi=150, bbox_inches="tight")

    # 6) drag vs distance
    k_range = [round(0.1 * i, 1) for i in range(1, 11)]
    fig_kvd = compare_drag_vs_distance(params, k_range)
    fig_kvd.savefig("drag_vs_distance.png", dpi=150, bbox_inches="tight")

    print("Done. Outputs saved.")
    plt.show()


if __name__ == "__main__":
    main()
