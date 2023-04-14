from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import fonctions as fc

FIGSIZE = (16, 9)
DPI = 120
GRAVITY_EARTH = 9.8
GRAVITY_MOON = 1.62
L1, L2 = 1.0, 0.8
M1, M2 = 1.2, 1.0


# Function to calculate the derivatives of the state variables
def derivs(state, t):
    res = np.zeros_like(state)
    res[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    res[1] = (M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta) +
              M2 * GRAVITY_MOON * sin(state[2]) * cos(delta) +
              M2 * L2 * state[3] * state[3] * sin(delta) -
              (M1 + M2) * GRAVITY_MOON * sin(state[0])) / den1

    res[2] = state[3]
    den2 = (L2 / L1) * den1
    res[3] = (-M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta) +
              (M1 + M2) * GRAVITY_MOON * sin(state[0]) * cos(delta) -
              (M1 + M2) * L1 * state[1] * state[1] * sin(delta) -
              (M1 + M2) * GRAVITY_MOON * sin(state[2])) / den2

    return res


time_step = 0.033
max_time = 40.0

time_points = np.arange(0.0, max_time, time_step)

# Initial state: angles (degrees) and angular velocities (degrees per second)
angle1 = 120.0
angular_velocity1 = 0.0
angle2 = -10.0
angular_velocity2 = 0.0

init_state_list = [np.radians([angle1, angular_velocity1, angle2, angular_velocity2]),
                   np.radians([angle1, angular_velocity1, angle2, angular_velocity2])]

# Integration
results_list = [integrate.odeint(derivs, init_state, time_points) for init_state in init_state_list]

use_euler_method = True
if use_euler_method:
    results_list = [integrate.odeint(derivs, init_state, time_points) for init_state in [init_state_list[0]]]
    euler_results_list = [
        fc.integrate_methode_euler(derivs, init_state, np.arange(0, max_time, time_step), time_step) for init_state
        in [init_state_list[1]]]
    results_list.append(euler_results_list[0])

# Plot settings
fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
fig_2 = plt.figure(figsize=FIGSIZE, dpi=DPI)

ax = fig.add_subplot(111)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 1)
fig.tight_layout()
fig_2.tight_layout()

lines = [ax.plot(res[:, 0], res[:, 1], 'o-', lw=3, markersize=20)[0] for res in results_list]

# Function to initialize the lines for animation
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Function to update the animation
def animate(i):
    print("Computing frame", i)
    for line, res in zip(lines, results_list):
        x1, y1 = L1 * sin(res[:, 0]), -L1 * cos(res[:, 0])
        x2, y2 = L2 * sin(res[:, 2]) + x1, -L2 * cos(res[:, 2]) + y1
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    # save the plot every 10 frames
    if i % 10 == 0 and i <= 1:
        filename = f"animation_comparaison_{i:03d}.png"
        plt.savefig(filename)
    fig.canvas.draw()
    return lines

plt.show()

# Function to compute Lyapunov exponent
def lyapunov_exponent(state1, state2, t, dt, n):
    # state1 and state2 are the initial states of the two slightly perturbed double pendulums
    # t is the time array
    # dt is the time step
    # n is the number of iterations to consider

    # initialization
    dist = np.linalg.norm(state1 - state2)
    d = np.zeros(n)
    d[0] = dist

    # integration loop
    for i in range(1, n):
        # integration of the two trajectories
        res1 = integrate.odeint(derivs, state1, [t[i - 1], t[i]])
        res2 = integrate.odeint(derivs, state2, [t[i - 1], t[i]])
        # calculation of the new distance
        dist = np.linalg.norm(res1[-1] - res2[-1])
        # normalization
        d[i] = dist / d[0]
        # update of the initial states
        state1 = res1[-1]
        state2 = res2[-1]

    # fit a line for the logarithm of the distance as a function of time
    p = np.polyfit(t[:n // 2], np.log(d[:n // 2]), 1)

    return p[0]  # Can be changed to 1/p[0] to have the Lyapunov time

compute_exponent = False
if compute_exponent:
    dt = 0.033
    t = np.arange(0.0, 20, dt)
    for i in range(4, len(results_list[0][:, 0])):
        plt.plot(i, lyapunov_exponent(init_state_list[0], init_state_list[1], t, dt, i), 'r.', label='Lyapunov Time')

    # Do not repeat labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys()).draw_frame(False)
    plt.xlabel('Time (s)')
    plt.ylabel('Lyapunov Time (τ)')
    plt.plot([0, 650], [0, 0], 'k--')
    # plt.ylim(-5, 40)

    plt.savefig('lyapunov_time', dpi=300)

    plt.show()

# Create animation object
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(results_list[0][:, 0])),
                              interval=33, blit=True, init_func=init)

# Set up animation writer
writer = animation.FFMpegWriter(fps=30, bitrate=5000)

# Save animation
if use_euler_method:
    ani.save('Comparaison_Euler.gif')
else:
    ani.save('Comparaison_1_degree.gif')


