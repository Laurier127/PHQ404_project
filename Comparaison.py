from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
for iii in range(10):
    plt.rcParams['animation.ffmpeg_path'] = r'/Volumes/Data/Youtube/[ffmpeg]/ffmpeg'
    FIGSIZE = (16, 9)
    DPI = 120  # 240 For 4K, 120 for 1080p, 80 for 720p

    G = 9.8
    G = 1.62 # sur la lune
    G = 0.1
    L1, L2 = 1.0, 0.8
    M1, M2 = 1.2, 1.0


    def derivs(state, t):
        # http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

        res = np.zeros_like(state)
        res[0] = state[1]

        del_ = state[2] - state[0]
        den1 = (M1 + M2) * L1 - M2 * L1 * cos(del_) * cos(del_)
        res[1] = (M2 * L1 * state[1] * state[1] * sin(del_) * cos(del_) +
                  M2 * G * sin(state[2]) * cos(del_) +
                  M2 * L2 * state[3] * state[3] * sin(del_) -
                  (M1 + M2) * G * sin(state[0])) / den1

        res[2] = state[3]
        den2 = (L2 / L1) * den1
        res[3] = (-M2 * L2 * state[3] * state[3] * sin(del_) * cos(del_) +
                  (M1 + M2) * G * sin(state[0]) * cos(del_) -
                  (M1 + M2) * L1 * state[1] * state[1] * sin(del_) -
                  (M1 + M2) * G * sin(state[2])) / den2

        return res


    dt = 0.033
    t = np.arange(0.0, 40, dt)

    # initial state : angles (degrees) and angular velocities (degrees per second)
    th1 = 120.0
    w1 = 0.0
    th2 = -10.0
    w2 = 0.0

    init_state_list = [np.radians([th1, w1, th2, w2]),
                       np.radians([th1, w1, th2 + 1, w2])]

    # Integration
    res_list = [integrate.odeint(derivs, init_state, t) for init_state in init_state_list]

    ###############################################################################

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


    lines = [ax.plot(res[:, 0], res[:, 1], 'o-', lw=3, markersize=20)[0] for res in res_list]


    def init():
        for line in lines:
            line.set_data([], [])
        return lines


    def animate(i):
        print("Computing frame", i)
        for line, res in zip(lines, res_list):
            x1, y1 = L1 * sin(res[:, 0]), -L1 * cos(res[:, 0])
            x2, y2 = L2 * sin(res[:, 2]) + x1, -L2 * cos(res[:, 2]) + y1
            line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

        # save the plot every 10 frames
        if i % 10 == 0 and i <=100:
            filename = f"animation_comparaison_{i:03d}.png"

            plt.savefig(filename)
            plt.show()
        fig.canvas.draw()
        return lines



    plt.show()


    def lyapunov_exponent(state1, state2, t, dt, n):
        # state1 et state2 sont les états initiaux des deux pendules doubles déformés
        # t est l'array des temps
        # dt est le pas de temps
        # n est le nombre d'itérations à considérer

        # initialisation
        dist = np.linalg.norm(state1 - state2)
        d = np.zeros(n)
        d[0] = dist

        # boucle d'intégration
        for i in range(1, n):
            # intégration des deux trajectoires
            res1 = integrate.odeint(derivs, state1, [t[i - 1], t[i]])
            res2 = integrate.odeint(derivs, state2, [t[i - 1], t[i]])
            # calcul de la nouvelle distance
            dist = np.linalg.norm(res1[-1] - res2[-1])
            # normalisation
            d[i] = dist / d[0]
            # mise à jour des états initiaux
            state1 = res1[-1]
            state2 = res2[-1]

        # ajustement de la droite pour les logarithmes de la distance en fonction du temps
        p = np.polyfit(t[:n // 2], np.log(d[:n // 2]), 1)

        return p[0]

    exposant = False
    if exposant:
        dt = 0.033
        t = np.arange(0.0, 20, dt)
        for i in range(4, len(res_list[0][:, 0])):
            plt.plot(i, lyapunov_exponent(init_state_list[0], init_state_list[1], t, dt, i), 'b.', label='exposant de Lyapunov')

        # Ne pas repeter les labels dans la legende
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys()).draw_frame(False)
        plt.xlabel('temps (s)')
        plt.ylabel('exposant de Lyapunov (τ)')
        plt.plot([0, 650], [0, 0], 'k--')
        plt.ylim(-5, 40)

        #plt.savefig('exposant_lyapunov', dpi=300)

        plt.show()

    ani = animation.FuncAnimation(fig, animate, np.arange(1+iii*100, len(res_list[0][:, 0])),
                                  interval=33, blit=True, init_func=init, repeat=False)
    # ani_e_l = animation.FuncAnimation(fig_2, animate_2, np.arange(10, len(res_list[0][:, 0])),
    #                              interval=33, blit=True, init_func=init)
    writer = animation.FFMpegWriter(fps=30, bitrate=5000)

    ani.save('04-double_pendulum_poly_{}.gif'.format(iii))

