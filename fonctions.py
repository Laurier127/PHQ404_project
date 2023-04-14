import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# on définit ici un solutionneur basé sur la méthode d'Euler
def methode_euler(fonction, vecteur_etat, duree, nombre_de_pas):
    """
    résout un système d'équations différentielles à l'aide de la méthode d'Euler
    Paramètres :
        fonction: fonction définissant le système. f(x, t) retourne un vecteur de dérivées
        vecteur_etat: point initial (vecteur)
        duree: durée de la simulation
        nombre_de_pas: nombre de pas
    returns:
        un tableau numpy de la solution
    """

    h = duree / nombre_de_pas

    # on ajoute 1 à la longueur de la solution pour prendre en compte les valeurs initiales
    grille = np.linspace(0, duree, nombre_de_pas+1)
    solution = np.empty((nombre_de_pas+1, len(vecteur_etat)))

    for I, t in enumerate(grille):
        solution[I, :] = vecteur_etat
        vecteur_etat += fonction(vecteur_etat, t) * h

    return solution


def integrate_methode_euler(derivs, state_init, t, dt):
    """
    Intègre numériquement un système d'équations différentielles d'ordre 2 avec la méthode d'Euler.
    :param derivs: une fonction qui prend en entrée un état et un temps et renvoie la dérivée de l'état à ce temps.
    :param state_init: l'état initial du système, une liste ou un tableau contenant les angles (en radians) et les vitesses angulaires (en radians par seconde) de chaque pendule.
    :param t_max: la durée totale de la simulation.
    :param dt: le pas de temps.
    :return: deux tableaux numpy contenant les temps et les états du système pour chaque instant.
    """
    print('here', np.arange(0, 30, 0.01))

    n = len(t)

    state = np.zeros((n, 4))
    state[0] = state_init

    for i in range(n - 1):
        deriv = derivs(state[i], t[i])
        state[i + 1] = state[i] + deriv * dt

    return state



def methode_predicteur_correcteur(fonction, vecteur_etat, duree, nombre_de_pas):
    """
    résout un système d'équations différentielles à l'aide de la méthode prédicteur-correcteur
    Paramètres :
        fonction: fonction définissant le système. f(x, t) retourne un vecteur de dérivées
        vecteur_etat: point initial (vecteur)
        duree: durée de la simulation
        nombre_de_pas: nombre de pas
    returns:
        un tableau numpy de la solution
    """

    h = duree / nombre_de_pas
    # on ajoute 1 à la longueur de la solution pour prendre en compte les valeurs initiales
    grille = np.linspace(0, duree, nombre_de_pas+1)
    solution = np.empty((nombre_de_pas+1, len(vecteur_etat)))

    #vecteur_etat_predit = vecteur_etat + fonction(vecteur_etat, 0) * h

    for I, t in enumerate(grille):
        solution[I, :] = vecteur_etat
        vecteur_etat_predit = vecteur_etat + fonction(vecteur_etat, t) * h
        vecteur_etat += 1 / 2 * h * (fonction(vecteur_etat, t) + fonction(vecteur_etat_predit, t))

    return solution

def methode_runge_kutta2(fonction, vecteur_etat, duree, nombre_de_pas, a=1 / 2):
    """
    résout un système d'équations différentielles à l'aide de la méthode de Runge-Kutta d'ordre 2
    Paramètres :
        fonction: fonction définissant le système. f(x, t) retourne un vecteur de dérivées
        vecteur_etat: point initial (vecteur)
        duree: durée de la simulation
        nombre_de_pas: nombre de pas
    returns:
        un tableau numpy de la solution
    """

    h = duree / nombre_de_pas
    # on ajoute 1 à la longueur de la solution pour prendre en compte les valeurs initiales
    grille = np.linspace(0, duree, nombre_de_pas + 1)
    solution = np.empty((nombre_de_pas + 1, len(vecteur_etat)))

    b = 1 - a
    alpha = 1/(2*b)
    beta = alpha

    for I, t in enumerate(grille):
        solution[I, :] = vecteur_etat
        k_1 = h * fonction(vecteur_etat, t)
        k_2 = h * fonction(vecteur_etat + beta * k_1, t + alpha * h)
        vecteur_etat += a*k_1+b*k_2

    return solution


def methode_scipy(fonction, vecteur_etat, duree, nombre_de_pas):
    """
    résout un système d'équations différentielles à l'aide de la méthode scipy.integrate.odeint
    Paramètres :
        fonction: fonction définissant le système. f(x, t) retourne un vecteur de dérivées
        vecteur_etat: point initial (vecteur)
        duree: durée de la simulation
        nombre_de_pas: nombre de pas
    returns:
        un tableau numpy de la solution
    """
    temps = np.linspace(0, duree, nombre_de_pas)
    return odeint(fonction, vecteur_etat, temps)


def kepler(x, t):
    """
    définis l'équation de kepler à résoudre
    Paramètres :
        x: vecteur à 4 dimensions composé comme suit (x,y,vx,vy)
        t: variable du temps (pas utilisé dans ce cas)
    returns:
        vecteur 4 dimensions (vx, vy, ax, ay)
    """
    R = np.sqrt(x[0]*x[0]+x[1]*x[1])
    f = 1.0/(R**2)
    return np.array([x[2], x[3], -f*x[0]/R, -f*x[1]/R])


def projectile(x,t):
    """
    définis l'équation d'un projectile
    Paramètres :
        x: vecteur à 4 dimensions composé comme suit (x,y,vx,vy)
        t: variable du temps (pas utilisé dans ce cas)
    returns:
        vecteur 4 dimensions (vx, vy, ax, ay)
    """
    g = 9.81
    return np.array([x[2], x[3], 0, -g])


def graph(X) :
    """
    affiche un graphique Matplotlib à l'Aide de la série de vecteurs envoyée
    Paramètres :
        X : un tableau numpy contenant les vecteurs
    returns:
    """
    plt.axes().set_aspect(1)
    plt.plot(X[:,0], X[:,1], 'b-', lw=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid()
    plt.show()

def comparatif_projectile (vecteur_etat=[0,0,10,49.05], duree=10, nombre_de_pas=2000):
    """
    Affiche 4 graphiques Matplotlib afin de comparer les méthodes de
        résolutions d'équations que l'on a implémenté avec la solution
        analytique pour un projectile.

    Paramètres :
        conditions_initiales: point initial (vecteur)
        duree: durée de la simulation
        nombre_de_pas: nombre de pas
    returns:
    """
    G = -9.81

    solution_analytique = np.empty((nombre_de_pas, 2))
    for I, t in enumerate(np.linspace(0, duree, nombre_de_pas)):
        x = vecteur_etat[2]*t + vecteur_etat[0]
        y = 1/2*G*t**2 + vecteur_etat[3]*t + vecteur_etat[1]

        solution_analytique[I:] = [x, y]

    init = [projectile, vecteur_etat, duree, nombre_de_pas]
    solutions = [methode_euler(*init),
                 methode_predicteur_correcteur(*init),
                 methode_runge_kutta2(*init),
                 methode_scipy(*init)]
    dim = 12
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(dim, dim/2), sharex=True, sharey=True)

    ax1.set_title("Euler")
    ax1.plot(solutions[0][:, 0], solutions[0][:, 1], 'r', label='Méthode d''Euler')
    ax1.plot(solution_analytique[:,0], solution_analytique[:,1], 'b', linestyle='dotted', label='solution analytique')

    ax2.set_title('Predicteur correcteur')
    ax2.plot(solutions[1][:, 0], solutions[1][:, 1], 'r', label='Méthode prédicteur correcteur' )
    ax2.plot(solution_analytique[:,0], solution_analytique[:,1], 'b', linestyle='dotted', label='solution analytique')

    ax3.set_title('Runge-Kutta d\'ordre 2')
    ax3.plot(solutions[2][:, 0], solutions[2][:, 1], 'r', label='Méthode RK-2')
    ax3.plot(solution_analytique[:,0], solution_analytique[:,1], 'b', linestyle='dotted', label='solution analytique')

    ax4.set_title('Scipy')
    ax4.plot(solutions[3][:, 0], solutions[3][:, 1], 'r', label='Méthode scipy')
    ax4.plot(solution_analytique[:,0], solution_analytique[:,1], 'b', linestyle='dotted', label='solution analytique')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    plt.show()