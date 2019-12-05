# -*- coding: Latin-1 -*-
# TP optim : minimizing a function
# par l'algorithme PSO
# Peio Loubiere & Rachid Chelouah pour l'EISTI
# septembre 2017
# usage : python PSO.py
from scipy import *
from math import *
from matplotlib.pyplot import *
from functools import *
import sys

# Instance of the problem
# FILE="NUMBER.tsp"
# choose one of the following functions:
if (len(sys.argv) > 1):
    FUNCTION = sys.argv[1]
else:
    print("No specified function name...")
    sys.exit("USE : python PSOCONT.py sphere | griewank | schwefel | rosenbrock")

# TPSPause = 0.1 # for displaying
# fig1 = figure()
# canv = fig1.add_subplot(1,1,1)
# xticks([])
# yticks([])

DIM = 4  # problem dimension
Nb_particle = 25 * DIM
Nb_cycles = 30 * Nb_particle

# defining the boundries depending on functions.
# for rosenbrock i chose [-2.048, 2.048] or [-5, 10]
# for schwefel [-500,500]
# the values were taken from the link in tp-pso
def boundries():
    if FUNCTION == "schwefel":
        INF = -500
        SUP = 500
    elif FUNCTION == "rosenbrock":
        '''INF = -2.048
        SUP = 2.048'''
        INF = -5
        SUP = 10
    else:
        INF = -600
        SUP = 600
    return INF, SUP


INF, SUP = boundries()


# limiting the position from exceeding one of the boundries
def limiting(pos):
    global INF, SUP
    if pos < INF:
        pos = INF
    elif pos > SUP:
        pos = SUP
    return pos


# usual params
# psi, cmax = 0.6, 1.62
psi, cmax = 0.65, 1.5
# psi, cmax = (0.7, 1.47)
# psi, cmax = (0.8, 1.62)

# Creating the the figure
fig1 = figure()
canv = fig1.add_subplot(1, 1, 1)
canv.set_xlim(INF, SUP)
canv.set_ylim(INF , SUP)
pZoom = 1  # 0.8 # zoom setting
TPSPause = 0.001  # for displaying


# Trace les deux premieres coordonnees des particleules
# pre-conditions :
#   - swarm : all particles,
#   - n_cycle : number of iteration
def draw(swarm, n_cycle):
    global canv
    # zoom centred on zero
    m, M = canv.get_xlim()
    canv.clear()
    canv.set_xlim(m * pZoom, M * pZoom)
    canv.set_ylim(m * pZoom, M * pZoom)
    # showing the first two dimensions of each particle
    for p in swarm:
        canv.plot(p['pos'][0], p['pos'][1], 'ro')
    title("Iteration : {}".format(n_cycle))
    pause(TPSPause)


# Figure des graphes de :
#   - l'ensemble des energies des fluctuations retenues
#   - la meilleure energie
def drawStats(Htemps, Hbest):
    # afFILEhage des courbes d'evolution
    fig2 = figure(2)
    subplot(1, 1, 1)
    semilogy(Htemps, Hbest)
    title('Evolution of the best distance')
    xlabel('Time')
    ylabel('Distance')
    show()


# Displaying the best found particle
# pre-conditions :
#   - best : best found particle,
def dispRes(best):
    print("point = {}".format(best['pos']))
    print("eval = {}".format(best['fit']))


def sphere(sol):
    return reduce(lambda acc, e: acc + e * e, sol, 0)


def griewank(sol):
    (s, p, i) = reduce(lambda acc, e: (acc[0] + e * e, acc[1] * cos(e / sqrt(acc[2])), acc[2] + 1), sol, (0, 1, 1))
    return s / 4000


def rosenbrock(sol):
    sum = 0

    for i in range(0, len(sol) - 1):
        sum += 100 * ((sol[i + 1] - (sol[i] ** 2)) ** 2) + (sol[i] - 1) ** 2

    return sum


def schwefel(sol):
    d = len(sol)

    sum = 0

    for i in range(0, d):
        sum += sol[i] * sin(sqrt(abs(sol[i])))

    tot = 418.9829 * d - sum

    return tot


# FUNCTION of evaluation',
# pre-condition :
#   - sol : point of the plane
# post-condition : f(point)
def eval(sol):
    global FUNCTION
    if FUNCTION == "sphere":
        return sphere(sol)
    elif FUNCTION == "griewank":
        return griewank(sol)
    elif FUNCTION == "rosenbrock":
        return rosenbrock(sol)
    elif FUNCTION == "schwefel":
        return schwefel(sol)
    else:
        print("filou, va!")
        exit(1)


# create a particle
# one particle is discribed by :
#   - pos : solution list of variables
#   - vit : movement velocity (null at the initialization)
#   - fit :  fitness of the solution
#   - bestpos : best visited position
#   - bestfit : evaluation of the best visited solution
#   - bestvois : best neighbor (global for this version)
def initOne(dim, inf, sup):
    pos = [random.uniform(inf, sup) for i in range(dim)]
    fit = eval(pos)
    return {'vit': [0] * dim, 'pos': pos, 'fit': fit, 'bestpos': pos, 'bestfit': fit, 'bestvois': []}


# Init of the population (swarm)
def initSwarm(nb, dim, inf, sup):
    return [initOne(dim, inf, sup) for i in range(nb)]


# Return the particle with the best fitness
def maxParticle(p1, p2):
    if (p1["fit"] < p2["fit"]):
        return p1
    else:
        return p2


# Returns a copy of the particle with the best fitness in the population
def getBest(swarm):
    return dict(reduce(lambda acc, e: maxParticle(acc, e), swarm[1:], swarm[0]))


# Update information for the particles of the population (swarm)
def update(particle, bestParticle):
    nv = dict(particle)
    if (particle["fit"] < particle["bestfit"]):
        nv['bestpos'] = particle["pos"][:]
        nv['bestfit'] = particle["fit"]
    nv['bestvois'] = bestParticle["bestpos"][:]
    return nv


###
## Local Update
##


def get_neiborhood(particle, swarm, nbn):
    neighbors = []
    particle_index = swarm.index(particle)

    # for even numbers:
    if nbn%2 ==0:
        # neighbors successors in the swarm list
        for i in range(particle_index + int(nbn/2), particle_index, -1):
            # if i is out of range for swarm list so we take from beginning
            # example: particle_index =8, len(swarm) = 8, nbn/2 = 3 ==> we take swarm[1,2,3]
            if i >= len(swarm):
                neighbors.append(swarm[i-len(swarm)])
            else:
                neighbors.append(swarm[i])
        # neighbors predecessors in the swarm list
        # if particle_index< nbn/2 ==> we will take from the end of the swarm list
        # example : particle_index = 1, nbn/2 = 3 ==> we take swarm[-3,-2,-1]
        for i in range(particle_index - int(nbn/2) -1, particle_index):
            neighbors.append(swarm[i])
    # for odd numbers we will randomly select weather to take successors>predecessors or the inverse
    else:
        if random.uniform(0,1) > 0.5:
            # neighbors successors in the swarm list
            # we take successors > predecessors
            for i in range(particle_index + int(nbn / 2) + 1, particle_index, -1):
                # if i is out of range for swarm list so we take from beginning
                # example: particle_index =8, len(swarm) = 8, nbn/2 = 3 ==> we take swarm[1,2,3]
                if i >= len(swarm):
                    neighbors.append(swarm[i - len(swarm)])
                else:
                    neighbors.append(swarm[i])
            # neighbors predecessors in the swarm list
            # if particle_index< nbn/2 ==> we will take from the end of the swarm list
            # example : particle_index = 1, nbn/2 = 3 ==> we take swarm[-3,-2,-1]
            for i in range(particle_index - int(nbn / 2) - 1, particle_index):
                neighbors.append(swarm[i])

        else:
            # neighbors successors in the swarm list
            # we take predecessors > successors
            for i in range(particle_index + int(nbn / 2), particle_index, -1):
                # if i is out of range for swarm list so we take from beginning
                # example: particle_index =8, len(swarm) = 8, nbn/2 = 3 ==> we take swarm[1,2,3]
                if i >= len(swarm):
                    neighbors.append(swarm[i - len(swarm)])
                else:
                    neighbors.append(swarm[i])
            # neighbors predecessors in the swarm list
            # if particle_index< nbn/2 ==> we will take from the end of the swarm list
            # example : particle_index = 1, nbn/2 = 3 ==> we take swarm[-3,-2,-1]
            for i in range(particle_index - int(nbn / 2) - 2, particle_index):
                neighbors.append(swarm[i])

    return neighbors


# Update information for the particles from a given nb of neighbors
def update_from_local_neighbors(particle, swarm, nbn):
    # get best neighbors
    best_particle = getBest(get_neiborhood(particle, swarm, nbn))

    # update
    nv = dict(particle)
    if (particle["fit"] < particle["bestfit"]):
        nv['bestpos'] = particle["pos"][:]
        nv['bestfit'] = particle["fit"]
    nv['bestvois'] = best_particle["bestpos"][:]
    return nv


#################

# Calculate the velocity and move a paticule
def move(particle, dim):
    global ksi, c1, c2, psi, cmax

    nv = dict(particle)

    velocity = [0] * dim
    for i in range(dim):
        velocity[i] = (particle["vit"][i] * psi +
                       cmax * random.uniform() * (particle["bestpos"][i] - particle["pos"][i]) +
                       cmax * random.uniform() * (particle["bestvois"][i] - particle["pos"][i]))
    position = [0] * dim

    for i in range(dim):
        position[i] = limiting(particle["pos"][i] + velocity[i])

    # position = limiting(position)

    nv['vit'] = velocity
    nv['pos'] = position
    nv['fit'] = eval(position)

    return nv


Htemps = []  # temps
Hbest = []  # distance

# initialization of the population
swarm = initSwarm(Nb_particle, DIM, INF, SUP)
# initialization of the best solution
best = getBest(swarm)
best_cycle = best

for i in range(Nb_cycles):
    # Update informations
    # swarm = [update(e, best_cycle) for e in swarm]

    swarm = [update_from_local_neighbors(e,swarm,10) for e in swarm]

    # velocity calculations and displacement
    swarm = [move(e, DIM) for e in swarm]
    # Update of the best solution
    best_cycle = getBest(swarm)
    if (best_cycle["bestfit"] < best["bestfit"]):
        best = best_cycle
        # draw(best['pos'], best['fit'])

    # historization of data
    if i % 2 == 0:
        Htemps.append(i)
        Hbest.append(best['bestfit'])

    # swarm display
    if i % 10 == 0:
        draw(swarm, i)

# END, displaying results
Htemps.append(i)
Hbest.append(best['bestfit'])
draw(swarm, i)

# displaying result on the console
dispRes(best)
drawStats(Htemps, Hbest)
