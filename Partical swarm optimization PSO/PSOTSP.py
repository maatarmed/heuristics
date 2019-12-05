# - * - coding: Latin-1 - * -
# resolve the problem of the traveling salesman
# by the simulation annealing algorithm
# Dominique Lefebvre for TangenteX.com
# P. Loubiere & R. Chelouah for EISTI
# September 2017
# use: python RSTSP.py FILE_NAME
from scipy import *
from math import *
from matplotlib.pyplot import *
import sys
from functools import reduce

# Instance of the problem
# FIC = "NUMERO.tsp"
if len(sys.argv) > 1:
    FIC = sys.argv[1]
else:
    print("No  specified file...")
    sys.exit("THE USE : python RSTSP.py Instance-number.tsp")

# Creating a figure
TPSPause = 1e-10  # for displaying
fig1 = figure()
canv = fig1.add_subplot(1, 1, 1)
xticks([])
yticks([])

##########################################################

Nb_cycles = 500
Nb_particle = 40
# usual params
psi, cmax = (0.7, 1.47)


# psi,cmax = (0.8, 1.62)

#######################################################


# Parsing the data file
# pre-condition: filename: file name (must exist)
# post-condition: (x, y) city coordinates
def parse(nomfic):
    absc = []
    ordo = []
    with open(nomfic, 'r') as inf:
        for line in inf:
            absc.append(float(line.split(' ')[1]))
            ordo.append(float(line.split(' ')[2]))
    return (array(absc), array(ordo))


# Refresh the figure of the trip, we trace the best route found
# pre-conditions:
# - best_route, best_dist: best ride found and its length,
# - x, y: coordinate tables of waypoints
def draw(best_route, best_dist, x, y):
    global canv, lx, ly
    canv.clear()
    canv.plot(x[best_route], y[best_route], 'k')
    canv.plot([x[best_route[-1]], x[best_route[0]]], [y[best_route[-1]],
                                                      y[best_route[0]]], 'k')
    canv.plot(x, y, 'ro')
    title("Distance : {}".format(best_dist))
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
    print("point = {}".format(best['bestpos']))
    print("eval = {}".format(best['bestfit']))


# Factoring functions calculating the total distance
# pre-condition: (x1, y1), (x2, y2) city coordinates
# post-condition: Euclidean distance between 2 cities
def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Calculation function of the system energy,
# pre-condition:
# - coords: coordinates of the path points
# - way: order of course of the cities
# post-condition: VC's Pb: the total distance of the trip
def energyTotale(coords, path):
    energy = 0.0
    coord = coords[path]
    # print(coord)
    for i in range(-1, N - 1):  # on calcule la distance en fermant la boucle
        energy += distance(coord[i], coord[i + 1])
    return energy


# fluctuation function around the "thermal" state of the system: exchange of 2 points
# pre-condition:
# - way: order of course of the cities
# - i, j indices of cities to swap
# post-condition: new order of course
def fluctuationTwo(path, i, j):
    nv = path[:]
    temp = nv[i]
    nv[i] = nv[j]
    nv[j] = temp
    return nv


# create a particle
# one particle is discribed by :
#   - pos : solution list of variables
#   - vit : movement velocity (null at the initialization)
#   - fit :  fitness of the solution
#   - bestpos : best visited position
#   - bestfit : evaluation of the best visited solution
#   - bestvois : best neighbor (global for this version)
def initOne():
    pos = [i for i in range(N)]
    random.shuffle(pos)
    fit = energyTotale(coords, pos)
    return {'vit': [(0, 0)], 'pos': pos, 'fit': fit, 'bestpos': pos, 'bestfit': fit, 'bestvois': []}


# Init of the population (swarm)
def initSwarm(nb):
    return [initOne() for i in range(nb)]


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


#######################

# adding a velocity in discrete case is concatenating the permutations
def add(v1, v2):
    if len(v2) == 0:
        return v1
    if len(v1) == 0:
        return v2

    for t in v2:
        v1.append(t)
    return v1


def times(v1, coeff):
    if len(v1) == 0:
        return []

    v = v1.copy()
    n = len(v1)

    while len(v) < coeff * n:
        v.extend(v1)

    return v[:int(coeff * n)]


def minus(p1, p2):
    p = p1.copy()
    permutations_list = []

    # repeat until p1 has become p2
    while (p != p2):
        for i in range(0, len(p)):
            if p[i] != p2[i]:
                permutations_list.extend([i, p.index(p2[i])])
                p = fluctuationTwo(p, i, p.index(p2[i]))

    return permutations_list


######################


def reshape_to_2d(l):
    a = []
    for t in range(0, len(l), 2):
        a.append([l[t], l[t + 1]])
    return a


# Calculate the velocity and move a particle
def move(particle):
    global ksi, c1, c2, psi, cmax
    velocity = []
    nv = dict(particle)
    print(particle)
    # calculate new velocity
    bp = reshape_to_2d(minus(particle["bestpos"], particle["pos"]))
    # print('bp = {}'.format(bp))
    bv = reshape_to_2d(minus(particle["bestvois"], particle["pos"]))
    # print('bv = {}'.format(bv))

    rbp = times(bp, cmax * random.uniform())
    # print('rbp = {}'.format(rbp))
    rbv = times(bv, cmax * random.uniform())
    # print('rbv = {}'.format(rbv))
    ps = times(particle["vit"], psi)
    # print('ps = {}'.format(ps))

    a = add(ps, rbp)
    # print('1st add : a = {}'.format(a))
    a = add(a, rbv)
    # print('2nd add : a = {}'.format(a))
    velocity = a
    # print('velocity = {}'.format(velocity))
    # apply all permutations of the velocity to the current position
    position = particle["pos"]

    for t in velocity:
        position = fluctuationTwo(position, t[0], t[1])
    # print('particle[pos] = {}'.format(particle['pos']))
    # print('position = {}'.format(position))
    # position = limiting(position)

    nv['vit'] = velocity
    nv['pos'] = position
    nv['fit'] = energyTotale(coords, position)

    return nv


###
## Local Update
##


def get_neiborhood(particle, swarm, nbn):
    neighbors = []
    particle_index = swarm.index(particle)

    # for even numbers:
    if nbn % 2 == 0:
        # neighbors successors in the swarm list
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
        for i in range(particle_index - int(nbn / 2) - 1, particle_index):
            neighbors.append(swarm[i])
    # for odd numbers we will randomly select weather to take successors>predecessors or the inverse
    else:
        if random.uniform(0, 1) > 0.5:
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
###


## MAIN

# Construction of the data from the file
(x, y) = parse(FIC)  # x, y are kept in the state for graphic display
coords = array(list(zip(x, y)), dtype=float)  # We build the array of coordinates (x, y)

# Problem dimensionality
N = len(coords)  # number of cities

# initializing history lists for the final graph
Htemps = []  # temps
Hbest = []  # distance

# initialization of the population
swarm = initSwarm(Nb_particle)
# initialization of the best solution
best = getBest(swarm)
best_cycle = best

for i in range(Nb_cycles):
    # Update informations
    #swarm = [update(e, best_cycle) for e in swarm]

    swarm = [update_from_local_neighbors(e, swarm, 8) for e in swarm]

    # velocity calculations and displacement
    swarm = [move(e) for e in swarm]
    # Update of the best solution
    best_cycle = getBest(swarm)
    if (best_cycle["bestfit"] < best["bestfit"]):
        best = best_cycle
        # draw(best['pos'], best['fit'])

    # historization of data
    Htemps.append(i)
    Hbest.append(best['bestfit'])

    # swarm display
    # if i % 10 == 0:
    # draw(swarm,i)
    # we trace the path of departure
    draw(best['bestpos'], best['bestfit'], x, y)

# END, displaying results
Htemps.append(i)
Hbest.append(best['bestfit'])
# draw(swarm,i)

# displaying result on the console
dispRes(best)
drawStats(Htemps, Hbest)
draw(best['bestpos'], best['bestfit'], x, y)
