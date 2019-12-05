# !/usr/bin/python
# -*- coding: Latin-1 -*-
# TP optim : maximisation de surface
# par l'algorithme PSO
# Mohamed Maatar Project heuristics
# Peio Loubiere pour l'EISTI
# septembre 2017
# usage : python surface.corr1.py 
from typing import List

from scipy import *
from math import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
import pyclipper
from functools import *

# Figure de visualisation de la parcelle
fig = plt.figure(1)
canv = fig.add_subplot(1, 1, 1)
canv.set_xlim(0, 500)
canv.set_ylim(0, 500)
fig2 = plt.figure(2)
canv2 = fig2.add_subplot(1, 1, 1)
canv2.set_xlim(0, 500)
canv2.set_ylim(0, 500)
# ************ Paramètres de la métaheuristique ***PSO=10000 DE=1500*********NB indiv 20*


# ***********************************************************

# ***************** Paramètres du problème ******************
# Différentes propositions de parcelles : 
polygone = ((10, 10), (10, 400), (400, 400), (400, 10))


# polygone = ((10,10),(10,300),(250,300),(350,130),(200,10))
# polygone = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
# polygone = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))

# ***********************************************************

# Transforme le polygone en liste pour l'affichage.
def poly2list(polygone):
    polygonefig = list(polygone)
    polygonefig.append(polygonefig[0])
    return polygonefig


# Constante polygone dessinable
polygonefig = poly2list(polygone)

def get_center(rect):
    return (rect[0][0] + rect[2][0]) / 2, (rect[0][1] + rect[2][1]) / 2

# Fenètre d'affichage
def dessine(polyfig, rectfig):
    global canv, codes
    canv.clear()
    canv.set_xlim(0, 500)
    canv.set_ylim(0, 500)
    # Dessin du polygone
    codes = [Path.MOVETO]
    for i in range(len(polyfig) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(polyfig, codes)
    patch = patches.PathPatch(path, facecolor='orange', lw=2)
    canv.add_patch(patch)

    # Dessin du rectangle
    codes = [Path.MOVETO]
    for i in range(len(rectfig) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(rectfig, codes)
    patch = patches.PathPatch(path, facecolor='grey', lw=2)
    canv.add_patch(patch)
    xo, yo = get_center(rectfig)
    canv.plot(xo, yo, 'ro')
    canv.plot(rectfig[0][0],rectfig[0][1], 'bo')
    canv.plot(rectfig[1][0], rectfig[1][1], 'go')
    # Affichage du titre (aire du rectangle)
    plt.figure(1)
    plt.title("Best all :Aire : {}".format(round(aire(rectfig[:]), 2)))

    plt.draw()
    plt.pause(0.2)


def dessine2(polyfig, rectfig):
    global canv2, codes2
    canv2.clear()
    canv2.set_xlim(0, 500)
    canv2.set_ylim(0, 500)
    # Dessin du polygone
    codes2 = [Path.MOVETO]
    for i in range(len(polyfig) - 2):
        codes2.append(Path.LINETO)
    codes2.append(Path.CLOSEPOLY)
    path = Path(polyfig, codes2)
    patch = patches.PathPatch(path, facecolor='blue', lw=2)
    canv2.add_patch(patch)

    # Dessin du rectangle
    codes2 = [Path.MOVETO]
    for i in range(len(rectfig) - 2):
        codes2.append(Path.LINETO)
    codes2.append(Path.CLOSEPOLY)
    path = Path(rectfig, codes2)
    patch = patches.PathPatch(path, facecolor='green', lw=2)
    canv2.add_patch(patch)
    xo, yo = get_center(rectfig)
    canv2.plot(xo, yo, 'ro')
    canv2.plot(rectfig[0][0], rectfig[0][1], color='orange', marker='o')
    canv2.plot(rectfig[1][0], rectfig[1][1], 'go')
    # Affichage du titre (aire du rectangle)
    plt.figure(2)
    plt.title("Best Cycle : Aire = {}".format(round(aire(rectfig[:]), 2)))

    plt.draw()
    plt.pause(0.2)


# Récupère les bornes de la bounding box autour de la parcelle
def getBornes(polygone):
    lpoly = list(polygone)  # tansformation en liste pour parcours avec reduce
    # return reduce(lambda (xmin,xmax,ymin,ymax),(xe,ye): (min(xe,xmin),max(xe,xmax),min(ye,ymin),max(ye,ymax)),lpoly[1:],(lpoly[0][0],lpoly[0][0],lpoly[0][1],lpoly[0][1]))
    return reduce(lambda acc, e: (min(e[0], acc[0]), max(e[0], acc[1]), min(e[1], acc[2]), max(e[1], acc[3])),
                  lpoly[1:], (lpoly[0][0], lpoly[0][0], lpoly[0][1], lpoly[0][1]))


# Transformation d'une solution du pb (centre/coin/angle) en rectangle pour le clipping
# Retourne un rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
def pos2rect(pos):
    # coin : point A
    xa, ya = pos[0], pos[1]
    # centre du rectangle : point O
    xo, yo = pos[2], pos[3]
    # angle  AÔD
    angle = pos[4]

    # point D : rotation de centre O, d'angle alpha
    alpha = pi * angle / 180  # degre en radian
    xd = cos(alpha) * (xa - xo) - sin(alpha) * (ya - yo) + xo
    yd = sin(alpha) * (xa - xo) + cos(alpha) * (ya - yo) + yo
    # point C : symétrique de A, de centre O
    xc, yc = 2 * xo - xa, 2 * yo - ya
    # point B : symétrique de D, de centre O
    xb, yb = 2 * xo - xd, 2 * yo - yd

    # round pour le clipping
    return ((round(xa), round(ya)), (round(xb), round(yb)), (round(xc), round(yc)), (round(xd), round(yd)))


# Distance entre deux points (x1,y1), (x2,y2)
def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Aire du rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
# 	= distance AB * distance BC
def aire(pos):
    p1 = pos[0]
    p2 = pos[1]
    p3 = pos[2]
    p4 = pos[3]
    return distance(p1, p2) * distance(p3, p2)


# def aire((pa, pb, pc, pd)):
#	return distance(pa,pb)*distance(pb,pc)

# Clipping
# Prédicat qui vérifie que le rectangle est bien dans le polygone
# Teste si 
# 	- il y a bien une intersection (!=[]) entre les figures et
#	- les deux listes ont la même taille et
# 	- tous les points du rectangle appartiennent au résultat du clipping 
# Si erreur (~angle plat), retourne faux
def verifcontrainte(rect, polygone):
    try:
        # Config
        pc = pyclipper.Pyclipper()
        pc.AddPath(polygone, pyclipper.PT_SUBJECT, True)
        pc.AddPath(rect, pyclipper.PT_CLIP, True)
        # Clipping
        clip = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        # all(iterable) return True if all elements of the iterable are true (or if the iterable is empty)
        return (clip != []) and (len(clip[0]) == len(rect)) and all(list(map(lambda e: list(e) in clip[0], rect)))
    except pyclipper.ClipperException:
        # print rect
        return False


# Crée un individu (centre/coin/angle) FAISABLE
# un individu est décrit par votre metaheuristique contenant au moins: 
# 	- pos : solution (centre/coin/angle) liste des variables
#	- eval :  aire du rectangle
#	- ... : autres composantes de l'individu
def initUn(polygone):
    global xmin, xmax, ymin, ymax
    anglemin = 1
    anglemax = 89
    boolOK = False
    pos = []
    while not boolOK:  # tant que non faisable
        xo = random.uniform(xmin, xmax)
        yo = random.uniform(ymin, ymax)

        xa = xo + pow(-1, random.randint(0, 1)) * random.uniform(10, min(xo - xmin, xmax - xo))
        ya = yo + pow(-1, random.randint(0, 1)) * random.uniform(10, min(yo - ymin, ymax - yo))

        angle = random.uniform(anglemin, anglemax)

        pos = [round(xa), round(ya), round(xo), round(yo), angle]
        rect = pos2rect(pos)
        # calcul du clipping
        boolOK = verifcontrainte(rect, polygone)
    ev = aire(pos2rect(pos))
    return {'vit': [0, 0, 0, 0, 30], 'pos': pos, 'eval': ev, 'bestpos': pos, 'bestfit': ev, 'bestvois': []}


# Init de la population
def initPop(nb, polygone):
    return [initUn(polygone) for i in range(nb)]


# Retourne la meilleure particle entre deux : dépend de la métaheuristique
def bestPartic(p1, p2):
    if (p1["eval"] > p2["eval"]):
        return p1
    else:
        return p2


# Retourne une copie de la meilleure particule de la population
def getBest(population):
    return dict(reduce(lambda acc, e: bestPartic(acc, e), population[1:], population[0]))


"""
PSO ALGORITH IMPLEMENTATION
"""
Nb_cycles = 20
Nb_particle = 30
# usual params
psi, cmax = (0.7, 2)


# adding a velocity in discrete case is concatenating the permutations

# Calculate the velocity and move a particle
def move(particle, polygone):
    global ksi, c1, c2, psi, cmax
    boolOK = False
    nv = dict(particle)
    dim = len(particle['vit'])
    position = []
    velocity: List[int] = [0] * len(particle['vit'])
    counter = 0
    while not boolOK:
        angle_is_ok = False
        for i in range(len(particle['vit'])):
            if i < dim - 1:
                velocity[i] = int(particle["vit"][i] * psi +
                                  cmax * random.uniform() * (particle["bestpos"][i] - particle["pos"][i]) +
                                  cmax * random.uniform() * (particle["bestvois"][i] - particle["pos"][i]))
            else:
                angle_tries_counter = 0
                while not angle_is_ok:
                    angle_tries_counter += 1

                    direction = float(particle["vit"][i] * psi +
                                   cmax * random.uniform() * (particle["pos"][i] - particle["bestpos"][i]) +
                                   cmax * random.uniform() * (particle["pos"][i] - particle["bestvois"][i]))
                    velocity[i] = particle['pos'][i] + direction
                    if angle_tries_counter > 7:
                        velocity[i] = random.uniform(particle["pos"][i], particle["bestpos"][i])
                    if velocity[i] < 89 and velocity[i] > 1 :
                        angle_is_ok = True
        position = [0] * dim

        for i in range(dim):
            if i < dim - 1:
                position[i] = round(particle["pos"][i] + velocity[i])
            else:
                position[i] = velocity[i]
        counter += 1
        if counter > 6000:
            position = initUn(polygone)['pos']
        boolOK = verifcontrainte(pos2rect(position), polygone)

        # position = limiting(position)
    rect = pos2rect(position)
    nv['vit'] = velocity
    nv['pos'] = position
    nv["eval"] = round(aire(rect), 2)
    # print(nv)
    return nv


# Update information for the particles of the population (swarm)
def update(particle, best_particle):
    nv = dict(particle)
    if particle["eval"] > particle["bestfit"]:
        nv['bestpos'] = particle["pos"]
        nv['bestfit'] = particle["eval"]
    nv['bestvois'] = best_particle["bestpos"]
    return nv


"""
END OF PSO ALGORITHM
"""
# *************************************** ALGO D'OPTIM ***********************************
# calcul des bornes pour l'initialisation
xmin, xmax, ymin, ymax = getBornes(polygone)
# initialisation de la population (de l'agent si recuit simulé) et du meilleur individu.
pop = initPop(Nb_particle, polygone)
print("---INIT")
l_init = [(i, e) for i, e in enumerate(pop[:5])]
# print([(i, e) for i, e in enumerate(pop[:5])])
best = getBest(pop)
best_cycle = best
velocity_list = [[vel['vit']] for vel in pop]
# boucle principale (à affiner selon la métaheuristique / le critère de convergence choisi)
for i in range(Nb_cycles):
    # Update informations
    pop = [update(e, best_cycle) for e in pop]
    # print([(i, e) for i, e in enumerate(pop[:5])])
    # déplacement
    pop = [move(e, polygone) for e in pop]
    for k, part in enumerate(pop):
        velocity_list[k].append(part['vit'])

    """for i in range(len(l_move)):
        print(l_init[i])
        print(l_update[i])
        print(l_move[i])
        print("-------\n")"""
    # Mise à jour de la meilleure solution et affichage
    best_cycle = getBest(pop)
    print("cycle = " + str(i))
    dessine2(polygonefig, poly2list(pos2rect(best_cycle["pos"])))
    if best_cycle["bestfit"] >= best["bestfit"]:
        best = best_cycle
        print(best)
        dessine(polygonefig, poly2list(pos2rect(best["pos"])))

print("DONE!")
# FIN : affichages
dessine2(polygonefig, poly2list(pos2rect(best_cycle["pos"])))
dessine(polygonefig, poly2list(pos2rect(best["pos"])))
plt.show()
