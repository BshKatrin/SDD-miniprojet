# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
Ekaterina BOGUSH
Amélie CHU
"""


# Fonctions utiles
# Version de départ : Février 2025

# import externe
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def genere_dataset_uniform(d, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        d: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """

    return np.random.uniform(binf, bsup, (2*n, d)), np.array([-1] * n + [+1] * n)


def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    data_neg = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    data_pos = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    return np.vstack((data_neg, data_pos)), np.array([-1] * nb_points + [+1] * nb_points)


def plot2DSet(desc, labels, nom_dataset="Dataset", avec_grid=False):
    """ ndarray * ndarray * str * bool-> affichage
        nom_dataset (str): nom du dataset pour la légende
        avec_grid (bool) : True si on veut afficher la grille
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """

    data_neg = desc[labels == -1]
    data_pos = desc[labels == +1]
    print(data_pos, data_neg)
    plt.scatter(data_neg[:, 0], data_neg[:, 1], marker="o", color="gold", label="classe -1")
    plt.scatter(data_pos[:, 0], data_pos[:, 1], marker="x", color="navy", label="classe +1")
    plt.title(nom_dataset)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("y1")
    if avec_grid:
        plt.grid()
    plt.show()


def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax = desc_set.max(0)
    mmin = desc_set.min(0)
    x1grid, x2grid = np.meshgrid(np.linspace(mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))
    grid = np.hstack((x1grid.reshape(x1grid.size, 1), x2grid.reshape(x2grid.size, 1)))

    # calcul de la prediction pour chaque point de la grille
    res = np.array([classifier.predict(grid[i, :]) for i in range(len(grid))])
    res = res.reshape(x1grid.shape)
    # tracer des frontieres
    #  colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid, x2grid, res, colors=["darksalmon", "skyblue"], levels=[-1000, 0, 1000])


def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    gauche_desc, gauche_label = genere_dataset_gaussian(np.array([1, 1]), np.array(
        [[var, 0], [0, var]]), np.array([0, 1]), np.array([[var, 0], [0, var]]), n)
    droite_desc, droite_label = genere_dataset_gaussian(np.array([0, 0]), np.array(
        [[var, 0], [0, var]]), np.array([1, 0]), np.array([[var, 0], [0, var]]), n)

    return np.concatenate((droite_desc, gauche_desc), axis=0), np.concatenate((droite_label, gauche_label), axis=0)
