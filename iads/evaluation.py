# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# Optimisation pour sparse
from scipy.sparse import csr_array, vstack
from sklearn.metrics import pairwise_distances

import numpy as np
import pandas as pd
from copy import deepcopy


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L), np.std(L)


def crossval(X, Y, n_iterations, iteration):
    """Sépare les données en train et test pour la validation croisée [NE regarde PAS la distribution des classes]
        - X, Y : données et label du dataset
        - n_itérations : nombre de tests au total
        - iteration : itération concernée
    """

    sample_size = X.shape[0] // n_iterations  # sample size of TEST dataset
    indices_test = np.arange(iteration * sample_size, (iteration + 1) * sample_size)

    indices_app = np.setdiff1d(np.arange(0, X.shape[0]), indices_test, assume_unique=True)

    Xtest, Ytest = X[indices_test], Y[indices_test]
    Xapp, Yapp = X[indices_app], Y[indices_app]

    return Xapp, Yapp, Xtest, Ytest


def crossval_strat(X, Y, n_iterations, iteration):
    """Sépare les données en train et test pour la validation croisée en respectant la distribution des classes.
        - X, Y : données et label du dataset
        - n_itérations : nombre de tests au total
        - iteration : itération concernée
    """

    def train_test_label(label, X, Y, n_iterations, iteration):
        """La fonction auxiliaire qui pour map qui permet de calculer (Y == label) une seule fois."""
        mask = (Y == label)
        return crossval(X[mask], Y[mask], n_iterations, iteration)

    labels = np.unique(Y)

    res_all = [train_test_label(label, X, Y, n_iterations, iteration) for label in labels]

    train_desc, train_label, test_desc, test_label = zip(*res_all)

    # numpy dense arrays
    if isinstance(X, np.ndarray):
        return (np.concatenate(train_desc),
                np.concatenate(train_label),
                np.concatenate(test_desc),
                np.concatenate(test_label))

    # X is a csr_array
    return (vstack(train_desc),
            np.concatenate(train_label),
            vstack(test_desc),
            np.concatenate(test_label))


def validation_croisee(C, DS, nb_iter, verbose=False, confusion_matrix=False):
    """ Classifieur * tuple[array, array] * int * bool -> tuple[ list[float], float, float]
    Performe la validation croisée.

    Parameters
    ----------
        C : Instance du Classifier
        DS : dataset et les labels. Ces arrays doivent avoir le même nombre des lignes
        nb_iter : Nombre d'itération à faire pour la validation croisée.
        verbose: Si True, afficher chaque itération de la validation croisée. Si False, afficher rien
        confusion_matrix : Si True, calculer la moyenne de la matrice de confusion.
            Si False, ne pas prendre en compte cette matrice.

    Return
    ------
        list[float] : Liste de taille 'nb_iter' avec les taux de bonne classification obtenus
        float : la moyenne des performances
        float : l'écary-type des performances
    """

    perf = []
    C_copy = deepcopy(C)
    classes = np.unique(DS[1])

    if confusion_matrix:  # init -> empty confusion matrix (filled with 0)
        conf_matrix_mean = pd.DataFrame(0, index=pd.Index(
            classes, name="Actual"), columns=pd.Index(classes, name="Predicted"))

    if verbose:
        print("------ affichage validation croisée")

    for i in range(nb_iter):

        desc_train, labels_train, desc_test, labels_test = crossval_strat(*DS, nb_iter, i)
        C_copy.train(desc_train, labels_train)

        data = C_copy.accuracy(desc_test, labels_test, confusion_matrix=confusion_matrix)

        if not confusion_matrix:
            accuracy = data
        else:
            accuracy, conf_matrix = data
            conf_matrix_mean += conf_matrix

        perf.append(accuracy)

        if verbose:
            print(f"Itération {i}: taille de base app.={desc_train.shape[0]}\t"
                  f"taille base test={desc_test.shape[0]}\tTaux de bonne classif: {accuracy:.4f}")
    if verbose:
        print("------ fin affichage validation croisée")

    if not confusion_matrix:
        return perf, *analyse_perfs(np.array(perf))
    return perf, *analyse_perfs(np.array(perf)), conf_matrix_mean / nb_iter
