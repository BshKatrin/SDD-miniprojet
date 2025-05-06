# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy
from itertools import combinations
import numpy as np
import pandas as pd
from typing import Union, Callable, Dict
import matplotlib.pyplot as plt
# ------------------------
# added kmeans


def inertie_cluster(Ens: Union[np.ndarray, pd.DataFrame]) -> float:
    """Calcule l'inertie inter-cluster (la somme (au carré) des distances euclidiennes des points au centroide).
    Hypothèse: len(Ens)> >= 2

    Parameters
    ----------  
        Ens : ensemble des données qui appartiennent au même cluster

    Returns
    -------
        Inertie inter-cluster.
    """
    centroid = centroide(Ens)
    return np.sum(np.power(dist_euclidienne(Ens, centroid), 2))


def inertie_globale(Base: Union[np.ndarray, pd.DataFrame], U: Dict[int, list[int]]) -> float:
    """
    Calcule l'inertie globale (somme des inerties intra-clusters).

    Parameters
    ----------
        Base : Base de données d'apprentissage
        U    : Dictionnaire d'affectatation (key : cluster, value : liste des points dans ce cluster). Doit être compatible avec 'Base'

    Returns
    -------
        Inertie globale
    """
    if isinstance(Base, pd.DataFrame):
        Base = Base.to_numpy()

    inertie = 0  # pour faire la somme
    for elems in U.values():
        inertie += inertie_cluster(Base[elems])
    return inertie


def init_kmeans(K: int, Ens: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Choisit K points de manière aléatoire. Ces points seront les centroides des clusters (initialisation pour 
        l'algorithme KMeans).

    Parameters
    ----------
        K   : nombre de cluster > 1 et <= n = le nombre d'exemples de 'Ens'
        Ens : Base de données d'apprentissage (contient n exemples).
    """

    if isinstance(Ens, pd.DataFrame):
        Ens = Ens.to_numpy()
    return Ens[np.random.choice(Ens.shape[0], size=K, replace=False)]


def plus_proche(Exe: Union[np.ndarray, pd.Series], Centres: np.ndarray) -> int:
    """Calcule l'indice de plus proche cluster (plus proche centroide)

    Parameters
    ----------
        Exe     : Exemble de base d'apprentissage.
        Centres : Coordonnées des centroides de chaque cluster

    Returns
    -------
        Indice (dans l'array 'Centres') de plus proche centroide
    """

    if isinstance(Exe, pd.Series):
        Exe = Exe.to_numpy()
    return np.argmin(np.linalg.norm(Centres - Exe, axis=1))


def affecte_cluster(Base: Union[pd.DataFrame, np.ndarray], Centres: np.ndarray) -> Dict[int, list[int]]:
    """Calcule l'affectation des points dans la dataset 'Base' dans les clusters (selon le plus proche centroide).

    Parameters
    ----------
        Base    : Base de données d'apprentissage
        Centres : Coordonnées des centroides de chaque cluster

    Returns
    -------
        Dictionnaire d'affectatation (key : cluster, value : liste des points dans ce cluster). Doit être compatible avec 'Base'
    """

    if isinstance(Base, pd.DataFrame):
        Base = Base.to_numpy()

    MA = {c: [] for c in range(len(Centres))}

    for i, ex in enumerate(Base):
        MA[plus_proche(ex, Centres)].append(i)
    return MA


def nouveaux_centroides(Base: Union[np.ndarray, pd.DataFrame], U: Dict[int, list[int]]) -> np.ndarray:
    """
    Recalcule les centroids selon l'affectation des points dans les clusters.

    Parameters
    ----------
        Base : Base de données d'apprentissage
        U    : Dictionnaire d'affectatation (key : cluster, value : liste des points dans ce cluster). Doit être compatible avec 'Base'

    Returns
    -------
        Coordonnées des nouveaux centroides.
    """

    if isinstance(Base, pd.DataFrame):
        Base = Base.to_numpy()
    centroids = []
    for elems in U.values():
        centroids.append(centroide(Base[elems]))
    return np.array(centroids)


def kmoyennes(K: int, Base: Union[np.ndarray, pd.DataFrame], epsilon: float, iter_max: int, verbose: bool = True) -> tuple[np.ndarray, dict[int, list[int]]]:
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
    Algorithme KMeans sur la dataset 'Base'.

    Parameters
    ----------
        K        : nombre de clusters (> 1)
        Base     : Base de données d'apprentissage
        epsilon  : critère de convergence (différence entre l'inertie globale)
        iter_max : nombre d'itération maximum
        verbose  : True s'il faut afficher les étapes d'algorithme. Sinon, False.
    """

    centroids = init_kmeans(K, Base)     # initialisation de K clusters
    U = affecte_cluster(Base, centroids)  # matrice d'affectation
    prev_iner_gl, curr_iner_gl = 0, inertie_globale(Base, U)

    if (verbose):
        print(f"Initialisaton, Inertie :  {curr_iner_gl:4f}")
    for i in range(iter_max):
        centroids = nouveaux_centroides(Base, U)
        U = affecte_cluster(Base, centroids)
        prev_iner_gl = curr_iner_gl
        curr_iner_gl = inertie_globale(Base, U)

        if (verbose):
            print(f"Itération {i+1} Inertie {curr_iner_gl:4f} Différence {np.abs(prev_iner_gl - curr_iner_gl):.4f}")

        # convergence
        if np.isclose(prev_iner_gl, curr_iner_gl, atol=epsilon):
            break

    return centroids, U


# ------------------------

def normalisation(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise les données dans le dataframe 'data'.

    Parameters
    ----------
        data : DataFrame à normaliser. Uniquement les colonnes avec les données numériques doivent être préservées.

    Returns
    -------
        DataFrame avec les données normalisées.
    """

    df = data.copy()
    for column in df.columns:
        min_max = df[column].agg(["min", "max"])
        df[column] = (df[column] - min_max["min"]) / (min_max["max"] -
                                                      min_max["min"]) if min_max["max"] != min_max["min"] else 0
    return df


def dist_euclidienne(X: Union[np.ndarray, pd.DataFrame], Y: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Calcule la distance euclidienne entre les exemples X et Y.
    Les dimensions de 2 vecteurs doivent être valides pour le calcul de distance (i.e. viennent de même espace vectoriel), i.e.
        X.shape = Y.shape.

    Parameters
    ----------
        X, Y : Les vecteurs entre lesquelles la distance euclidienne sera calculée.

    Returns
    -------
        Distance euclidienne entre X et Y.
    """

    return np.linalg.norm(X - Y)


def dist_cosine(X: Union[np.ndarray, pd.DataFrame], Y: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Calcule la distance cosinus entre les exemples X et Y"""
    return 1 - np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))


def dist_manhattan(X, Y):
    """Calcule la distance Manhattan entre les exemples X et Y"""

    return np.sum(np.abs(np.array(X) - np.array(Y)))


def dist_minkowski(X, Y, p=5):
    """Calcule la distance Minkowski entre les exemples X et Y"""

    return np.sum(np.abs(X - Y) ** p) ** (1 / p)


def dist_infinie(X, Y):
    """Calcul de la distance infinie (distance Chebyshev) les exemples X et Y"""

    return np.max(np.abs(X - Y))


def centroide(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Calcule les coordonnées de centroid des exemples dans le DataFrame 'data'.capitalize
        Hypothèse data n'est pas vide

    Parameters
    ----------
        data: DataFrame avec des exemples. DataFrame doit contenir uniquement les valeurs numériques.

    Returns
    -------
        Les coordonnées de centroide.
    """

    return np.mean(data, axis=0)


def dist_centroides(X1, X2, func_dist: Callable = dist_euclidienne):
    """Calcule la distance entre entre 2 groupes de vecteurs, i.e. la distance entre leurs centroids.

    Parameters
    ---------- 
        X1, X2 :  2 vecteurs.
        func_dist : Fonction qui calcule la distance entre 2 vecteurs.

    Returns
    -------
        Distance entre les centroids de ces 2 vecteurs.
    """

    return func_dist(centroide(X1), centroide(X2))

# ---- Clustering hiérarchique ----


def initialise_CHA(data: pd.DataFrame) -> Dict[int, int]:
    """Faire la partition avec chaque élément de dataframe 'data' comme un cluster, i.e. chaque cluster contient 1 élément.""

    Parameters
    ----------
        data : DataFrame avec des exemples.

    Returns
    -------
        Dictionnaire (key : numéro du cluster à partir de 0, value : position (pour y accéder avec .iloc) des exemples dans 'data')
            qui représente la partition.
    """
    return {ind: [ind] for ind in range(data.shape[0])}


def fusionne(df: pd.DataFrame, P0: Dict, func_dist: Callable = dist_euclidienne, verbose: bool = False):
    """Fusionne les deux clusters les plus proche dans la partition P0.
    Hypohtèse : P0 contient au moins 2 clusters.


    Parameters
    ----------
        df        : DataFrame avec des exemples (doit être compatibles avec la partition P0).
        P0        : La partition des exemples de 'data'
        func_dist : Fonction qui calcule la distance.
        verbose   : True s'il faut afficher les étapes de fusion. False, sinon.

    Returns
    -------
        tuples de 3 éléments:
            - partition après fusion (P0 n'est pas modifié, une nouvelle partition est retournée)
            - clés dans P0 de 2 clusters fusionnés
            - distance entre ces 2 clusters fusionnés
    """

    clusters_pair, min_dist = None, np.inf

    # Trouver les clusters les plus proches
    for (c0, c1) in combinations(P0.keys(), 2):
        dist_cent = dist_centroides(df.iloc[P0[c0]], df.iloc[P0[c1]], func_dist)
        if dist_cent < min_dist:
            min_dist = dist_cent
            clusters_pair = (c0, c1)

    # Fusion
    P1 = {cluster: elems for cluster, elems in P0.items() if cluster not in clusters_pair}

    c0, c1 = clusters_pair
    new_cluster = max(P0.keys()) + 1
    P1[new_cluster] = P0[c0] + P0[c1]

    if verbose:
        print(f"fusionne: distance mininimale trouvée entre [{c0}, {c1}]  =  {min_dist}")
        print(f"fusionne: les 2 clusters dont les clés sont [{c0}, {c1}] sont fusionnés")
        print(f"fusionne: on crée la  nouvelle clé {new_cluster} dans le dictionnaire")
        print(f"fusionne: les clés de [{c0}, {c1}] sont supprimées car leurs clusters ont été fusionnés.")

    return P1, clusters_pair, min_dist


def plot_dendrogramme(lst_clusters, title: str):
    """Plotter la dendrogramme selon les clusters donnés par 'lst_clusters'.

    Parameters
    ---------- 
        lst_clusters : Linkage matrix.
        title        : Titre pour le graphique.
    """

    # Paramètre de la fenêtre d'affichage:
    plt.figure(figsize=(30, 15))  # taille : largeur x hauteur
    plt.title(title, fontsize=25)
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme pour notre clustering :
    scipy.cluster.hierarchy.dendrogram(
        lst_clusters,
        leaf_font_size=24.,  # taille des caractères de l'axe des X
    )


def CHA_centroid(df: pd.DataFrame, func_dist: Callable = dist_euclidienne, verbose: bool = False, dendrogramme: bool = False):
    """Effecteur un clustering hiérarchique avec linkage = "centroid" (jusqu'à la création d'un cluster unique).

    Parameters
    ----------
        df           : DataFrame avec des exemples 
        func_dist    : Fonction qui calcule la distance entre 2 vecteurs
        verbose      : True s'il faut afficher les étapes de clustering hiérarchique. False, sinon.
        dendrogramme : True s'il faut plotter dendrogramme. Sinon, False.

    Returns
    -------
        Une liste composée des listes contenant chacune : 
            - les 2 indices d'éléments fusionnés
            - la distance les séparant
            - la somme du nombre d'éléments des 2 éléments fusionnés
    """

    if verbose:
        print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")

    P = initialise_CHA(df)
    result = []
    while (len(P.keys()) >= 2):  # tant que P contient au moins 2 elements
        P_fusionne, (c0, c1), dist = fusionne(df, P, func_dist, verbose)
        n_elements = len(P[c0]) + len(P[c1])
        result.append([c0, c1, dist, n_elements])
        P = P_fusionne

        if verbose:
            print(f"CHA_centroid: une fusion réalisée de {c0} avec {c1} de distance {dist}")
            print(f"CHA_centroid: le nouveau cluster contient {n_elements} exemples")

    if verbose:
        print("CHA_centroid: plus de fusion possible, il ne reste qu'un cluster unique")

    if dendrogramme:
        plot_dendrogramme(result, 'Dendrogramme Centroid Linkage')

    return result


def maj_dist_matrix(dist_matrix: np.ndarray, i0: int, i1: int, indices_map: Dict[int, int], new_cluster: int, min_max: str = "max"):
    """Fonction auxiliaire pour "CHA_complete" et "CHA_simple". Permet de mettre à jour la matrice 'dist_matrix'
    en supprimant les lignes et les colonnes i0 et i1.

    Parameters
    ----------
        dist_matrix : Matrice de distance entre les clusters.
        i0, i1      : les indices des lignes et des colonnes à supprimer (i.e. on fusionne les clusters associés à ces indices).
        indices_map : Dictionnaire de mapping entre les indices de "dist_matrix" et les numéros des clusters.
        new_cluster : Le numéro de nouveau cluster créé.
        min_max     : (="max" ou "min"). Mettre "max" pour CHA_complete et "min" pour CHA_simple.

    Returns:
        dist_matrix et indices_map mis à jour.
    """

    # maj matrice de distance
    n = dist_matrix.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[[i0, i1]] = False  # ne pas prendre ligne i0 et colonne i1

    if min_max == "max":
        cluster_dist = np.maximum(dist_matrix[i0, mask], dist_matrix[mask, i1])
    else:
        cluster_dist = np.minimum(dist_matrix[i0, mask], dist_matrix[mask, i1])

    # Construire la nouveelle matrice
    new_dist_matrix = np.zeros(shape=(n-1, n-1))
    new_dist_matrix[:-1, :-1] = dist_matrix[mask, :][:, mask]  # recopier les elements precedents
    new_dist_matrix[-1, :-1] = cluster_dist  # dernière ligne sera associé au cluster créé
    new_dist_matrix[:-1, -1] = cluster_dist  # de même pour la dernière colonne
    new_dist_matrix[-1, -1] = np.inf

    # maj de mapping

    del indices_map[i0]
    del indices_map[i1]

    indices_map = {ind: cluster for cluster, ind in zip(indices_map.values(), range(0, n-1))}
    indices_map[n-2] = new_cluster

    return new_dist_matrix, indices_map

# -- complete CHA : la distance entre 2 clusters est donné par la distance maximal entre deux points des deux clusters


def CHA_complete(df: Union[np.ndarray, pd.DataFrame], dist_func: Callable = dist_euclidienne, verbose: bool = False, dendrogramme: bool = False):
    """Effecteur un clustering hiérarchique avec linkage = "complete" (jusqu'à la création d'un cluster unique).

    Parameters
    ----------
        df           : DataFrame ou array avec des exemples (lignes = exemples, colonnes = features).
        dist_func    : Fonction qui calcule la distance entre 2 vecteurs.
        verbose      : True s'il faut afficher les étapes de clustering hiérarchique. False, sinon.
        dendrogramme : True s'il faut plotter dendrogramme. Sinon, False.

    Returns
    -------
        Une liste composée des listes contenant chacune : 
            - les 2 indices d'éléments fusionnés
            - la distance les séparant
            - la somme du nombre d'éléments des 2 éléments fusionnés
    """

    if verbose:
        print("CHA_complete: clustering hiérarchique ascendant, version Complete Complete")

    P = initialise_CHA(df)

    # optimisation avec la matrice
    dist_matrix = squareform(pdist(df, dist_func))  # matrice des distance pour complete linkage (i.e. max des dist)
    np.fill_diagonal(dist_matrix, np.inf)  # pour ne pas choisir diagonale

    # Mapping entre les indices de 'dist_matrix' et les numéros des clusters
    indices_map = {index: cluster for cluster, index in zip(P.keys(), range(0, df.shape[0]))}
    result = []
    new_cluster = max(P.keys()) + 1  # numéro de nouveau cluster

    while (len(P.keys()) >= 2):  # tant que P contient au moins 2 elements
        # find min dist
        (i0, i1) = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)

        dist = dist_matrix[i0, i1]

        # Maj de partition P
        c0, c1 = indices_map[i0], indices_map[i1]
        n_elements = len(P[c0]) + len(P[c1])

        result.append([c0, c1, dist, n_elements])
        P[new_cluster] = P[c0] + P[c1]
        P = {cluster: elems for cluster, elems in P.items() if cluster not in (c0, c1)}

        dist_matrix, indices_map = maj_dist_matrix(dist_matrix, i0, i1, indices_map, new_cluster, "max")

        new_cluster += 1

        if verbose:
            print(f"CHA_complete: une fusion réalisée de {c0} avec {c1} de distance {dist}")
            print(f"CHA_complete: le nouveau cluster contient {n_elements} exemples")

    if verbose:
        print("CHA_complete: plus de fusion possible, il ne reste qu'un cluster unique")

    if dendrogramme:
        plot_dendrogramme(result, 'Dendrogramme Complete Linkage')

    return result

# -- Simple CHA : la distance entre 2 clusters est donné par la distance minimale entre deux points des deux clusters


def CHA_simple(df: Union[pd.DataFrame, np.ndarray], func_dist: Callable = dist_euclidienne, verbose: bool = False, dendrogramme: bool = False):
    """Effecteur un clustering hiérarchique avec linkage = "simple" (jusqu'à la création d'un cluster unique).

    Parameters
    ----------
        df           : DataFrame ou array avec des exemples (lignes = exemples, colonnes = features).
        dist_func    : Fonction qui calcule la distance entre 2 vecteurs.
        verbose      : True s'il faut afficher les étapes de clustering hiérarchique. False, sinon.
        dendrogramme : True s'il faut plotter dendrogramme. Sinon, False.

    Returns
    -------
        Une liste composée des listes contenant chacune : 
            - les 2 indices d'éléments fusionnés
            - la distance les séparant
            - la somme du nombre d'éléments des 2 éléments fusionnés
    """

    if verbose:
        print("CHA_simple: clustering hiérarchique ascendant, version Simple Complete")

    P = initialise_CHA(df)
    dist_matrix = squareform(pdist(df, func_dist))  # matrice des distance pour complete linkage (i.e. max des dist)
    np.fill_diagonal(dist_matrix, np.inf)  # pour ne pas choisir diagonale

    # Mapping entre les indices de 'dist_matrix' et les numéros des clusters
    indices_map = {index: cluster for cluster, index in zip(P.keys(), range(0, df.shape[0]))}
    result = []
    new_cluster = max(P.keys()) + 1  # numéro de nouveau cluster

    while (len(P.keys()) >= 2):  # tant que P contient au moins 2 elements
        # find min dist
        (i0, i1) = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)

        dist = dist_matrix[i0, i1]

        # Maj de partition P
        c0, c1 = indices_map[i0], indices_map[i1]
        n_elements = len(P[c0]) + len(P[c1])

        result.append([c0, c1, dist, n_elements])
        P[new_cluster] = P[c0] + P[c1]
        P = {cluster: elems for cluster, elems in P.items() if cluster not in (c0, c1)}

        dist_matrix, indices_map = maj_dist_matrix(dist_matrix, i0, i1, indices_map, new_cluster, "min")

        new_cluster += 1

        if verbose:
            print(f"CHA_simple: une fusion réalisée de {c0} avec {c1} de distance {dist}")
            print(f"CHA_simple: le nouveau cluster contient {n_elements} exemples")

    if verbose:
        print("CHA_simple: plus de fusion possible, il ne reste qu'un cluster unique")

    if dendrogramme:
        plot_dendrogramme(result, 'Dendrogramme Simple Linkage')

    return result

# -- Average CHA : la distance entre 2 clusters est donné par la la moyenne des distances entre tous les points des deux clusters


def CHA_average(df: Union[pd.DataFrame, np.ndarray], func_dist: Callable = dist_euclidienne, verbose: bool = False, dendrogramme: bool = False):
    """Effecteur un clustering hiérarchique avec linkage = "average" (jusqu'à la création d'un cluster unique).

    Parameters
    ----------
        df           : DataFrame ou array avec des exemples (lignes = exemples, colonnes = features).
        dist_func    : Fonction qui calcule la distance entre 2 vecteurs.
        verbose      : True s'il faut afficher les étapes de clustering hiérarchique. False, sinon.
        dendrogramme : True s'il faut plotter dendrogramme. Sinon, False.

    Returns
    -------
        Une liste composée des listes contenant chacune : 
            - les 2 indices d'éléments fusionnés
            - la distance les séparant
            - la somme du nombre d'éléments des 2 éléments fusionnés
    """

    def calc_dist(P, dist_matrix, c0, c1):
        dist = dist_matrix[P[c0], :][:, P[c1]]
        return np.sum(dist) / (len(P[c0]) * len(P[c1]))

    if verbose:
        print("CHA_average: clustering hiérarchique ascendant, version Average Linkage")

    P = initialise_CHA(df)
    dist_matrix = squareform(pdist(df, func_dist))  # matrice des distance entre les points

    result = []
    new_cluster = max(P.keys()) + 1  # numéro de nouveau cluster

    while (len(P.keys()) >= 2):  # tant que P contient au moins 2 elements
        clusters_pairs = list(combinations(P.keys(), 2))
        clusters_dist = [calc_dist(P, dist_matrix, c0, c1) for c0, c1 in clusters_pairs]

        ind = np.argmin(clusters_dist)
        c0, c1 = clusters_pairs[ind]
        dist = clusters_dist[ind]

        n_elements = len(P[c0]) + len(P[c1])
        result.append([c0, c1, dist, n_elements])
        P[new_cluster] = P[c0] + P[c1]
        P = {cluster: elems for cluster, elems in P.items() if cluster not in (c0, c1)}

        new_cluster += 1
        if verbose:
            print(f"CHA_average: une fusion réalisée de {c0} avec {c1} de distance {dist}")
            print(f"CHA_average: le nouveau cluster contient {n_elements} exemples")

    if verbose:
        print("CHA_average: plus de fusion possible, il ne reste qu'un cluster unique")

    if dendrogramme:
        plot_dendrogramme(result, 'Dendrogramme Average Linkage')

    return result


def CHA(DF: Union[pd.DataFrame, np.ndarray], func_dist: Callable = dist_euclidienne, linkage: str = 'centroid', verbose=False, dendrogramme=False):
    """
    Effectuer un clustering hiérarchique des données dans le dataframe 'DF' avec la distance euclidienne.
    Les méthodes de la fusion des clusters possibles (linkage) :
        - centroid : distance entre les clusters et la distance entre leurs centroids
        - complete : distance entre les clusters et la distance entre les points les plus éloignés entre ces clusters
        - simple   : distance entre les clusters et la distance entre les points les plus proches entre ces clusters
        - average  : 

    Parameters
    ----------
        DF : DataFrame avec les données. Il doit contenire uniquement les colonnes avec les données numérique. Les données
            doivent être normalisées. Array de numpy peut être passé uniquement si linkage est complete | simple | average
        func_dist    : Fonction qui calcule la distance entre 2 vecteurs.
        linkage      : méthode de fusion des clusters. Chaîne de caractères : centroid, complete, simple, average
        verbose      : True s'il faut faire des affichages quels clusters sont fusionnés. Sinon, False.
        dendrogramme : True s'il faut plotter dendrogramme. Sinon, False.

    Returns
    -------
        Une liste qui contient
    """

    match linkage:
        case "centroid":
            return CHA_centroid(DF, func_dist, verbose, dendrogramme)
        case "complete":
            return CHA_complete(DF, func_dist, verbose, dendrogramme)
        case "average":
            return CHA_average(DF, func_dist, verbose, dendrogramme)
        case "simple":
            return CHA_simple(DF, func_dist, verbose, dendrogramme)
        case _:
            print(f"Methode \"{linkage}\" n'est pas implémenté.")
            return []
