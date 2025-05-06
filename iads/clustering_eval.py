import itertools
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_array
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from wordcloud import WordCloud
import iads.Clustering as clust


def distribution_target(true_targets: np.ndarray, assigned_clusters: np.ndarray, true_labels: np.ndarray) -> Figure:
    """Plotte la distribution (heatmap) des targets dans les clusters obtenus :
        sur axe Y         : targets
        sur axe X en haut : le nombre d'exemples dans chaque cluster
        sur axe X en bas  : le numéro du cluster

    Parameters
    ----------
        Les 3 paramètres passés sont associés aux exemples dans le même ordre -> tous les arrays sont de même taille.
        true_targets      : Les vrais targets des exemples (les targets varient de 0 à M-1 avec M : le nombre de targets)
        assigned_clusters : Les numéros des clusters assignés à chaque exemple 
        true_labels       : Les labels à mettre sur l'axe de Y. Dans le cadre de dataset 20newsgroups ce sont des labels
            de type alt.altheism, comp.graphics etc.

    Returns
    -------
        Figure de matplotlib avec heatmap plotté (utile pour la sauvegarde).
    """

    actual_pd = pd.Categorical(true_targets, categories=np.unique(true_targets))

    clusters, nb_points = np.unique(assigned_clusters, return_counts=True)
    pred_pd = pd.Categorical(assigned_clusters, categories=clusters)

    conf = pd.crosstab(actual_pd, pred_pd,
                       normalize=False,
                       colnames=["Clusters obtenus"],
                       rownames=["Nombre d'éléments d'un cluster d'origine"],
                       dropna=False)

    # Mettre les vrais labels sur l'axe de y
    conf.index = np.unique(true_labels)

    fig = plt.figure(figsize=(10, 6))
    ax = sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', cbar=False)

    # Mettre en haut le nombre de points par cluster
    for i, cluster in enumerate(conf.columns):
        count = nb_points[cluster]
        ax.text(x=i + 0.5, y=-0.5, s=str(count), ha='center', va='bottom', fontsize=12)

    ax.annotate("Nombre de points par cluster :", xy=(-0.15, 1.025),
                xycoords="axes fraction", ha='center', va='bottom', fontsize=12)
    fig.suptitle("Distribution des labels par cluster", fontsize=15)

    plt.tight_layout()
    plt.show()
    return fig


def predict_label(affectation: dict[int, list[int]]) -> list[int]:
    """"Convertit le dictionnaire des affectations des exemples aux clusters en une liste
    telle que l'élément d'indice i correspond au cluster auquel l'exemple i a été affecté.

    Parameters
    ----------
        affectation : Dictionnaire d'affectation dont chaque clé indique le numéro du cluster 
            et la valeur associée est une liste des indices des éléments

    Returns
    -------
        Affectation (sous forme de la liste) de chaque exemple au cluster
    """

    aff = [(i, cluster) for cluster, ids in affectation.items() for i in ids]
    indices, clusters = zip(*aff)

    cluster_labels = np.full(max(indices) + 1, -1, dtype=int)  # list size of labels

    cluster_labels[list(indices)] = list(clusters)
    return cluster_labels


def get_n_components_PCA(vectors: csr_array, explained_var: float) -> int:
    """Trouve le nombre de dimensions suffisant pour expliquer une proportion donnée de la variance
        lors de la réduction de dimension avec PCA.

    Parameters
    ----------
        vectors : les vecteurs associés aux exemples (lignes = exemples, colonnes = features).
        explained_var : la proportion de la variance qu'il faut expliquer

    Returns
    ------- 
        Le nombre de dimension suffisant pour expliquer 'explained_var'  de la variance 
        [= le paramtère n_components de sklearn.decomposition.PCA].
    """

    model = PCA()
    model.fit(vectors)
    cumsum_var = model.explained_variance_ratio_.cumsum()
    index = np.argmax(cumsum_var >= explained_var)
    return index + 1


def plot_clusters_wordcloud(news: pd.DataFrame, clusters_labels: list[int], stopwords: list[str], n_clusters: int = 18) -> Figure:
    """Plotte le nuage de mots pour chaque cluster 

    Parameters
    ----------
        news            : Dataframe associé au dataset 20newsgroups qui contient les vrais (non vectorisés) messages après le nettoyage.
        clusters_labels : Array qui représente une affectation des exemples aux clusters. La taille de la liste doit être la même
            que le nombre des lignes dans dataframe 'news' (+ même ordre)
        stopwords       : Liste de stopwords
        n_clusters      : Le nombre de clusters au total.

    Returns
    -------
        Figure de matplotlib avec heatmap plotté (utile pour la sauvegarde)
    """

    fig, axes = plt.subplots(3, 6, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(n_clusters):
        ax = axes[i]

        cmap = ListedColormap(cm.get_cmap('tab20').colors[:6])

        text = " ".join(news[clusters_labels == i]['messages'].values.flatten())
        wordcloud = WordCloud(background_color='white', stopwords=stopwords,
                              max_words=100, colormap=cmap).generate(text)

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Cluster {i}', fontsize=18)
    fig.suptitle("Nuage de mots par cluster", fontsize=20)
    plt.tight_layout()
    plt.show()
    return fig


def separabilite(centers: np.ndarray, func_dist: Callable = clust.dist_euclidienne) -> float:
    """Calcule la métrique 'séparabilité' des clusters. 
    La séparabilité est définie comment la somme des distances entre chaque paire de centroïdes des clusters.

    Parameters
    ----------
        centers   : Les vecteurs des centroïdes des clusters 
        func_dist : La fonction qui calcule la distance entre 2 vecteurs. Par défaut : distance euclidienne.

    Returns
    -------
        La séparibilité des clusters.
    """

    index = range(len(centers))
    pairs = list(itertools.combinations(index, 2))

    sep_score = 0
    min_dist = np.inf
    max_dist = -1

    for i, j in pairs:
        distance = func_dist(centers[i], centers[j])
        sep_score += distance

        min_dist = min(min_dist, distance)
        max_dist = max(max_dist, distance)

    return sep_score, min_dist, max_dist  # min_dist pour séparabilité, max_dist pour Dunn 2


def min_dist_clusters(Base: Tuple[np.ndarray, pd.DataFrame], partition: dict[int, list[int]]) -> float:
    """Calcule la distance minimale entre 2 clusters. La distance entre 2 clusters 
    est définie comme le minimum des distances entre 2 points de 2 clusters.

    Parameters
    ----------
        Base        : Dataset des exemples sur lequel le clustering K-means a été appliqué.
        partition   : Affectation des exemples aux clusters. Dictionnaire qui a pour la clé : le numéro des clusters
            et la valeur associée : la liste des indices des exemples affectés à ce cluster.

    Returns
    -------
        La distance minimale entre 2 clusters. 
    """

    index = partition.keys()  # indice des clusters

    if isinstance(Base, pd.DataFrame):
        Base = Base.to_numpy()

    # calculer les distances entre tous les points des pairs
    clusters = {k: Base[v] for k, v in partition.items()}

    min_dist = np.inf

    for i, j in list(itertools.combinations(index, 2)):
        clust1, clust2 = clusters[i], clusters[j]
        min_dist = min(np.min(cdist(clust1, clust2)), min_dist)

    return min_dist


def co_distance(centers: np.array, Base: pd.DataFrame, partition: dict[int, list[int]], func_dist: Callable = clust.dist_euclidienne) -> float:
    """Calcule la métrique 'co-distance' pour les clusters. 
    Co-distance est composée de 2 sommes :
        - Pour chaque cluster il faut calculer la somme des points dans ce cluster vers son centroïd (notons cette somme Di)
        - Ensuite, il faut faire la somme sur les Di pour chaque cluster i

    Parameters
    ---------- 
        centers   : Les vecteurs des centroïdes des clusters 
        Base        : Dataset des exemples sur lequel le clustering K-means a été appliqué.
        partition   : Affectation des exemples aux clusters. Dictionnaire qui a pour la clé : le numéro des clusters
            et la valeur associée : la liste des indices des exemples affectés à ce cluster.
        func_dist : La fonction qui calcule la distance entre 2 vecteurs. Par défaut : distance euclidienne

    Returns
    -------
        Co-distance calculée pour telle affectation au clusters
    """
    co_dst_score = 0

    if isinstance(Base, pd.DataFrame):
        Base = Base.to_numpy()

    for clust, index in partition.items():
        points = Base[index]

        distances = np.array([func_dist(point, centers[clust]) for point in points])
        co_dst_score += np.sum(distances)

    return co_dst_score


def variation_k_evalution(vect: Tuple[np.ndarray, pd.DataFrame], range_max: int = 26, verbose=False):
    """Évalue (selon K-means, pour k allant de 1 à 'range_max'-1) pour des messages vectorisés les métriques de la qualité 
    de clustering suivantes :
        - inertie globale
        - co-distance
        - séparabilité
        - indice de Xie-Beni
        - indice de Dunn 

    Parameters
    ----------
        vect      : La matrice de vectorisation des messages (lignes = messages, colonnes = mots)
        range_max : La valeur de K maximale à prendre (K sera dans l'intervalle [1, range_max[)
        verbose   : Si True, alors les étapes de clustering K-means seront affichés.

    Returns
    -------
        6 valeurs sont rétournées :
            np.ndarray[int] : les valeurs de K évaluées
            list[float]     : l'inértie globale pour chaque k
            list[float]     : co-distance pour chaque k 
            list[float]     : séparabilité pour chaque k 
            list[float]     : indice de Xie-Beni pour chaque k 
            list[float]     : indice de Dunn pour chaque k 

    """

    k_values = np.arange(1, range_max)

    k_global_inertia = []
    k_co_dist = []
    k_separability = []
    k_XB_index = []
    k_Dunn_index = []

    for k in k_values:
        centres, affectation = clust.kmoyennes(k, vect, 0, 1000, verbose)

        inertie_glob = clust.inertie_globale(vect, affectation)
        k_global_inertia.append(inertie_glob)

        sep_score, semin, d_max = separabilite(centres)  # semin dist min between 2 centroids
        k_separability.append(sep_score)

        co_dist = co_distance(centres, vect, affectation)
        k_co_dist.append(co_dist)

        k_XB_index.append(inertie_glob/semin)

        k_Dunn_index.append(co_dist/semin)

    return k_values, k_global_inertia, k_co_dist, k_separability, k_XB_index, k_Dunn_index


def scores_plot(title: str, k_values, k_global_inertia, k_co_dist, k_separability, k_XB_index, k_Dunn_index) -> Figure:
    """Plotte les métriques d'évaluation de la qualité du clustering K-means en fonction de K.
    Sur une figure 2 axes sont plottés (à gauche et à droite)
        - Sur l'axe 1 (à gauche) les métriques séparabilité, inertie globale, co-distance sont plottées
        - Sur l'axe 2 (à droite) les métriques indice de Xie-Beni, indice de Dunne sont plottés.

    Parameters
    ----------
        np.ndarray[int] : les valeurs de K évaluées
        list[float]     : l'inértie globale pour chaque k
        list[float]     : co-distance pour chaque k 
        list[float]     : séparabilité pour chaque k 
        list[float]     : indice de Xie-Beni pour chaque k 
        list[float]     : indice de Dunn pour chaque k 

    Returns
    -------
        Figure de matplotlib avec heatmap plotté (utile pour la sauvegarde).
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
    fig.suptitle(title, fontsize=14)

    # Plot 1
    data = pd.DataFrame(data={"k": k_values, "Inertie globale": k_global_inertia,
                        "Séparabilité": k_separability, "Co-distance": k_co_dist})
    data = data.melt(id_vars="k", value_vars=["Inertie globale", "Séparabilité", "Co-distance"],
                     var_name="Métrique", value_name="Valeur de métrique")
    sns.lineplot(data=data, x="k", y="Valeur de métrique", hue="Métrique", ax=ax1)
    ax1.set_xlabel("Nombre de clusters (k)")

    ax1.legend(loc="lower left", frameon=False)

    # Plot 2 : Indice de Dunn
    data = pd.DataFrame(data={"k": k_values,
                        "Indice de Xie-Bene": k_XB_index, "Indice de Dunn": k_Dunn_index})
    data = data.melt(id_vars="k", value_vars=["Indice de Dunn", "Indice de Xie-Bene"],
                     var_name="Métrique", value_name="Valeur de métrique")
    sns.lineplot(data=data, x="k", y="Valeur de métrique", hue="Métrique", ax=ax2)
    ax2.set_xlabel("Nombre de clusters (k)")
    ax2.legend(loc="lower left", frameon=False)

    return fig


def get_target_vectors(M: csr_array, news: pd.DataFrame) -> np.ndarray:
    """Calcule la vectorisation de chaque targets en fonction de la vectorisation des exemples associés.
    La vectorisaton de chaque target est calculée en prenant la moyenne des vecteurs des exemples qui ont ce target.

    Parameters
    ----------
        M    : La matrice de vectorisation des exemples donné dans le dataframe 'news' (lignes = exemples, colonnes = features]
        news : Dataframe associé au dataset 20newsgroups

    Returns
    -------
        Matrice de vectorisation de chaque target (lignes = targets). Les targets sont placés dans l'ordre croissant, i.e. 
            la ligne 0 est la vectorisation de target qui a la plus petite valeur.
    """

    targets_vectors = []
    values = news["target"].values
    for target in np.sort(news["target"].unique()):
        mask = (values == target)
        targets_vectors.append(M[mask].mean(axis=0))
    return np.array(targets_vectors)
