from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import PCA
import iads.Clustering as clust
import itertools
from typing import Callable, Tuple
from scipy.spatial.distance import cdist


def distribution_target(true_targets: np.ndarray, assigned_clusters: np.ndarray, true_labels: np.ndarray):
    """Affiche la distribution des targets dans les clusters obtenus"""

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


def predict_label(affectation: dict) -> list:
    """Renvoie les labels prédit par l'affectation k-means pour chaque exemple"""

    aff = [(i, cluster) for cluster, ids in affectation.items() for i in ids]
    indices, clusters = zip(*aff)

    cluster_labels = np.full(max(indices) + 1, -1, dtype=int)  # list size of labels

    cluster_labels[list(indices)] = list(clusters)
    return cluster_labels


def get_n_components_PCA(vectors: csr_array, explained_var: float) -> int:
    """Get a number of parameters for PCA to explain 'explained_var'"""
    model = PCA()
    model.fit(vectors)
    cumsum_var = model.explained_variance_ratio_.cumsum()
    index = np.argmax(cumsum_var >= explained_var)
    return index + 1


def plot_clusters_wordcloud(news, clusters_labels, stopwords, n_clusters=18):
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


def separabilite(centers: np.array, func_dist: Callable = clust.dist_euclidienne) -> float:
    """ Calcule la somme des distances entre 2 centroids pour toutes les distances pair à pair
    Renvoie la somme desistances, la distance minimal et maximal entre les clusters """
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


def min_dist_clusters(Base: Tuple[np.ndarray, pd.DataFrame], affectation: dict) -> float:
    """Renvoie la distance minimale entre 2 clusters, la distance entre 2 clusters 
    étant le min des distances entre 2 points de 2 clusters."""

    index = affectation.keys()  # indice des clusters

    if isinstance(Base, pd.DataFrame):
        Base = Base.to_numpy()

    # calculer les distances entre tous les points des pairs
    clusters = {k: Base[v] for k, v in affectation.items()}

    min_dist = np.inf

    for i, j in list(itertools.combinations(index, 2)):
        clust1, clust2 = clusters[i], clusters[j]
        min_dist = min(np.min(cdist(clust1, clust2)), min_dist)

    return min_dist


def co_distance(centers: np.array, Base: pd.DataFrame, partition: dict, func_dist: Callable = clust.dist_euclidienne) -> float:
    """Renvoie la co distance, somme distance des points au centroid"""
    co_dst_score = 0

    if isinstance(Base, pd.DataFrame):
        Base = Base.to_numpy()

    for clust, index in partition.items():
        points = Base[index]

        distances = np.array([func_dist(point, centers[clust]) for point in points])
        co_dst_score += np.sum(distances)

    return co_dst_score


def variation_k_evalution(vect: Tuple[np.ndarray, pd.DataFrame], range_max=26, verbose=False):
    """Evalue pour un espace défini des exemples (bag of words, frequences ou tf_idf), les métriques définit précédemment:
    - inertie globale
    - co-distance
    - séparabilité
    - index de Xie-Beni
    - index de Dunn

    pour k-means avec k variant de 1 à range_max
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

        # d_min = min_dist_clusters(bow_no_stopw, affectation)

        # # Dunn index 2
        # k_Dunn_index2.append(d_min/d_max)

        # XieBeni index = glob_inertia / semin
        k_XB_index.append(inertie_glob/semin)

        # Dunn index = co_dist / semin
        k_Dunn_index.append(co_dist/semin)

    return k_values, k_global_inertia, k_co_dist, k_separability, k_XB_index, k_Dunn_index


def scores_plot(title, k_values, k_global_inertia, k_co_dist, k_separability, k_XB_index, k_Dunn_index):
    # affichage des résultats

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
    fig.suptitle(title, fontsize=14)

    # Plot 1
    data = pd.DataFrame(data={"k": k_values, "Inertie globale": k_global_inertia,
                        "Séparabilité": k_separability, "Co-distance": k_co_dist})
    data = data.melt(id_vars="k", value_vars=["Inertie globale", "Séparabilité", "Co-distance"],
                     var_name="Métrique", value_name="Valeur de métrique")
    sns.lineplot(data=data, x="k", y="Valeur de métrique", hue="Métrique", ax=ax1)
    ax1.set_xlabel("Nombre de clusters (k)")
    # ax1.set_title("Indice de Xie-Beni")
    ax1.legend(loc="lower left", frameon=False)

    # Plot 2 : Indice de Dunn
    data = pd.DataFrame(data={"k": k_values,
                        "Indice de Xie-Bene": k_XB_index, "Indice de Dunn": k_Dunn_index})
    data = data.melt(id_vars="k", value_vars=["Indice de Dunn", "Indice de Xie-Bene"],
                     var_name="Métrique", value_name="Valeur de métrique")
    sns.lineplot(data=data, x="k", y="Valeur de métrique", hue="Métrique", ax=ax2)
    ax2.set_xlabel("Nombre de clusters (k)")
    # ax2.set_title("Indice de Dunn")
    ax2.legend(loc="lower left", frameon=False)
    return fig


def get_target_vectors(M: csr_array, news: pd.DataFrame) -> np.ndarray:
    targets_vectors = []
    values = news["target"].values
    for target in np.sort(news["target"].unique()):
        mask = (values == target)
        targets_vectors.append(M[mask].mean(axis=0))
    return np.array(targets_vectors)
