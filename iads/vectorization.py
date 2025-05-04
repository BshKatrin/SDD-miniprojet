from collections import Counter
from scipy.sparse import csr_array
import pandas as pd
import numpy as np


def get_bow_vect(news: pd.DataFrame, corpus: list[str], binary: bool = False) -> csr_array:
    """
    Hypothèse : 

    Remarque : contruction avec dok_array et ensuite la conversion en csr_array prenait plus de temps
    (0.8 sec pour 10% de dataset) que la construction de csr_array directement (0.0sec pour 10% de dataset) à partir des listes 
    data et indices de rows (indices), cols (indptr).

    Source : scipy docs sur csr_array + https://stackoverflow.com/questions/43828704/how-to-cluster-sparse-data-using-sklearn-kmeans
    """

    corpus_mapping = {word: i for i, word in enumerate(corpus)}
    # vects = dok_array((news.shape[0], len(corpus)), dtype=int)

    # for row_ind, msg in enumerate(news["messages"].str.split()):
    #     for word in msg:
    #         if binary:
    #             vects[row_ind, corpus_mapping[word]] = 1
    #         else:
    #             vects[row_ind, corpus_mapping[word]] += 1
    # return vects.tocsr()

    data, rows, cols = [], [], []  # pour construction de csr_array

    for row_ind, msg in enumerate(news["messages"].str.split()):
        if not binary:
            counter = Counter(msg)
        for word in set(msg):
            if word not in corpus_mapping:  # mot doit être dans le corpus
                continue
            col_ind = corpus_mapping[word]

            if binary:
                rows.append(row_ind)
                cols.append(col_ind)
                data.append(1)
            else:
                rows.append(row_ind)
                cols.append(col_ind)
                data.append(counter[word])

    return csr_array((data, (rows, cols)), shape=(news.shape[0], len(corpus)), dtype=int)


def get_tfidf_vect(news: pd.DataFrame, corpus: list[str]) -> csr_array:
    data, rows, cols = [], [], []  # pour construction de csr_array
    corpus_mapping = {word: i for i, word in enumerate(corpus)}
    idf_words = np.zeros(len(corpus))

    # TF
    for row_ind, msg in enumerate(news["messages"].str.split()):
        counter = Counter(msg)
        for word in set(msg):
            if word not in corpus_mapping:  # mot doit être dans le corpus
                continue
            col_ind = corpus_mapping[word]
            rows.append(row_ind)
            cols.append(col_ind)
            data.append(counter[word])
            idf_words[col_ind] += 1

    # Multiplication par idf
    idf_words = np.log((1 + news.shape[0]) / (1 + idf_words))
    return csr_array((data, (rows, cols)), shape=(news.shape[0], len(corpus)), dtype=float).multiply(idf_words).tocsr()


def normalize(M: csr_array) -> csr_array:
    """Normalize chaque exemple (row) de M selon la norme L2 (norme euclidienne), i.e.
    chaque example sera représenté par le vecteur de norme 1."""

    l2_norms = np.sqrt(M.multiply(M).sum(axis=1))
    return M.multiply(1 / l2_norms.reshape(-1, 1)).tocsr()
