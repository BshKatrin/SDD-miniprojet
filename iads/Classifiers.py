# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université

Ekaterina BOGUSH
Amélie CHU
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2025

# Optimisation pour sparse
from scipy.sparse import csr_array
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import norm as sparse_norm

import graphviz as gv
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict


def calc_conf_matrix(true_labels: np.ndarray, predict_labels: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    """Calcule la matrice de confusion (sous la forme de crosstab de pandas).

    Parameters
    ----------
        true_labels : les labels réels des examples
        predict_labels : les labels prédits des examples
        classes : classes (uniques) présents dans le dataset

    Returns
        Matrice de confusion sous la forme de DataFrame formé à l'aide de pandas.crosstab
    """

    actual_pd = pd.Categorical(true_labels, categories=classes)
    pred_pd = pd.Categorical(predict_labels, categories=classes)

    return pd.crosstab(actual_pd, pred_pd, normalize=True, rownames=["Actual"], colnames=["Predicted"], dropna=False)


class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set, confusion_matrix=False):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes.

            Si confusion_matrix = False, alors calcule l'accuracy.
            Si confusion_matrix = True, alors calcule la matrice de confusion. 
        """
        # predictions = np.apply_along_axis(self.predict, axis=1, arr=desc_set)
        predictions = np.array([self.predict(row) for row in desc_set])
        accuracy = np.where(predictions == label_set)[0].size / label_set.size
        classes = np.unique(label_set)

        if not confusion_matrix:
            return accuracy

        return accuracy, calc_conf_matrix(label_set, predictions, classes)


class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """

        Classifier.__init__(self, input_dimension)
        self.k = k

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """

        # calc distances euclidien à tous les points
        eucl_distance = np.linalg.norm(self.desc_set - x, axis=1)
        # k plus proches voisins
        k_voisins_ind = np.argpartition(eucl_distance, kth=self.k)[:self.k]
        percentage = np.where(self.lable_set[k_voisins_ind] == 1)[0].size / self.k
        return (percentage - 0.5) * 2  # -0.5 pour centrer autour de 0, *2 pour ramener à [-1, 1]

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        score_x = self.score(x)
        return +1 if score_x > 0 else -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """

        self.desc_set = deepcopy(desc_set)
        self.lable_set = deepcopy(label_set)


class ClassifierKNN_MC(Classifier):
    """Classifieur KNN multi-classe"""

    def __init__(self, input_dimension, k=10, C=2, dist_type="euclidean"):
        """Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer. Par défaut, 10.
                - C (int) : nombre de classes qui sera présents dans le dataset
            Hypothèse : input_dimension > 0, k > 0
        """
        super().__init__(input_dimension)
        self.k = k
        self.classes = C
        self.entropy = False
        self.dist_type = dist_type

    def train(self, desc_set, label_set, bin=False, entropy=False):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set  # deepcopy was deleted since very large dataset are possible
        self.label_set = label_set
        self.entropy = entropy

        if bin:  # binarisation
            self.apply_binarization()
        if entropy:  # delete high entropy
            self.del_high_entropy()

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """

        # Optimisation pour sparse le cas où

        distance = pairwise_distances(self.desc_set, x.reshape(1, -1), metric=self.dist_type).flatten()
        # k plus proches voisins
        k_voisins_ind = np.argpartition(distance, kth=self.k)[:self.k]
        k_labels = self.label_set[k_voisins_ind]
        unique, counts = np.unique(k_labels, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return self.score(x[self.mask]) if self.entropy else self.score(x)

    def apply_binarization(self):
        """ binairer les valeurs d'intensité des pixels.
            mettre à 0 si la valeur <= 0.5, à 1 si > 0.5
        """

        self.desc_set[self.desc_set <= 0.5] = 0
        self.desc_set[self.desc_set > 0.5] = 1

    def del_high_entropy(self):
        """ supprime les pixels avec le taux d'entropie haute
            (10% des valeurs les plus élévées sont éliminiées), la dimension est donc réduit
        """

        dig_values = np.digitize(self.desc_set, [0, 1, 2], right=True)  # floats à 0, 1, 2
        entropies = []
        for pixel in dig_values.transpose():
            probas = np.bincount(pixel) / np.sum(pixel)
            entropies.append(entropy(probas))

        threshold = np.percentile(entropies, 90)  # éliminer 10% des valeurs d'entropie plus élévées
        self.mask = entropies < threshold
        self.desc_set = self.desc_set[:, self.mask]
        self.k = np.count_nonzero(self.mask)


class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        v = np.random.uniform(-1, 1, input_dimension)  # paramètres de la droite sont tirées aléatoirement
        self.w = v / np.linalg.norm(v)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # print("Pas d'apprentissage pour ce classifieur")
        pass

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score_x = self.score(x)
        return +1 if score_x > 0 else -1


class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """

    def __init__(self, input_dimension, learning_rate=0.01, init=True, sparse=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w:
                    - si True (par défaut): initialisation de w à 0,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self, input_dimension)
        self.learning_rate = learning_rate
        self.sparse = sparse

        if not sparse:
            self.w = np.zeros(self.dimension) if init else (2 * np.random.uniform(0, 1, self.dimension) - 1) * 0.001
        else:
            self.w = np.zeros((1, self.dimension)) if init else (
                2 * np.random.uniform(0, 1, (1, self.dimension)) - 1) * 0.001

        # self.allw = [deepcopy(self.w)]  # stockage des premiers poids

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement. Modifie le vecteur w sur place.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        shuffled_index = np.arange(0, desc_set.shape[0])  # les indices
        np.random.shuffle(shuffled_index)  # permuter les indices
        for i in shuffled_index:  # parcours de tous les elements

            if self.predict(desc_set[i]) != label_set[i]:
                self.w = self.w + label_set[i] * desc_set[i] * self.learning_rate

    def train(self, desc_set, label_set, nb_max=30, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """

        norms = []
        while (nb_max):
            w_prev = deepcopy(self.w)
            self.train_step(desc_set, label_set)

            norm = np.linalg.norm(self.w - w_prev)
            norms.append(norm)
            if norm <= seuil:  # convergence
                break
            nb_max -= 1

        return norms

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        if not self.sparse:
            return np.dot(x, self.w)
        return x.dot(self.w.T)[0, 0]

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) > 0 else -1


class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """

    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w:
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        #  Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ", self.w, " learning rate= ", learning_rate)

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        shuffled_index = np.arange(0, desc_set.shape[0])
        np.random.shuffle(shuffled_index)

        for i in shuffled_index:
            score = self.score(desc_set[i])
            # print(score, label_set[i])
            if score * label_set[i] < 1:  # mauvaise classification
                # corriger w
                self.w += self.learning_rate * (label_set[i] - score) * desc_set[i]
                # self.allw.append(deepcopy(self.w))


class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes"""

    def __init__(self, cl_bin, classes):
        """ Constructeur de Classifier
            Argument:
                - cl_bin: classifieur binaire positif/négatif
                - classes: labels possibles de dataset utilisé après
            Hypothèse : input_dimension > 0
        """
        super().__init__(cl_bin.dimension)
        self.classifiers = []
        self.cl_bin = deepcopy(cl_bin)  # pour la référence
        self.classes = deepcopy(classes)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """

        self.classifiers = [deepcopy(self.cl_bin) for _ in range(len(self.classes))]
        for index, cl in enumerate(self.classes):
            Ytmp = np.where(label_set == cl, 1, -1)
            self.classifiers[index].train(desc_set, Ytmp)

    def score(self, x):
        """ rend l'array des scores de prédiction sur x (valeur réelle) selon chaque classifieur binaire
            x: une description
        """
        return np.array([cl.score(x) for cl in self.classifiers])

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return self.classes[np.argmax(self.score(x))]


class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe = None       # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.

    def ajoute_feuille(self, classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe = classe
        self.Les_fils = None   # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple
            on rend la valeur None si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:

            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ', self.nom_attribut, ' -> Valeur inconnue: ', exemple[self.attribut])
            return None

    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.Les_fils:
            total += self.Les_fils[noeud].compte_feuilles()
        return total

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée
        """
        if self.est_feuille():
            g.node(prefixe, str(self.classe), shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i = 0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g, prefixe+str(i))
                g.edge(prefixe, prefixe+str(i), valeur)
                i = i+1
        return g


class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom
        self.seuil = None  #  seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe = None       # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup

    def ajoute_feuille(self, classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe = classe
        self.Les_fils = None   # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe

        # on compare au seuil et descente récursive dans le noeud associé à la valeur de l'attribut

        if exemple[self.attribut] <= self.seuil:
            return self.Les_fils['inf'].classifie(exemple)

        elif exemple[self.attribut] > self.seuil:
            return self.Les_fils['sup'].classifie(exemple)

        else:  # enlever else
            print('\t*** Warning: attribut ', self.nom_attribut, ' -> Valeur inconnue: ', exemple[self.attribut])
            return None

    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.Les_fils:
            total += self.Les_fils[noeud].compte_feuilles()
        return total

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc
            pas expliquée
        """
        if self.est_feuille():
            g.node(prefixe, str(self.classe), shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g, prefixe+"g")
            self.Les_fils['sup'].to_graph(g, prefixe+"d")
            g.edge(prefixe, prefixe+"g", '<=' + str(self.seuil))
            g.edge(prefixe, prefixe+"d", '>' + str(self.seuil))
        return g


def construit_AD(X, Y, epsilon, LNoms=[]):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt
        LNoms : liste des noms de features (colonnes) de description
    """

    entropie_ens = entropie(Y)
    # print("Entropie", entropie_ens)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None

        for num_attr in range(X.shape[1]):
            values, counts = np.unique(X[:, num_attr], return_counts=True)

            attr_entropie = np.sum([entropie(Y[np.argwhere(X[:, num_attr] == value)]) * count / X.shape[0]
                                   for value, count in zip(values, counts)])
            # print(num_attr, attr_entropie)
            # trouver min
            if attr_entropie < min_entropie:
                i_best, min_entropie, Xbest_valeurs = num_attr, attr_entropie, values

        if len(LNoms) > 0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best, LNoms[i_best])
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v, construit_AD(X[X[:, i_best] == v], Y[X[:, i_best] == v], epsilon, LNoms))
    return noeud


def construit_AD_num(X, Y, epsilon, LNoms=[]):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt
        LNoms : liste des noms de features (colonnes) de description
    """

    # dimensions de X:
    (nb_lig, nb_col) = X.shape

    entropie_classe = entropie(Y)
    # print(entropie_classe)
    if (entropie_classe <= epsilon) or (nb_lig <= 1):
        # ARRET : on crée une feuille
        # print("Feuille")
        noeud = NoeudNumerique(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = 0.0  #  meilleur gain trouvé (initalisé à 0.0 => aucun gain)
        i_best = -1  #  numéro du meilleur attribut (init à -1 (aucun))
        Xbest_tuple = None
        Xbest_seuil = None

        for num_attr in range(nb_col):
            # values, counts = np.unique(Y[:, num_attr], return_counts=True)
            # print(values, counts)
            # if counts.size >= 2:  # au moins 2 valeurs uniques

            # partition
            (best_seuil, seuil_ent), liste_vals = discretise(X, Y, num_attr)  # peut renvoyer ((None , +Inf), ([],[]))
            # print("Best", best_seuil)
            # print(num_attr, best_seuil)
            if liste_vals == ([], []):
                partition = ((X, Y), (None, None))
            else:
                partition = partitionne(X, Y, num_attr, best_seuil)

            if entropie_classe - seuil_ent > gain_max:
                Xbest_tuple, Xbest_seuil = partition, best_seuil
                gain_max = entropie_classe - seuil_ent
                i_best = num_attr

        if (i_best != -1):  # Un attribut qui amène un gain d'information >0 a été trouvé
            if len(LNoms) > 0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best, LNoms[i_best])
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data, left_class), (right_data, right_class)) = Xbest_tuple
            noeud.ajoute_fils(Xbest_seuil,
                              construit_AD_num(left_data, left_class, epsilon, LNoms),
                              construit_AD_num(right_data, right_class, epsilon, LNoms))
        else:  # aucun attribut n'a pu améliorer le gain d'information
            # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1, "Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))

    return noeud


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """

    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)  # Appel du constructeur de la classe mère
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None

    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass

    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x
        """
        return self.racine.classifie(x)

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()

    def draw(self, GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """

    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None

    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.racine = construit_AD_num(desc_set, label_set, self.epsilon, self.LNoms)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass

    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x
        """
        return self.racine.classifie(x)

    # def accuracy(self, desc_set, label_set, confusion_matrix=False):  # Version propre à aux arbres
    #     """ Permet de calculer la qualité du système sur un dataset donné
    #         desc_set: ndarray avec des descriptions
    #         label_set: ndarray avec les labels correspondants
    #         Hypothèse: desc_set et label_set ont le même nombre de lignes
    #     """
    #     nb_ok = 0
    #     if confusion_matrix:
    #         predictions = []

    #     for i in range(desc_set.shape[0]):
    #         pred_label = self.predict(desc_set[i, :])
    #         if pred_label == label_set[i]:
    #             nb_ok = nb_ok+1
    #             if confusion_matrix:
    #                 predictions.append(pred_label)

    #     acc = nb_ok/(desc_set.shape[0] * 1.0)

    #     if not confusion_matrix:
    #         return acc
    #     return acc, calc_conf_matrix(label_set, predictions, np.unique(label_set))

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()

    def affiche(self, GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ---------------------------


# ---- Fonction utiles ----- (déplacer dans utils.py ?)


def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    values, counts = np.unique(Y, return_counts=True)
    return values[np.argmax(counts)]


def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    if len(P) == 1 or len(P) == 0:
        return 0  # prolongation par continuité

    P = np.array(P)
    log_res = np.zeros(len(P))  # pour mettre 0 au lieu de inf

    return np.sum(-P * np.log(P, out=log_res, where=(P != 0)) / np.log(len(P)))


def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    _, counts = np.unique(Y, return_counts=True)
    return shannon(counts / np.sum(counts))


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:, num_col])

    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([], []))

    #  Initialisation
    best_seuil = None
    best_entropie = float('Inf')

    #  pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []

    nb_exemples = len(m_class)

    for v in l_valeurs:
        cl_inf = m_class[m_desc[:, num_col] <= v]
        cl_sup = m_class[m_desc[:, num_col] > v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)

        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf)  # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup)  # entropie de l'ensemble des sup

        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
            + (nb_sup / float(nb_exemples)) * val_entropie_sup

        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)

        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v

    return (best_seuil, best_entropie), (liste_coupures, liste_entropies)


def partitionne(m_desc, m_class, n, s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    return ((m_desc[m_desc[:, n] <= s], m_class[m_desc[:, n] <= s]),
            (m_desc[m_desc[:, n] > s], m_class[m_desc[:, n] > s]))


def maj_dist_matrix(dist_matrix: np.array, i0: int, i1: int, indices_map: Dict[int, int], new_cluster: int, min_max: str = "max"):
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


class ClassifierBaggingTree(Classifier):
    def __init__(self, input_dimension, B, sample_percent, entropy_thresh, replace, tree_classifier):
        """Initialise le classifieur Bagging Tree

        Parameters
        ----------
            input_dimension: dimension des données
            B : nombre d'arbres à construire
            sample_percent : pourcentage d'exemples de la base d'apprentissag utilisés pour constituer l'échantillon
            replace: True s'il faut tirer l'échantillon avec remise. False s'il faut le faire sans remise.
            entropy_thresh : seuil d'entropie pour arrêter la construction de chaque arbre
        """

        super().__init__(input_dimension)
        self.B = B
        self.sample_percent = sample_percent
        self.replace = replace
        self.entropy_thresh = entropy_thresh
        self.trees = []
        self.tree_classifier = self.tree_classifier

    def train(self, desc_set, label_set):
        """Construit 'B' arbres entrainés sur les différents échantillon."""

        LabeledSet = (desc_set, label_set)
        m = int(desc_set.shape[0] * self.sample_percent)
        for _ in range(self.B):
            desc, label = echantillonLS(LabeledSet, m, self.replace)
            tree = self.tree_classifier(self.dimension, self.entropy_thresh)
            tree.train(desc, label)
            self.trees.append(tree)

    def score(self, x):
        sm = 0
        for tree in self.trees:
            sm += tree.predict(x)
        return sm / self.B

    def predict(self, x):
        score = self.score(x)
        return +1 if score >= 0 else -1


def tirage(VX, m, avecRemise=False):
    """ VX: vecteur d'indices
        m : nombre d'exemples à sélectionner (hypothèse: m <= len(VX))
        avecRemise: booléen, true si avec remise, ou faux sinon
    """
    return np.random.choice(VX, m, replace=avecRemise)
    # if avecRemise:
    #     return [random.choice(VX) for _ in range(m)]
    # return random.sample(VX, m)


def echantillonLS(LS, m, avecRemise):
    """ LS: LabeledSet (couple de np.arrays)
        m : entier donnant la taille de l'échantillon voulu (hypothèse: m <= len(LS))
        avecRemise: booléen pour le mode de tirage
    """
    (desc, labels) = LS
    indices = tirage(range(len(desc)), m, avecRemise)
    return desc[indices], labels[indices]


class NaiveBayes(Classifier):
    """Classifier Naive Bayes (optimisé pour les matrices creuses). Lissage de Laplace est utilisé avec alpha = 1, K = 2 
    pour éviter le problème de fréquence nulle.
    Source : https://towardsdatascience.com/laplace-smoothing-in-naive-bayes-algorithm-9c237a8bdece/.

    Ainsi, lors des testes on a vu que pour certains exemples la probabilité pour chaque target est 0. 
    Cependant, aucune probabilité P(mot | target) était à 0 mais elles était très petites. 
    Le problème venait de underflow car on multipliait les probabilités trop petites et donc le produit final était 0.

    Selon le site (https://www.geeksforgeeks.org/addressing-numerical-underflow-in-naive-bayes-classification/#understanding-numerical-underflow-issues)
    ce issue peut être résolu en appliquant log sur le produit (log est la fonction croissante donc maximier la proba = 
    maximiser sa valeur en log). 
    Log permet de tranformer le produit en somme qui est plus stable par rapport à la multiplication.
    On utilise les propriétés de log suivantes :
        - log(x * y) = log(x) + log(y)
        - log(x^y) = y * log(x)
    """

    def __init__(self, input_dimension, corpus, classes):
        """Initialise les paramètres de Naive Bayes.
        Les dictionnaires supplémentaires du classifieur:
            - probas_words : Dictionnaire qui stocke les probas P(mot | target). Le dictionnaire est sous la forme
                {target_value : [P(mot 0 | target), ..., P(mot N | target)]}.

            - probas_class : Dictionnaire qui stocke les probas P(target).

        Parameters
        ----------
            input_dimension : dimension des données (= nombre de mots dans le corpus)
            corpus : le corpus
            classes : les classes (uniques) qui seront présents dans le dataset.
        """

        super().__init__(input_dimension)
        self.probas_words = dict()
        self.probas_class = dict()
        self.classes = deepcopy(classes)
        self.corpus = deepcopy(corpus)

    def train(self, desc_set: csr_array, label_set: np.ndarray):
        """Calcule les probas P(mot | target) pour chaque target.
        Remplie le dictionnaire self.probas_words et self.probas_class.

        Parameters
        ----------
            desc_set : Les exemples de dataset déjà vectorisées comme Bag-of-Words !non! binaire.
            label_set : les labels (targets) associés aux exemples de 'desc_set'.
        """

        for cl in self.classes:
            mask = (label_set == cl)
            examples = desc_set[mask, :]

            # P(mot | target) avec lissage de Laplace (alpha = 1, K = 2)
            self.probas_words[cl] = (examples.sum(axis=0).A1 + 1) / (examples.sum() + 2)

            # P(target)
            self.probas_class[cl] = examples.shape[0] / desc_set.shape[0]

    def _get_mask(self, example: csr_array) -> csr_array:
        """Calcule le vecteur similaire à bag-of-words pour un example.

        Parameters
        ----------
            example : Example de dataset déjà vectorisé comme Bag-of-Words !non! binaire

        Returns
        -------
            Masque dont chaque colonne est associé à un mot (même ordre des mots que dans 'example').
            Le masque aura 0 si le mot n'est pas présent dans 'example' et 1 si le mot est présent.
            Ce masque est utilisé pour faire la puissance lors de prédiction selon Naive Bayes (Xmot dans le cours).
        """

        mask = example.copy()
        mask.data = (mask.data > 0).astype(int)
        return mask

    def score(self, example: csr_array) -> dict[int, float]:
        """Calcule P(target | example) pour chaque target présent dans self.classes

        Parameters
        ----------
            example : Example de dataset déjà vectorisé comme Bag-of-Words !non! binaire

        Returns
        -------
            Dictionnaire dont chaque clé est une valeur de target et la valeur associée est la probabilité 
            P(target | example).
        """

        mask = self._get_mask(example).toarray().flatten()
        scores = dict()

        for cl in self.classes:
            # Problème --> pas stable
            # pow1 = np.power(self.probas_words[cl], mask)
            # pow0 = np.power(1-self.probas_words[cl], 1-mask)
            # scores[cl] = np.prod(pow1) * np.prod(pow0) * self.probas_class[cl]

            # Version log
            pow1 = np.log(self.probas_words[cl]) * mask         # car log(x^y) = y * log(x)
            pow0 = np.log(1-self.probas_words[cl]) * (1-mask)

            scores[cl] = np.sum(pow1) + np.sum(pow0) + np.log(self.probas_class[cl])  # car log(x * y) = log(x) + log(y)

        return scores

    def predict(self, example: csr_array):
        """Prédit le classe d'example selon les probabilités P(classe | example). 
        Le classe prédit est celui qui a la probabilité la plus élevée.

        Parameters
        ----------
            example : Example de dataset déjà vectorisé comme Bag-of-Words !non! binaire

        Returns
        -------
            Classe le plus probable d'example donné.
        """

        scores = self.score(example)
        return max(scores, key=scores.get)
