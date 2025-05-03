# Guide

Ce `README.md` nous permet d'expliquer à quoi correspond chaque fichier supplémentaire qu'on a créé pour ce projet.

## Groupe

- BOGUSH Ekaterina
- CHU Amélie

## Remarques

- On travaille **_sur 10% de dataset_** `20newsgroups` (la distribution des `targets` est préservée). Sinon, nos algorithmes de clustering et classifieurs implementés dans la librairie `iads` prennent trop de temps
- Les classifieurs KNN multi-classe, Perceptron sans biais, Naïf Bayes ont été optimisé pour `sparse array` de `scipy` afin d'améliorer leurs performance.
- Le classifieur Naïf Bayes a été réécrit dès le début (le raisonnement principal était la factorisation du code et l'optimisation pour `sparse array`). Ainsi, on a ajouté le lissage de Laplace pour résoudre le problème de fréquence nulle. On a également remplacé le produit par la somme car on avait le problème que la probabilité devient nulle même si aucune valeur dans la multiplication était 0. Voir plus de détails dans les docs de classifieur (`iads.Classifiers.NaiveBayes`)
- Afin d'analyser et comprendre les clusters on ajoute la colonne `label` dans le dataset. Cette colonne correspond au vrai label du target (par exemple, target 0 c'est le label `alt.atheism`). Source : https://www.kaggle.com/code/yeayates21/20-newsgroups-to-pandas-dataframe. Pour regarder si la correspondce était la bonne, on a regarde les nuages des points pour chaque `target` sur le dataset entier

## Organisation des fichiers

### Notebooks (`/miniprojet`)

**Remarque** : l'ordre dans lequel on a mis les fichiers est un ordre recommandé pour la lecture des notebooks.

- Dans le notebook `sampling.ipynb` on regarde la distribution des `targets` et prend 10% de dataset `20newsgroups` en préservant la distribution des `targets`.
- `supervised_sw_ipynb` contient l'évaluation des classifieurs sur le dataset **avec** stopwords
- `supervised_sw_ipynb` contient l'évaluation des classifieurs sur le dataset **sans** stopwords
- Dans le notebook `unsupervised_no_sw.ipynb` on travaille sur le dataset **sans** stopwords. Ce notebook contient l'explication sur la fléau de la dimension, identification de $k$ pour un algorithme `K-means`, clustering hiérarchique directement sur les targets (voir le notebook pour le raisonnement et le méthode de vectorisation).
- `visualization.ipynb` contient l'analyse des clusters trouvés avec `K-means` pour $k=18$ et la visualisaiton des données.

### Librairie (`/iads`)

On a ajouté les fichiers qui contiennent les fonctions supplémentaires qu'on a écrit pour ce dataset.

- `preprocessing.py` contient les fonctions pour le prétraitement de dataset.
- `vectorization.py` contient les fonctions qui permettent de vectoriser les messages selon
- - Bag-of-words binaire
- - Bag-of-words non binaire
- - Tf-Idf

Ces fonctions ont été optimisées pour `sparse array`.

- `clustering_eval.py` contient les fonctions pour l'évaluation des clusters (utile pour les notebooks `supervised_sw_ipynb` et `visualization.ipynb`)
