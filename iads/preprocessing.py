from nltk.stem import WordNetLemmatizer
from string import punctuation
from unidecode import unidecode
import pandas as pd
from numpy.typing import NDArray


def clean_news(news: pd.DataFrame, stopwords: list[str], delete_stopwords: bool = True, delete_digits: bool = True) -> pd.DataFrame:
    """
    Fait un prétraitement de la base de données 'news' :
        1. Supprime les NaNs
        2. Convertit unicode en ascii
        3. Supprime '\n \r \t' et toute la punctuation 
        4. Supprime les stopwords [si nécessaire]
        5. Supprime les chiffres qui sont seuls (sans lettres à coté) [si nécessaire]

    Remarque : le caractère de punctuation l'apostrophe (') est également supprimé car pas de mots utiles dans 
    Parameters
    ----------
        news             : Dataframe qui contient dataset '20newsgroups'
        stopwords        : Liste des stopwords
        delete_stopwords : Si True, alors les stopwords seront enlevés. Si False, ils seront préservés
        delete_digits    : Si True, alors supprimer les mots qui sont formés uniquement de chiffres. Si False, préserver tels mots

    Returns
    -------
        Dataframe 'news' nettoyé selon les 4 étapes ci-dessus (et avec la colonne 'Message index' ajouté,
            c'est utile pour lemmatisation après)
    """

    def del_stopwords(tokenized_msg: list[str]) -> list[str]:
        return [word for word in tokenized_msg if word not in stopwords and word != "\'"]

    def del_digits(tokenized_msg: list[str]) -> list[str]:
        return [word for word in tokenized_msg if not word.isdigit()]

    # To be able to implode afterwards if necessary --> see later if it is useful
    news = news.reset_index().rename(columns={"index": "Message index"})

    # Delete NaNs
    print("Number of Nans", news[news["messages"].isna()].shape[0])
    news = news[~news["messages"].isna()]

    # Convert unicode to ascii. Delete \n \r \t
    news.loc[:, "messages"] = news["messages"].apply(unidecode).str.replace(r'[\n\r\t]', ' ', regex=True)

    # Delete all punctuation (including ' since no meaningful words with ' were found, just abbreviations)
    punc = punctuation.replace('\'', '')
    # punc = punctuation
    trans_table = str.maketrans(punc, ' ' * len(punc))
    news.loc[:, "messages"] = news["messages"].str.lower().str.translate(trans_table).str.split()

    # Delete stopwords if necessary
    if delete_stopwords:
        news.loc[:, "messages"] = news["messages"].apply(del_stopwords)

    if delete_digits:
        news.loc[:, "messages"] = news["messages"].apply(del_digits)

    # Delete empty messages
    news = news[news["messages"].map(len) > 0]

    return news


def lemmatize_filter(news: pd.DataFrame, min_len: int = 3, min_count: int = 10) -> pd.DataFrame:
    """Lemmatise et filtre les mots des messages présents dans le dataframe 'news'.

    Parameters
    ----------
        news      : Dataframe qui contient le dataset 20newsgroups
        min_len   : La longueur de mot minimale. Les mots de longueur inférieure serong supprimés.
        min_count : Le nombre d'appararition minimale d'un mot. 
            Les mots qui apparaissent moins que 'min_count' seront supprimés.

    Returns
    -------
        Dataframe de même forme que 'news' mais avec les mots lemmatisés et filtrés. 
    """

    words = news.explode("messages")

    # Lemmatisation
    wnl = WordNetLemmatizer()
    unique_words = pd.DataFrame(data={"word": words["messages"].unique()})
    unique_words["lemma"] = unique_words["word"].apply(wnl.lemmatize)
    words = words.merge(unique_words, left_on="messages", right_on="word").drop(columns="word")

    # Filtrage min length
    words = words[words["lemma"].apply(len) >= min_len]

    # Filtrage min count
    count = words["lemma"].value_counts()
    words = words[words["lemma"].isin(count[count >= min_count].index)]

    messages = words.groupby("Message index")["lemma"].apply(" ".join).reset_index()

    news = news.drop(columns=["messages"])

    return news.merge(messages, on="Message index").rename(columns={"lemma": "messages"})


def get_corpus(news: pd.DataFrame) -> list[str]:
    """Construit le corpus des mots présents dans le dataset 'news'

    Parameters
    ----------
        news : Dataframe qui contient le dataset 20newsgroups !néttoyé et lemmatisé!

    Returns
    -------
        Corpus construit
    """

    return news["messages"].str.split().explode().unique().tolist()


def add_real_labels(news: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne dans le dataframe 'news' qui correspond aux vrais labels des messages (en fonction de target).

    Parameters
    ----------
        news : Dataframe qui contient le dataset 20newsgroups

    Returns
    -------
        Dataframe 'news' avec la colonne ajoutée (rien d'autre est modifié)
    """

    categories = dict({
        0: "alt.atheism",
        1: "comp.graphics",
        2: "comp.sys.ibm.pc.hardware",
        3: "comp.sys.mac.hardware",
        4: "comp.windows.x",
        5: "misc.forsale",
        6: "rec.autos",
        7: "rec.motorcycles",
        8: "rec.sport.baseball",
        9: "rec.sport.hockey",
        10: "sci.crypt",
        11: "sci.electronics",
        12: "sci.med",
        13: "sci.space",
        14: "soc.religion.christian",
        15: "talk.politics.guns",
        16: "talk.politics.mideast",
        17: "talk.politics.misc",
        18: "talk.religion.misc",
        19: "comp.os.ms-windows.misc"
    })
    return news.assign(label=news["target"].map(categories))
