import pandas as pd
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance, cosine_distance
from nltk import word_tokenize
import advertools as adv
from stempel import StempelStemmer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.metrics.distance import edit_distance as lev
import streamlit as st
import io
import spacy


buffer = io.BytesIO()
stemmer = StempelStemmer.default()
stopwords = (list(adv.stopwords['polish']))
lemmatization_model = spacy.load("pl_core_news_sm")
# nltk.download('punkt')

def stemming_tokenizer(phrases):
    words = word_tokenize(phrases)
    words = [stemmer.stem(word.lower()) for word in words]
    return words

def lemmatization_tokenizer(phrases):
    words = word_tokenize(phrases)
    words = [lemmatization_model(word.lower()).lemma_ for word in words]
    return words

def cluster_morphology(keywords, clustering_type, nr_clusters=1, min_cluster = 2, sensivity = 0.2, distance_type="euclidean", normalization_type ="lemmatization"):
    keywords = list(filter(None, keywords))
    tokenizer = stemming_tokenizer if normalization_type == 'stemming' else lemmatization_tokenizer

    if clustering_type == "k-means Tfidf":
        tfidf_vectorizer = TfidfVectorizer(max_df=0.2, max_features=10000, min_df=0.008, stop_words=stopwords,
                                           tokenizer = tokenizer, ngram_range=(1, 2))
        tfidf = tfidf_vectorizer.fit_transform(keywords)
        cluster = KMeans(n_clusters=nr_clusters,random_state=0).fit(tfidf).labels_.tolist()
        results = pd.DataFrame(sorted(zip(cluster, keywords)), columns=["cluster_id", "keyword"])

    elif clustering_type == "DBSCAN":
        tfidf_vectorizer = TfidfVectorizer(max_df=0.2, max_features=10000, min_df=0.008, stop_words=stopwords,
                                           use_idf=True, ngram_range=(1, 2))
        tfidf = tfidf_vectorizer.fit_transform(keywords)
        cluster = DBSCAN(eps = sensivity, min_samples = min_cluster, metric = distance_type).fit(tfidf).labels_.tolist()
        results = pd.DataFrame(sorted(zip(cluster, keywords)), columns=["cluster_id", "keyword"])

    elif clustering_type == "GAACluster":
        clusterer = GAAClusterer(nr_clusters)
        vec = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
        clusters = clusterer.cluster(vec.keywords, True)
        results = pd.DataFrame(sorted(zip(clusters, keywords)), columns=["cluster_id", "keyword"])
    else:
        vec = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
        if distance_type == "euclidean":
            clusterer = KMeansClusterer(num_means=nr_clusters, distance=euclidean_distance, repeats=5)
        elif distance_type == "cosine":
            clusterer = KMeansClusterer(num_means=nr_clusters, distance=cosine_distance, repeats=5)
        else:
            clusterer = KMeansClusterer(num_means=nr_clusters, distance=lev, repeats=5)

        classified = clusterer.classify(vec.keywords)
        results = pd.DataFrame(sorted(zip(classified, keywords)), columns=["cluster_id", "keyword"])

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        results.to_excel(writer, sheet_name='clustered_kw')
    writer.save()
    st.download_button(label="Pobierz swoje wyniki", data=buffer)

    st.table(results)