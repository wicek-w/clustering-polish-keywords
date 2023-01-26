import pandas as pd
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance, cosine_distance
from nltk import word_tokenize
import advertools as adv
from stempel import StempelStemmer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.metrics.distance import edit_distance as lev
import streamlit as st
from io import BytesIO
import spacy
from sentence_transformers import SentenceTransformer, util
import xlsxwriter

buffer = BytesIO()
stemmer = StempelStemmer.default()
stopwords = (list(adv.stopwords['polish']))
lemmatization_model = spacy.load("pl_core_news_sm")

def stemming_tokenizer(phrases):
    words = word_tokenize(phrases)
    words = [stemmer.stem(word.lower()) for word in words]
    return words

def lemmatization_tokenizer(phrases):
    # words = word_tokenize(phrases)
    words = lemmatization_model(phrases)
    words = [word.lower().lemma_ for word in words]
    return words

def excel_output(results) :
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        results.to_excel(writer, sheet_name='clustered_kw')
    writer.save()
    st.download_button(label="Pobierz swoje wyniki",
                   data=buffer,
                   file_name="clustered_keywords.xlsx",
                   mime="application/vnd.ms-excel")
    return 0
def cluster_morphology(keywords, clustering_type, nr_clusters=1, min_cluster = 2, sensivity = 0.2, distance_type="euclidean", normalization_type ="lemmatization"):
    keywords = list(filter(None, keywords))
    tokenizer = stemming_tokenizer if normalization_type == 'stemming' else lemmatization_tokenizer

    if clustering_type == "k-means Tfidf":
        tfidf_vectorizer = TfidfVectorizer(max_df=0.3, max_features=10000, min_df=0.01, stop_words=stopwords,
                                           tokenizer = tokenizer, use_idf=True, ngram_range=(1, 2))
        tfidf = tfidf_vectorizer.fit_transform(keywords)
        cluster = KMeans(n_clusters=nr_clusters,random_state=0).fit(tfidf).labels_.tolist()
        results = pd.DataFrame(sorted(zip(cluster, keywords)), columns=["cluster_id", "keyword"])

    elif clustering_type == "DBSCAN":
        tfidf_vectorizer = TfidfVectorizer(max_df=0.3, max_features=10000, min_df=0.01, stop_words=stopwords,
                                           tokenizer = tokenizer, use_idf=True, ngram_range=(1, 2))
        tfidf = tfidf_vectorizer.fit_transform(keywords)
        cluster = DBSCAN(eps = sensivity, min_samples = min_cluster, metric = distance_type).fit(tfidf).labels_.tolist()
        results = pd.DataFrame(sorted(zip(cluster, keywords)), columns=["cluster_id", "keyword"])

    elif clustering_type == "GAACluster":
        clusterer = GAAClusterer(nr_clusters)
        vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
        bow = vectorizer.fit_transform(keywords)
        clusters = clusterer.cluster(bow)
        results = pd.DataFrame(sorted(zip(clusters, keywords)), columns=["cluster_id", "keyword"])
    else:
        vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
        bow = vectorizer.fit_transform(keywords)
        if distance_type == "euclidean":
            clusterer = KMeansClusterer(num_means=nr_clusters, distance=euclidean_distance, repeats=5)
        elif distance_type == "cosine":
            clusterer = KMeansClusterer(num_means=nr_clusters, distance=cosine_distance, repeats=5)
        else:
            clusterer = KMeansClusterer(num_means=nr_clusters, distance=lev, repeats=5)

        classified = clusterer.classify(bow)
        results = pd.DataFrame(sorted(zip(classified, keywords)), columns=["cluster_id", "keyword"])

    excel_output(results)
    st.table(results)
    return 0

def clustering_semantic_fast(keywords, cluster_accuracy = 80, min_cluster_size = 2, transformer = 'sdadas/st-polish-paraphrase-from-distilroberta'):
    corpus_sentences_list =[]
    cluster_name_list = []
    cluster_accuracy = cluster_accuracy / 100
    model = SentenceTransformer(transformer)
    corpus_embeddings = model.encode(keywords, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_cluster_size, threshold=cluster_accuracy)

    for nr, cluster in enumerate(clusters):
        for id in cluster[:]:
            cluster_name_list.append("Cluster {}, #{} Elements ".format(nr + 1, len(cluster)))
            corpus_sentences_list.append(keywords[id])

    results = pd.DataFrame(sorted(zip(cluster_name_list, corpus_sentences_list)), columns=["cluster_id", "keyword"])

    excel_output(results)
    st.table(results)
    return 0

def clustering_semantic_kmeans(keywords, transformer = 'sdadas/st-polish-paraphrase-from-distilroberta', num_clusters = 5):
    corpus_sentences_list =[]
    cluster_name_list = []

    model = SentenceTransformer(transformer)
    corpus_embeddings = model.encode(keywords, batch_size=64)
    clusterer = KMeans(n_clusters=num_clusters)
    clusterer.fit(corpus_embeddings)
    cluster_assignment = clusterer.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(keywords[sentence_id])

    for nr, cluster in enumerate(clustered_sentences):
        for kw in cluster[:]:
            cluster_name_list.append("Cluster {}, #{} Elements ".format(nr + 1, len(cluster)))
            corpus_sentences_list.append(kw)

    results = pd.DataFrame(sorted(zip(cluster_name_list, corpus_sentences_list)), columns=["cluster_id", "keyword"])

    excel_output(results)
    st.table(results)
    return 0

def clustering_semantic_agglomerative(keywords, transformer = 'sdadas/st-polish-paraphrase-from-distilroberta'):
    corpus_sentences_list =[]
    cluster_name_list = []
    model = SentenceTransformer(transformer)
    corpus_embeddings = model.encode(keywords, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=6)
    clusterer.fit(corpus_embeddings)

    clusters = clusterer.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(keywords[sentence_id])

    for nr, cluster in clustered_sentences.items():
        for kw in cluster[:]:
            cluster_name_list.append("Cluster {}, #{} Elements ".format(nr + 1, len(cluster)))
            corpus_sentences_list.append(kw)

    results = pd.DataFrame(sorted(zip(cluster_name_list, corpus_sentences_list)), columns=["cluster_id", "keyword"])

    excel_output(results)
    st.table(results)
    return 0
