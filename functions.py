import pandas as pd
import numpy as np
from nltk.cluster import GAAClusterer, euclidean_distance, cosine_distance, KMeansClusterer
from nltk import word_tokenize
import advertools as adv
from stempel import StempelStemmer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.metrics.distance import edit_distance
import streamlit as st
from io import BytesIO
import spacy
from sentence_transformers import SentenceTransformer, util
import xlsxwriter
import scipy
import nltk
from sklearn.metrics.cluster import silhouette_score, adjusted_rand_score


buffer = BytesIO()
stemmer = StempelStemmer.default()
stopwords = (list(adv.stopwords['polish']))
lemmatization_model = spacy.load("pl_core_news_sm")
lemmatizer = lemmatization_model.get_pipe("lemmatizer")
nltk.download('punkt')

def stemming_tokenizer(phrases):
    words = word_tokenize(phrases)
    words = [stemmer.stem(word) if stemmer.stem(word) is not None else word for word in words]
    return words

def lemmatization_tokenizer(phrases):
    words = lemmatization_model(phrases)
    words = [word.lemma_ for word in words]
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

    if clustering_type == "DBSCAN":
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=0.01, stop_words=stopwords,
                                           tokenizer = tokenizer, use_idf=True, ngram_range=(1, 2))
        tfidf = tfidf_vectorizer.fit_transform(keywords)
        cluster = DBSCAN(eps = sensivity, min_samples = min_cluster).fit(tfidf).labels_.tolist()
        results = pd.DataFrame(sorted(zip(cluster, keywords)), columns=["cluster_id", "keyword"])

    elif clustering_type == "GAACluster":
        clusterer = GAAClusterer(nr_clusters)
        vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
        bow = vectorizer.fit_transform(keywords)
        clusters = clusterer.cluster(bow)
        results = pd.DataFrame(sorted(zip(clusters, keywords)), columns=["cluster_id", "keyword"])
    else:
        if clustering_type == "k-means Tfidf":
            vectorizer = TfidfVectorizer(max_df=0.3, max_features=10000, min_df=0.008, stop_words=stopwords,
                                               tokenizer = tokenizer, use_idf=True, ngram_range=(1, 2))
            vector = vectorizer.fit_transform(keywords)
            vector = scipy.sparse.csr_matrix.toarray(vector)
        else:
            vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
            vector = vectorizer.fit_transform(keywords)
        clusterer = KMeansClusterer(num_means=nr_clusters, distance=euclidean_distance, repeats=20)
        clusters = clusterer.cluster(vector)
        # if distance_type == "euclidean":
        #     clusterer = KMeansClusterer(num_means=nr_clusters, distance=euclidean_distance, repeats=20)
        #     clusters = clusterer.cluster(vector)
        # elif distance_type == "cosine":
        #     clusters = KMeansClusterer(num_means=nr_clusters,distance=cosine_distance, repeats=20)
        # else:
        #     clusters = KMeansClusterer(num_means=nr_clusters,distance=edit_distance, repeats=20).classify(vector)
        results = pd.DataFrame(sorted(zip(clusters, keywords)), columns=["cluster_id", "keyword"])

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
            cluster_name_list.append("Grupa {}, #{} Elementów ".format(nr + 1, len(cluster)))
            corpus_sentences_list.append(keywords[id])

    results = pd.DataFrame(sorted(zip(cluster_name_list,cluster_name_list, corpus_sentences_list)), columns=["cluster_id", "cluster_name", "keyword"])

    results["length"] = results["keyword"].astype(str).map(len)
    results = results .sort_values(by="length", ascending=True)

    results['cluster_name'] = results.groupby('cluster_name')['keyword'].transform('first')
    results.sort_values(['cluster_name', "keyword"], ascending=[True, True], inplace=True)

    results['cluster_name'] = results['cluster_name'].fillna("ups! nie przypisano do żadnego klastra")

    del results['length']
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
            cluster_name_list.append("Grupa {}, #{} Elementów ".format(nr + 1, len(cluster)))
            corpus_sentences_list.append(kw)

    results = pd.DataFrame(sorted(zip(cluster_name_list,cluster_name_list, corpus_sentences_list)), columns=["cluster_id", "cluster_name", "keyword"])

    results["length"] = results["keyword"].astype(str).map(len)
    results = results .sort_values(by="length", ascending=True)

    results['cluster_name'] = results.groupby('cluster_name')['keyword'].transform('first')
    results.sort_values(['cluster_name', "keyword"], ascending=[True, True], inplace=True)

    results['cluster_name'] = results['cluster_name'].fillna("ups! nie przypisano do żadnego klastra")

    del results['length']
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
            cluster_name_list.append("Grupa {}, #{} Elementów ".format(nr + 1, len(cluster)))
            corpus_sentences_list.append(kw)


    results = pd.DataFrame(sorted(zip(cluster_name_list,cluster_name_list, corpus_sentences_list)), columns=["cluster_id", "cluster_name", "keyword"])

    results["length"] = results["keyword"].astype(str).map(len)
    results = results .sort_values(by="length", ascending=True)

    results['cluster_name'] = results.groupby('cluster_name')['keyword'].transform('first')
    results.sort_values(['cluster_name', "keyword"], ascending=[True, True], inplace=True)

    results['cluster_name'] = results['cluster_name'].fillna("ups! nie przypisano do żadnego klastra")

    del results['length']
    excel_output(results)
    st.table(results)
    return 0

def clustering_semantic_dbscan(keywords, transformer = 'sdadas/st-polish-paraphrase-from-distilroberta', min_cluster=2, distance_type='euclidean'):
    corpus_sentences_list =[]
    cluster_name_list = []
    model = SentenceTransformer(transformer)
    corpus_embeddings = model.encode(keywords, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    clusterer = DBSCAN(eps = 5.5, min_samples = min_cluster, metric = distance_type)
    clusterer.fit(corpus_embeddings)

    clusters = clusterer.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(keywords[sentence_id])

    for nr, cluster in clustered_sentences.items():
        for kw in cluster[:]:
            cluster_name_list.append("Grupa {}, #{} Elementów".format(nr + 1, len(cluster)))
            corpus_sentences_list.append(kw)

    results = pd.DataFrame(sorted(zip(cluster_name_list,cluster_name_list, corpus_sentences_list)), columns=["cluster_id", "cluster_name", "keyword"])

    results["length"] = results["keyword"].astype(str).map(len)
    results = results .sort_values(by="length", ascending=True)

    results['cluster_name'] = results.groupby('cluster_name')['keyword'].transform('first')
    results.sort_values(['cluster_name', "keyword"], ascending=[True, True], inplace=True)

    results['cluster_name'] = results['cluster_name'].fillna("ups! nie przypisano do żadnego klastra")

    del results['length']
    excel_output(results)
    st.table(results)
    return 0
