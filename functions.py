import pandas as pd
import numpy as np
from nltk.cluster import GAAClusterer
from nltk import word_tokenize
import advertools as adv
from stempel import StempelStemmer
# from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.metrics.distance import edit_distance
import streamlit as st
from io import BytesIO
import spacy
from sentence_transformers import SentenceTransformer, util
import xlsxwriter
import nltk
from sklearn.preprocessing import normalize

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

def similarity(phrases):
    dist = np.array([[edit_distance(list(w1),list(w2)) for w1 in phrases] for w2 in phrases])
    dist = -1*dist
    return dist

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
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=20000, min_df=0.01, stop_words=stopwords,
                                           tokenizer = tokenizer, use_idf=True, ngram_range=(1, 2))
        tfidf = tfidf_vectorizer.fit_transform(keywords)
        clusters = DBSCAN(eps = sensivity, min_samples = min_cluster).fit(tfidf).labels_.tolist()
        # silhouette_avg = silhouette_score(tfidf, clusters)
        results = pd.DataFrame(sorted(zip(clusters, clusters, keywords)), columns=["cluster_id", "cluster_name", "keyword"])

    elif clustering_type == "aglomeracyjna":
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=20000, min_df=0.01, stop_words=stopwords,
                                           tokenizer = tokenizer, use_idf=True, ngram_range=(1, 2))
        vector = tfidf_vectorizer.fit_transform(keywords)
        clusterer = AgglomerativeClustering(linkage='complete', n_clusters=nr_clusters)
        clusters = clusterer.fit_predict(vector.toarray())
        # silhouette_avg = silhouette_score(vector, clusters)
        results = pd.DataFrame(sorted(zip(clusters, clusters, keywords)), columns=["cluster_id", "cluster_name", "keyword"])

    else:
        if clustering_type == "k-means Tfidf":
            vectorizer = TfidfVectorizer(max_df=0.3, max_features=20000, min_df=0.008, stop_words=stopwords,
                                             tokenizer = tokenizer, use_idf=True, ngram_range=(1, 2))
            vector = vectorizer.fit_transform(keywords)
        else:
            vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
            vector = vectorizer.fit_transform(keywords)
        if distance_type == "euclidean":
            clusters = KMeans(n_clusters=nr_clusters, random_state=20).fit_predict(vector)
            results = pd.DataFrame(sorted(zip(clusters, clusters, keywords)), columns=["cluster_id", "cluster_name", "keyword"])
            # silhouette_avg = silhouette_score(vector, clusters)
        elif distance_type == "cosine":
            similarity_matrix = cosine_similarity(vector)
            clusters = KMeans(n_clusters=nr_clusters, random_state=20).fit_predict(similarity_matrix)
            results = pd.DataFrame(sorted(zip(clusters, clusters, keywords)), columns=["cluster_id", "cluster_name", "keyword"])
            # silhouette_avg = silhouette_score(similarity_matrix, clusters)
        else:
            corpus_sentences_list =[]
            cluster_name_list = []
            distance = similarity(keywords)
            clusterer = KMeans(n_clusters=nr_clusters)
            clusterer.fit(distance)
            cluster_assignment = clusterer.labels_
            # silhouette_avg = silhouette_score(distance, cluster_assignment)
            clustered_sentences = [[] for i in range(nr_clusters)]
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
    # st.write(silhouette_avg)
    results['cluster_name'] = results['cluster_name'].fillna("ups! nie przypisano do żadnego klastra")
    del results['length']
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

    # silhouette_avg = silhouette_score(corpus_embeddings, cluster_name_list)
    # st.write(silhouette_avg)

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
    corpus_embeddings = model.encode(keywords, batch_size=256)
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

    # silhouette_avg = silhouette_score(corpus_embeddings, cluster_assignment)
    # st.write(silhouette_avg)

    del results['length']
    excel_output(results)
    st.table(results)
    return 0

def clustering_semantic_agglomerative(keywords, transformer = 'sdadas/st-polish-paraphrase-from-distilroberta'):
    corpus_sentences_list =[]
    cluster_name_list = []
    model = SentenceTransformer(transformer)
    corpus_embeddings = model.encode(keywords, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=6.5)
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

    # silhouette_avg = silhouette_score(corpus_embeddings, clusters)
    # st.write(silhouette_avg)

    results["length"] = results["keyword"].astype(str).map(len)
    results = results .sort_values(by="length", ascending=True)

    results['cluster_name'] = results.groupby('cluster_name')['keyword'].transform('first')
    results.sort_values(['cluster_name', "keyword"], ascending=[True, True], inplace=True)

    results['cluster_name'] = results['cluster_name'].fillna("ups! nie przypisano do żadnego klastra")

    del results['length']
    excel_output(results)
    st.table(results)
    return 0

def clustering_semantic_dbscan(keywords, transformer = 'sdadas/st-polish-paraphrase-from-distilroberta', min_cluster=2):
    corpus_sentences_list =[]
    cluster_name_list = []
    model = SentenceTransformer(transformer)
    corpus_embeddings = model.encode(keywords, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    clusterer = DBSCAN(eps = 5.5, min_samples = min_cluster)
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

    # silhouette_avg = silhouette_score(corpus_embeddings, clusters)
    # st.write(silhouette_avg)

    del results['length']
    excel_output(results)
    st.table(results)
    return 0
