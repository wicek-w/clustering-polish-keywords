import math
import streamlit as st
import pandas as pd
import functions as f
import io

buffer = io.BytesIO()
header = st.container()
dataset = st.container()
model = st.container()
finish = st.container()

with header:
    st.title("Klasteryzacja fraz")
    st.text("W tej aplikacji możesz sklasyfikować podane frazy za pomocą różnych algorytmów")

with dataset:
    st.header("Podaj dane:")
    # input a keywords
    kw_input = pd.DataFrame(st.text_area("Podaj listę fraz, które chcesz skatogeryzować (każda fraza powinna być w osobnej linijce:").split('\n'))
    kw_input = kw_input[0].tolist()
    st.write(kw_input)
    clustering_type = st.selectbox("Choose a type of the algorithm to cluster your keywords", options=["-","k-means Tfidf","DBSCAN", "GAACluster", "k-means NLTK", "graph-based"], index = 0)

with model:
    nlp_type = st.selectbox("Na czym ma być oparta klasteryzacja?", options=["budowa słów", "semantyka"], index=0)
    if nlp_type == "semantyka":
        '======'
    else:
        distance = {'euklidesowa': 'euclidean',
                'cosinusowa': 'cosine',
                'cityblock': 'cityblock',
                'manhattan': 'manhattan',
                'levenshtein': 'lev_distance'}
        normalization_type = st.selectbox("Jaką normalizację zastosować", options=["stemming", "lematyzacje"], index=0)

        if clustering_type == "k-means NLTK":
            distance_type = st.selectbox("Jaką odległość chcesz wykorzystać", options=["euklidesowa", "cosinusowa","levenshtein"], index=0)
            results = f.cluster_morphology(keywords=kw_input, clustering_type=clustering_type, distance_type=distance[distance_type], normalization_type=normalization_type)
        elif clustering_type == "DBSCAN":
            distance_type = st.selectbox("Jaką odległość chcesz wykorzystać", options=["-","euklidesowa", "cosinusowa","manhattan", "cityblock"], index=1)
            sensivity = st.number_input(label='Podaj epsilon: ', min_value=0.2, key=1)
            min_cluster = st.number_input(label='Podaj minimalną liczbę klastrów (większą niż 1): ', min_value=2, key=2)
            results = f.cluster_morphology(keywords=kw_input, clustering_type=clustering_type, min_cluster=min_cluster, sensivity=sensivity, distance_type=distance[distance_type], normalization_type=normalization_type)
        elif clustering_type in ["k-means Tfidf", "GAACluster"]:
            nr_clusters = st.number_input(label='Podaj liczbę klastrów (domyślnie pierwiastek z liczby słów - jeśli chcesz by tak pozostało wybierz 1): ', min_value=1, key=2)
            nr_clusters = int(math.sqrt(len(kw_input))) if nr_clusters == 1 else nr_clusters
            results = f.cluster_morphology(keywords=kw_input, clustering_type=clustering_type, nr_clusters=nr_clusters, normalization_type=normalization_type)
