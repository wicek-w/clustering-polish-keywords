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
    st.title("Klasteryzacja fraz w języku polskim")
    st.text("W tej aplikacji możesz pogrupować podane frazy na różny sposób")

with dataset:
    st.header("Podaj dane:")
    # input a keywords
    kw_input = pd.DataFrame(st.text_area("Podaj listę fraz, które chcesz pogrupować (każda fraza powinna być w osobnej linijce). Powinno być ich maksymalnie 10 tysięcy:").split('\n'))
    kw_input = kw_input[0].tolist()
    st.write(kw_input)
    if len(kw_input)>0:
        nlp_type = st.selectbox("Na czym ma być oparta klasteryzacja?", options=["-","budowa słów", "semantyka (kontekst)"], index=0)

with model:
    if len(kw_input)>0:
        if nlp_type == "semantyka (kontekst)":
            st.write("To może chwile potrwać, ale uwierz warto!")
            clustering_type = st.selectbox("Wybierz metodę klasteryzacji: ",
                                           options=["-",
                                                    "aglomeracyjna dla dużych liczby fraz (od kilku do 50 tysięcy) - uwaga, ten algorytm działa trochę dłużej",
                                                    "aglomeracyjna",
                                                    "k-means",
                                                    "DBSCAN"],
                                           index = 0)
            model_type = st.selectbox("Wybierz model na podstawie, którego powinny być sklasteryzowane Twoje frazy: ",
                                      options=["sdadas/st-polish-paraphrase-from-distilroberta",
                                               "sentence-transformers/stsb-roberta-base-v2", "Voicelab/sbert-large-cased-pl"],
                                      index = 0)
            if clustering_type == 'aglomeracyjna':
                if st.button('Zaczynajmy!'):
                    f.clustering_semantic_agglomerative(kw_input,model_type)
            elif clustering_type == 'k-means':
                nr_clusters = st.number_input(label='Podaj liczbę klastrów (domyślnie pierwiastek z liczby słów - jeśli chcesz by tak pozostało wybierz 1): ', min_value=1, key=2)
                nr_clusters = int(math.sqrt(len(kw_input))) if nr_clusters == 1 else nr_clusters
                if st.button('Zaczynajmy!'):
                    f.clustering_semantic_kmeans(kw_input, model_type, nr_clusters)
            elif clustering_type == "DBSCAN":
                min_cluster = st.number_input(label='Podaj minimalna liczbę obserwacji w grupie:', min_value=2)
                if st.button('Zaczynajmy!'):
                    f.clustering_semantic_dbscan(kw_input, model_type, min_cluster= min_cluster)
            else:
                accuracy = st.slider("Jak dokładny powino być grupowanie? (Im bliżej 100 tym bardziej restrykcyjne grupowanie): ", min_value=0, max_value=100)
                min_cluster =st.number_input(label='Minimalna liczba fraz w grupie: ', min_value=2)
                if st.button('Zaczynajmy!'):
                    results = f.clustering_semantic_fast(kw_input, accuracy, min_cluster, model_type)
        else:
            clustering_type = st.selectbox("Wybierz metodę klasteryzacji: ", options=["-", "k-means Tfidf", "DBSCAN", "aglomeracyjna", "k-means bag-of-words"], index = 0)
            distance = {'euklidesowa': 'euclidean',
                'cosinusowa': 'cosine',
                'levenshtein': 'edit_distance'}
            normalization_type = st.selectbox("Jaką normalizację zastosować",
                                              options=["stemming", "lematyzacje"],
                                              index=0)

            if clustering_type in ["k-means Tfidf", "k-means bag-of-words"]:
                distance_type = st.selectbox("Jaką odległość chcesz wykorzystać",
                                             options=["euklidesowa",
                                                      "cosinusowa",
                                                      "levenshtein"],
                                             index=0)

                nr_clusters = st.number_input(label='Podaj liczbę klastrów (domyślnie pierwiastek z liczby słów - jeśli chcesz by tak pozostało wybierz 1): ', min_value=1, key=2)
                nr_clusters = int(math.sqrt(len(kw_input))) if nr_clusters == 1 else nr_clusters
                if st.button('Zaczynajmy!'):
                    results = f.cluster_morphology(keywords=kw_input, clustering_type=clustering_type, nr_clusters=nr_clusters, distance_type=distance[distance_type], normalization_type=normalization_type)
            elif clustering_type == "DBSCAN":
                min_cluster = st.number_input(label='Podaj minimalna liczbę obserwacji w grupie:', min_value= 2)
                if st.button('Zaczynajmy!'):
                    f.cluster_morphology(keywords=kw_input, clustering_type=clustering_type, min_cluster= min_cluster, normalization_type=normalization_type)
            elif clustering_type in ["aglomeracyjna"]:
                nr_clusters = st.number_input(label='Podaj liczbę klastrów (domyślnie pierwiastek z liczby słów - jeśli chcesz by tak pozostało wybierz 1): ', min_value=1, key=2)
                nr_clusters = int(math.sqrt(len(kw_input))) if nr_clusters == 1 else nr_clusters
                if st.button('Zaczynajmy!'):
                    f.cluster_morphology(keywords=kw_input, clustering_type=clustering_type, nr_clusters=nr_clusters, normalization_type=normalization_type)
