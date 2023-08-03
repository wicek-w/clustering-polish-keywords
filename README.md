# Klasteryzacja słów kluczowych w języku polskim
Projekt wykonany w ramach pracy inżynierskiej "Klasteryzacja  fraz w języku polskim - implementacja oraz analiza porównawcza wybranych metod" na Uniwersytecie Łódzkim.

Dostępna aplikacja https://clustering-polish-keywords.streamlit.app/ pozwalająca na klasteryzację fraz w języku polskim na bazie morfologii słów lub powiązań sematycznych dla wybranych metod.

# Clustering Keywords in Polish

This repository contains a Streamlit application for clustering keywords in Polish. The application allows users to cluster keywords based on morphology or semantics, using various clustering algorithms and distance metrics.

## Methods Based on Keyword Morphology

User can choose normalization of keywords - stemming or lemmatization. The following methods are available for clustering based on keyword morphology:

### K-means with Bag-of-Words or TFIDF vectorization

This method clusters keywords using the KMeans algorithm and Bag-of-Words/TFIDF vectorization technique. Users can choose from Euclidean distance, Edit distance, or Cosine similarity as the distance metric.

The KMeans algorithm is a widely used clustering algorithm that partitions data into clusters based on similarity. The Bag-of-Words technique represents text data as a set of words, ignoring grammar and word order. The TFIDF (Term Frequency-Inverse Document Frequency) technique is a popular method for clustering text data based on their similarity. The technique assigns a weight to each word in the text, based on its frequency in the document and its frequency in the corpus. 

### DBSCAN

This method clusters keywords using the DBSCAN algorithm.

The DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm is a density-based clustering algorithm that groups together data points that are close to each other in space. This method is useful for clustering keywords that are closely related.

### Agglomerative Hierarchical

This method clusters keywords using the Hierarchical algorithm to create number of clusters given by user.

## Methods Based on Semantics
Semantic analysis is a technique that assigns a meaning to each word in the text, based on its context. This method is useful for clustering keywords based on their meaning. Users can  choose clustering based on semantics using the following methods:

### KMeans

This method clusters keywords using the KMeans algorithm and semantic analysis.

### DBSCAN

This method clusters keywords using the DBSCAN algorithm and semantic analysis.

### Agglomerative Hierarchical 

This method clusters keywords using the Hierarchical Agglomerative algorithm and semantic analysis.

To use the application, simply run the Streamlit app (https://clustering-polish-keywords.streamlit.app/) and select the desired clustering method.
