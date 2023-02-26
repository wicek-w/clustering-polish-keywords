# Klasteryzacja słów kluczowych w języku polskim
Projekt wykonany w ramach pracy inżynierskiej "Klasteryzacja  fraz w języku polskim - implementacja oraz analiza porównawcza wybranych metod" na Uniwersytecie Łódzkim.

Dostępna aplikacja https://clustering-polish-keywords.streamlit.app/ pozwalająca na klasteryzację fraz w języku polskim na bazie morfologii słów lub powiązań sematycznych dla wybranych metod.

# Clustering Keywords in Polish

This repository contains a Streamlit application for clustering keywords in Polish. Users can choose from various clustering methods based on keyword morphology or semantics.

## Methods Based on Keyword Morphology

### KMeans with Bag-of-Words Vectorizing

This method clusters keywords using the KMeans algorithm and Bag-of-Words vectorization technique. Users can choose from Euclidean distance, Edit distance, or Cosine similarity as the distance metric.

### KMeans with TFIDF Vectorizing

This method clusters keywords using the TFIDF vectorization technique. Users can choose from Euclidean distance, Edit distance, or Cosine similarity as the distance metric.

### DBSCAN

This method clusters keywords using the DBSCAN algorithm.

### GAACluster

This method clusters keywords using the GAACluster algorithm.

## Methods Based on Semantics

Users can also choose clustering based on semantics using the following methods:

### KMeans

This method clusters keywords using the KMeans algorithm and semantic analysis.

### DBSCAN

This method clusters keywords using the DBSCAN algorithm and semantic analysis.

### Hierarchical Agglomerative

This method clusters keywords using the Hierarchical Agglomerative algorithm and semantic analysis.

To use the application, simply run the Streamlit app (https://clustering-polish-keywords.streamlit.app/) and select the desired clustering method.
