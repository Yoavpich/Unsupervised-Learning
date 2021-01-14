import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ClusteringMethods


# This function made to duplicate the Dataset into an dataframe
def get_data3():
    # To use only the first N rows, use 'nrows=N' in read_csv
    df = pandas.read_csv("DataSet3 - e-shop.csv", sep=";",  nrows=60000)
    columns = ['year', 'month', 'day', 'order', 'country', 'session ID', 'page 1 (main category)',
               'page 2 (clothing model)', 'colour', 'location',
               'model photography', 'price', 'price 2', 'page']
    # The first column ('year') is the same for all the data so we will ignore it
    # Also we will ignore the class ('country')
    to_drop = ['year', 'country']
    cluster = df.drop(to_drop, axis=1)
    to_tag = ['country']
    tag = df[to_tag]
    return cluster, tag


# Helper for sort_column8
def get_letter(e):
    return e[0]


# Helper for sort_column8
def get_number(e):
    return int(e[1:])


# Special sort for column 8 of the Dataset
def sort_column8(c):
    # First we will sort by the numbers
    sorted_c = sorted(c, key=get_number)
    # Than we will sort by the letters
    sorted_c = sorted(sorted_c, key=get_letter)
    return sorted_c


# --------------------------- Handle The Data --------------------------- #
# ------------------------------ Get Data ------------------------------ #
dataset3 = get_data3()[0]
# print(dataset3)


# ------------------------------ Get Tags ------------------------------ #
tags = get_data3()[1].values.tolist()
tags_1D = [item for sublist in tags for item in sublist]
# print(tags)


# ------------------------ Handle The 8th Column ------------------------ #
# Get the 8th column
data = dataset3.values.tolist()
column8 = []
for row in data:
    # Because we start the indexes from 0 and we deleted the first column and the class
    # So the index will be 5
    column8.append(row[5])
column8_copy = column8
# Remove duplications
column8 = list(dict.fromkeys(column8))
# Sort column8
column8 = sort_column8(column8)
# Create number values for the 8th column by using dictionary
column8_dic = {}
for i in range(len(column8)):
    column8_dic.update({column8[i]: i + 1})


# ----------------- Turn The 8th Column Into R^1 Values ----------------- #
new_column8 = []
for k in column8_copy:
    new_column8.append(column8_dic[k])
updated_column8 = pandas.Series(new_column8, name='page 2 (clothing model)')
dataset3.update(updated_column8)


# ------------------------------ Normalize ------------------------------ #
dataset3 = StandardScaler().fit_transform(dataset3)


# --------------------------------- PCA --------------------------------- #
# Now the dataset contain only numbers in R^2, and we can use PCA
pca = PCA(n_components=2)
dataset3_2D = pca.fit_transform(dataset3)


# --------------------------------- Plot --------------------------------- #
ClusteringMethods.plot_data(dataset3_2D)


# ------------------------------ Algorithms ------------------------------ #
ClusteringMethods.elbow_method(dataset3_2D)
print(ClusteringMethods.correct_clusters_choice(dataset3_2D))  # We got 2
# k=2 as we infer from the Elbow Method and the Silhouette Score
number_of_clusters = 2


# ------------------------------- K-Means ------------------------------- #
mi_kmeans = ClusteringMethods.k_means_clustering(dataset3_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of K-Means: " + str(mi_kmeans))

# ---------------------------- Fuzzy C Means ---------------------------- #
mi_fcm = ClusteringMethods.fuzzy_cmeans_clustering(dataset3_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Fuzzy C Means: " + str(mi_fcm))

# --------------------------------- GMM --------------------------------- #
mi_gmm = ClusteringMethods.gmm_clustering(dataset3_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of GMM: " + str(mi_gmm))

# ----------------------- Hierarchical Clustering ----------------------- #
# ClusteringMethods.hierarchical_clustering(dataset3_2D)
mi_hc = ClusteringMethods.agglomerative_clustering(dataset3_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Hierarchical Clustering: " + str(mi_hc))

# ------------------------- Spectral Clustering ------------------------- #
mi_sc = ClusteringMethods.spectral_clustering(dataset3_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Spectral Clustering: " + str(mi_sc))


# ------------------------------- T - Test ------------------------------- #
# We will take a few samples (13 Samples) of the MI from each of the clustering
# methods and than use their mean for the T-test
arg = ClusteringMethods.get_mi_matrix(dataset3_2D, tags_1D, number_of_clusters, 13)
best = ClusteringMethods.find_best_algorithm(arg[0], arg[1], 3)
print(best)
