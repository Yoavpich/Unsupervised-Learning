import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ClusteringMethods


# This function made to duplicate the Dataset into an dataframe
def get_data2():
    # To use only the first N rows, use 'nrows=N' in read_csv
    df = pandas.read_csv("DataSet2 - Diabetes.csv", sep=",", nrows=10000)
    columns_to_convert = ['race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2',
                          'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
                          'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                          'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                          'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                          'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                          'diabetesMed', 'readmitted']
    # Convert the above columns to int values
    convert(df, columns_to_convert)
    # We will ignore the class ('race', 'gender')
    to_tag = ['race', 'gender']
    cluster = df.drop(to_tag, axis=1)
    tag = df[to_tag]
    return cluster, tag


def convert(dataframe, columns_to_convert):
    for c_name in columns_to_convert:
        column = dataframe[c_name].tolist()
        dic = {}
        counter = 0
        for i in range(len(column)):
            if column[i] not in dic.keys():
                dic[column[i]] = counter
                counter += 1
            column[i] = dic[column[i]]
        newc = pandas.Series(column, name=c_name)
        dataframe.update(newc)
        # print(c_name + " number of options = " + str(len(dic.keys())))


# --------------------------- Handle The Data --------------------------- #
# ------------------------------ Get Data ------------------------------ #
dataset2 = get_data2()[0]
# print(dataset2)

# ------------------------------ Get Tags ------------------------------ #
tags = get_data2()[1].values.tolist()
# Normalize the tags
tags = StandardScaler().fit_transform(tags)
# Use PCA to reduce the tags in to 1D
pca = PCA(n_components=1)
tags_1D = pca.fit_transform(tags)
# Turn it into a list
tags_1D = [item for sublist in tags_1D for item in sublist]
# print(tags_1D)


# ------------------------------ Normalize ------------------------------ #
dataset2 = StandardScaler().fit_transform(dataset2)

# --------------------------------- PCA --------------------------------- #
# Now the dataset contain only numbers in R^2, and we can use PCA
pca = PCA(n_components=2)
dataset2_2D = pca.fit_transform(dataset2)

# --------------------------------- Plot --------------------------------- #
ClusteringMethods.plot_data(dataset2_2D)


# ------------------------------ Algorithms ------------------------------ #
ClusteringMethods.elbow_method(dataset2_2D)
print(ClusteringMethods.correct_clusters_choice(dataset2_2D))  # We got 3
# k=3 as we infer from the Elbow Method and the Silhouette Score
number_of_clusters = 3


# ------------------------------- K-Means ------------------------------- #
mi_kmeans = ClusteringMethods.k_means_clustering(dataset2_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of K-Means: " + str(mi_kmeans))

# ---------------------------- Fuzzy C Means ---------------------------- #
mi_fcm = ClusteringMethods.fuzzy_cmeans_clustering(dataset2_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Fuzzy C Means: " + str(mi_fcm))

# --------------------------------- GMM --------------------------------- #
mi_gmm = ClusteringMethods.gmm_clustering(dataset2_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of GMM: " + str(mi_gmm))

# ----------------------- Hierarchical Clustering ----------------------- #
# ClusteringMethods.hierarchical_clustering(dataset2_2D)
mi_hc = ClusteringMethods.agglomerative_clustering(dataset2_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Hierarchical Clustering: " + str(mi_hc))

# ------------------------- Spectral Clustering ------------------------- #
mi_sc = ClusteringMethods.spectral_clustering(dataset2_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Spectral Clustering: " + str(mi_sc))


# ------------------------------- T - Test ------------------------------- #
# We will take a few samples (13 Samples) of the MI from each of the clustering
# methods and than use their mean for the T-test
arg = ClusteringMethods.get_mi_matrix(dataset2_2D, tags_1D, number_of_clusters, 13)
best = ClusteringMethods.find_best_algorithm(arg[0], arg[1], 2)
print(best)
