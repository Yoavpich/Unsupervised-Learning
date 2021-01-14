import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ClusteringMethods


# This function made to duplicate the Dataset into an dataframe
def get_data1():
    # To use only the first N rows, use 'nrows=N' in read_csv
    df = pandas.read_csv("DataSet1 - Online Shoppers Intention.csv", sep=",")
    columns = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
               'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
               'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
               'VisitorType', 'Weekend', 'Revenue']
    # We will ignore the class ('VisitorType', 'Weekend', 'Revenue')
    to_tag = ['VisitorType', 'Weekend', 'Revenue']
    cluster = df.drop(to_tag, axis=1)
    tag = df[to_tag]
    return cluster, tag


# Return 1 for 'TRUE' and 0 for 'FALSE'
def bool_str_to_int(value: bool) -> int:
    if value:
        return 1
    return 0


# Return the number of the month
def month_to_num(month: str) -> int:
    if month == 'Jan':
        return 1
    elif month == 'Feb':
        return 2
    elif month == 'Mar':
        return 3
    elif month == 'Apr':
        return 4
    elif month == 'May':
        return 5
    elif month == 'June':
        return 6
    elif month == 'Jul':
        return 7
    elif month == 'Aug':
        return 8
    elif month == 'Sep':
        return 9
    elif month == 'Oct':
        return 10
    elif month == 'Nov':
        return 11
    elif month == 'Dec':
        return 12


def visitor_type_to_num(visitor: str) -> int:
    if visitor == 'New_Visitor':
        return 0
    elif visitor == 'Returning_Visitor':
        return 1
    elif visitor == 'Other':
        return 2


# --------------------------- Handle The Data --------------------------- #
# ------------------------------ Get Data ------------------------------ #
dataset1 = get_data1()[0]
# print(dataset1)


# ------------------------------ Get Tags ------------------------------ #
tags = get_data1()[1]
new_column0 = []
new_column1 = []
new_column2 = []
for row in tags.values.tolist():
    new_column0.append(visitor_type_to_num(row[0]))
    new_column1.append(bool_str_to_int(row[1]))
    new_column2.append(bool_str_to_int(row[2]))
tags.update(pandas.Series(new_column0, name='VisitorType'))
tags.update(pandas.Series(new_column1, name='Weekend'))
tags.update(pandas.Series(new_column2, name='Revenue'))
tags = tags.values.tolist()
# Normalize the tags
tags = StandardScaler().fit_transform(tags)
# Use PCA to reduce the tags in to 1D
pca = PCA(n_components=1)
tags_1D = pca.fit_transform(tags)
# Turn it into a list
tags_1D = [item for sublist in tags_1D for item in sublist]
# print(tags_1D)

# ------------------------ Handle The 11th Column ------------------------ #
# Get the 11th column - month (Index 10)
new_column11 = []
for row in dataset1.values.tolist():
    # We start the indexes from 0 so the index will be 10
    new_column11.append(month_to_num(row[10]))
dataset1.update(pandas.Series(new_column11, name='Month'))

# ------------------------------ Normalize ------------------------------ #
dataset1 = StandardScaler().fit_transform(dataset1)

# --------------------------------- PCA --------------------------------- #
# Now the dataset contain only numbers in R^2, and we can use PCA
pca = PCA(n_components=2)
dataset1_2D = pca.fit_transform(dataset1)

# --------------------------------- Plot --------------------------------- #
ClusteringMethods.plot_data(dataset1_2D)


# ------------------------------ Algorithms ------------------------------ #
ClusteringMethods.elbow_method(dataset1_2D)
print(ClusteringMethods.correct_clusters_choice(dataset1_2D))  # We got 3
# k=3 as we infer from the Elbow Method and the Silhouette Score
number_of_clusters = 3


# ------------------------------- K-Means ------------------------------- #
mi_kmeans = ClusteringMethods.k_means_clustering(dataset1_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of K-Means: " + str(mi_kmeans))

# ---------------------------- Fuzzy C Means ---------------------------- #
mi_fcm = ClusteringMethods.fuzzy_cmeans_clustering(dataset1_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Fuzzy C Means: " + str(mi_fcm))

# --------------------------------- GMM --------------------------------- #
mi_gmm = ClusteringMethods.gmm_clustering(dataset1_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of GMM: " + str(mi_gmm))

# ----------------------- Hierarchical Clustering ----------------------- #
# ClusteringMethods.hierarchical_clustering(dataset1_2D)
mi_hc = ClusteringMethods.agglomerative_clustering(dataset1_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Hierarchical Clustering: " + str(mi_hc))

# ------------------------- Spectral Clustering ------------------------- #
mi_sc = ClusteringMethods.spectral_clustering(dataset1_2D, tags_1D, number_of_clusters)
# Compute how well the clustering method fits the external classification
print("MI of Spectral Clustering: " + str(mi_sc))


# ------------------------------- T - Test ------------------------------- #
# We will take a few samples (17 Samples) of the MI from each of the clustering
# methods and than use their mean for the T-test
arg = ClusteringMethods.get_mi_matrix(dataset1_2D, tags_1D, number_of_clusters, 17)
best = ClusteringMethods.find_best_algorithm(arg[0], arg[1], 1)
print(best)
