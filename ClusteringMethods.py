import pandas
import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind


# Dataset should be in R^2 (After PCA)
def plot_data(dataset):
    principalDf = pandas.DataFrame(data=dataset, columns=['dem1', 'dem2'])
    plt.scatter(principalDf['dem1'], principalDf['dem2'], s=7)
    plt.show()


# Search for the correct number of clusters by using the 'Elbow Method'
# Dataset should be in R^2 (After PCA)
def elbow_method(dataset):
    distortions = []
    K = range(1, 18)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(dataset)
        distortions.append(kmeanModel.inertia_)
    # Print plot
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# Run the algorithm with the chosen k
# Dataset should be in R^2 (After PCA)
# Return the MI
def k_means_clustering(dataset, tags, k, show_plt=True):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(dataset)
    # Make plot of the result
    if show_plt:
        plt.title('K - Means')
        plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, s=7, cmap='rainbow')
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=15, alpha=0.3)
        plt.show()
    return metrics.adjusted_mutual_info_score(tags, labels)


# Run the algorithm with the chosen k
# Dataset should be in R^2 (After PCA)
# Return the MI
def fuzzy_cmeans_clustering(dataset, tags, k, show_plt=True):
    fcm = FCM(n_clusters=k)
    fcm.fit(dataset)
    labels = fcm.predict(dataset)
    if show_plt:
        plt.title('Fuzzy C Means')
        plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, s=7, cmap='rainbow')
        plt.show()
    return metrics.adjusted_mutual_info_score(tags, labels)


# Search for the correct number of clusters by using the Silhouette Score
# Dataset should be in R^2 (After PCA)
def correct_clusters_choice(dataset, show_plt=True):
    max_score = -1
    best_match = 1
    scores = []
    r = range(2, 15)
    for k in r:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(dataset)
        score = silhouette_score(dataset, labels)
        scores.append(score)
        if score > max_score:
            max_score = score
            best_match = k
    if show_plt:
        plt.plot(r, scores)
        plt.title("Silhouette")
        plt.xlabel("Number Of Clusters")
        plt.ylabel("Silhouette Score")
        plt.show()
    return best_match


# Run the algorithm with the chosen k
# Dataset should be in R^2 (After PCA)
# Return the MI
def gmm_clustering(dataset, tags, k, show_plt=True):
    gmm = GaussianMixture(n_components=k)
    labels = gmm.fit_predict(dataset)
    if show_plt:
        plt.title('Gaussian Mixture Model')
        plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, s=7, cmap='rainbow')
        plt.show()
    return metrics.adjusted_mutual_info_score(tags, labels)


def hierarchical_clustering(dataset):
    dendrogram(linkage(dataset, method="ward"))
    plt.title('Dendrogram')
    plt.show()


# Run the algorithm with the chosen k
# Dataset should be in R^2 (After PCA)
# Return the MI
def agglomerative_clustering(dataset, tags, k, show_plt=True):
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(dataset)
    if show_plt:
        plt.title('Agglomerative Clustering')
        plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='rainbow', s=7)
        plt.show()
    return metrics.adjusted_mutual_info_score(tags, labels)


# Run the algorithm with the chosen k
# Dataset should be in R^2 (After PCA)
# Return the MI
def spectral_clustering(dataset, tags, k, show_plt=True):
    sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=0)
    labels = sc.fit_predict(dataset)
    if show_plt:
        plt.title('Spectral Clustering')
        plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='rainbow', s=7)
        plt.show()
    return metrics.adjusted_mutual_info_score(tags, labels)


# Get the following information (dataset, tags, k) and the asked number of samples
# Return a list contain 5 lists of num_of_samples MI results of clustering_method
# Also return a dictionary that's maps the index with the clustering method
def get_mi_matrix(dataset, tags, k, num_of_samples):
    matrix_keys = {0: 'K-Means', 1: 'Fuzzy C Means', 2: 'Gaussian Mixture Model',
                   3: 'Agglomerative Clustering', 4: 'Spectral Clustering'}
    mi_matrix = [[], [], [], [], []]
    for i in range(num_of_samples):
        mi_matrix[0].append(k_means_clustering(dataset, tags, k, show_plt=False))
        mi_matrix[1].append(fuzzy_cmeans_clustering(dataset, tags, k, show_plt=False))
        mi_matrix[2].append(gmm_clustering(dataset, tags, k, show_plt=False))
        mi_matrix[3].append(agglomerative_clustering(dataset, tags, k, show_plt=False))
        mi_matrix[4].append(spectral_clustering(dataset, tags, k, show_plt=False))
    return mi_matrix, matrix_keys


# Check if the mean of the first algorithm's mi scores is greater than the second one
# Return the P-value
def t_test(mi_1, mi_2):
    _, p_value = ttest_ind(mi_1, mi_2, equal_var=False)
    mean1 = sum(mi_1) / len(mi_1)
    mean2 = sum(mi_2) / len(mi_2)
    if mean2 > mean1:
        return p_value / 2
    return 1 - p_value / 2


# Turn a list into a string
def list_to_str(li):
    s = "["
    for k in li:
        s += str(k) + ", "
    s += "]"
    return s


# Compare between the different clustering method using t_test (above)
# mi_matrix contain the mi scores of the 5 algorithms
# matrix_keys is a dic that's fit between row in the mi_matrix to clustering method's name
# Return the most accurate clustering method name and a .txt file with the information from the tests
def find_best_algorithm(mi_matrix, matrix_keys, current_dataset):
    # Create a text for documentation of the process
    txt = "-+-+-+-+-+-+-+-+-+-+-+-\n" \
          "Find Best Algorithm - Dataset {} \n" \
          "Significance level = 5% (Recommended)\n" \
          "-+-+-+-+-+-+-+-+-+-+-+-\n\n".format(current_dataset)
    i = 1
    best_algorithm = 0
    while i < len(mi_matrix):
        # Define the significance level as 5%
        txt += "Test #" + str(i) + ": " + matrix_keys[best_algorithm] + " & " + matrix_keys[i] + "\n"
        p_value = t_test(mi_matrix[best_algorithm], mi_matrix[i])
        txt += "\tP-value: " + str(p_value) + "\n"
        if p_value <= 0.05:
            txt += "\tFrom the P-value above we can infer that " + matrix_keys[i] + \
                   " is better than " + matrix_keys[best_algorithm] + "\n\n"
            best_algorithm = i
        else:
            # Else its stay the same
            txt += "\tFrom the P-value above we can infer that " + matrix_keys[best_algorithm] + \
                   " is better than " + matrix_keys[i] + "\n\n"
        i += 1
    txt += "For conclusion, the best algorithm for the current data (according to the T-test) is " \
           + matrix_keys[best_algorithm]
    file = open("Statistical Test For Dataset {}.txt".format(current_dataset), 'w')
    file.write(txt)
    file.close()
    return matrix_keys[best_algorithm]
