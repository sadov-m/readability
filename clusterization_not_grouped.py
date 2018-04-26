import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from scipy.spatial.distance import cosine

vizualize = True
data = pd.read_csv('dataset\\all_data\\result.csv')

expert_labels = []
for elem in data['filename']:
    expert_labels.append(re.findall('\d+', elem)[0])

data = data.drop(labels=['filename'], axis=1)

clusterize = KMeans(n_clusters=3, random_state=42)
output = clusterize.fit_predict(data)

data_res = []
#clusterize.labels_ = [str(label + 1) for label in clusterize.labels_]
data_res.append(({
        'ARI': metrics.adjusted_rand_score(expert_labels, clusterize.labels_),
        'AMI': metrics.adjusted_mutual_info_score(expert_labels, clusterize.labels_),
        'Homogenity': metrics.homogeneity_score(expert_labels, clusterize.labels_),
        'Completeness': metrics.completeness_score(expert_labels, clusterize.labels_),
        'V-measure': metrics.v_measure_score(expert_labels, clusterize.labels_),
        'Silhouette': metrics.silhouette_score(data, clusterize.labels_)}))

results = pd.DataFrame(data=data_res, columns=['ARI', 'AMI', 'Homogenity',
                                           'Completeness', 'V-measure',
                                           'Silhouette'], index=['K-means'])

print(results)

if vizualize:
    pca = PCA(n_components=2)
    X_top_plot = pca.fit_transform(data)
    plt.figure()
    clusters = clusterize.labels_
    plt.scatter(X_top_plot[:, 0], X_top_plot[:, 1], c=expert_labels)  # plot all points
    plt.show()

data.to_csv('clusterization_result.csv', sep=',',
            encoding='utf-8')
