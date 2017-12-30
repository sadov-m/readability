import pandas as pd
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

vizualize = False
labels = joblib.load('labels_for_tagged_55')
print(Counter(labels))

data = pd.read_csv('resulting_dfs\\for_tagged_december\\result_begtin.csv')

text_name = []
for elem in data['filename']:
    text_name.append(int(elem.split('\\')[-1].split('.')[0]))

new_labels = [labels[i] for i in text_name]

"""data['final_score'] = data[[' Flesch-Kincaid', ' Coleman-Liau index',
                                                  ' Dale-Chale readability formula', ' readability level (SMOG)',
                                                  ' Automated Readability Index']].mean(axis=1)"""

data = data.drop(labels=['filename'], axis=1)

clusterize = KMeans(n_clusters=4, random_state=42)
output = clusterize.fit_predict(data)

data_res = []

data_res.append(({
        'ARI': metrics.adjusted_rand_score(new_labels, clusterize.labels_),
        'AMI': metrics.adjusted_mutual_info_score(new_labels, clusterize.labels_),
        'Homogenity': metrics.homogeneity_score(new_labels, clusterize.labels_),
        'Completeness': metrics.completeness_score(new_labels, clusterize.labels_),
        'V-measure': metrics.v_measure_score(new_labels, clusterize.labels_),
        'Silhouette': metrics.silhouette_score(data, clusterize.labels_)}))

results = pd.DataFrame(data=data_res, columns=['ARI', 'AMI', 'Homogenity',
                                           'Completeness', 'V-measure',
                                           'Silhouette'], index=['K-means'])

print(results)

if vizualize:
    pca = PCA(n_components=2)
    X_top_plot = pca.fit_transform(data)
    plt.figure()
    clusters = new_labels
    plt.scatter(X_top_plot[:, 0], X_top_plot[:, 1], c=new_labels)  # plot all points
    plt.show()
else:
    data['filename'] = [str(elem) + '.txt' for elem in text_name]
    data['cluster'] = clusterize.labels_
    data.to_csv('resulting_dfs\\for_tagged_december\\result_begtin_for_analysis.csv')
