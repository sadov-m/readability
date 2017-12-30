import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.externals import joblib

vizualize = False
col_names = ['name', 'first_class', '2nd_class_W', '2nd_class_S', '3rd_class_W', '3rd_class_S', '4th_class_W',
             '4th_class_S', 'first_class_names', '2nd_class_W_names', '2nd_class_S_names', '3rd_class_W_names',
             '3rd_class_S_names', '4th_class_W_names', '4th_class_S_names']
"""first = pd.read_csv(r'dataframes_with_results\\first_df.csv', delimiter=',')
first_df = first.loc[:, ('1st', '2nd', '3rd', '4th', '5th', '6th', '7th')]

second = pd.read_csv(r'dataframes_with_results\\second_df.csv', delimiter=',')
second_df = second.loc[:, ('1st', '2nd', '3rd', '4th', '5th', '6th', '7th')]

third = pd.read_csv(r'dataframes_with_results\\third_df.csv', delimiter=',')
third_df = third.loc[:, ('1st', '2nd', '3rd', '4th', '5th', '6th', '7th')]

fourth = pd.read_csv(r'dataframes_with_results\\fourth_df.csv', delimiter=',')
fourth_df = fourth.loc[:, ('1st', '2nd', '3rd', '4th', '5th', '6th', '7th')]
df_result = pd.concat([first_df, second_df, third_df, fourth_df]), axis=1)"""

data = pd.read_csv('resulting_dfs\\for_tagged_december\\result_grouped_feats.csv', delimiter='\t',
                        names=col_names)

text_name = []
for elem in data['name']:
    text_name.append(int(elem.split('\\')[-1].split('.')[0]))

labels = joblib.load('labels_for_tagged_55')
new_labels = [labels[i] for i in text_name]

df_numbers = data.loc[:, ('first_class', '2nd_class_W', '2nd_class_S', '3rd_class_W', '3rd_class_S', '4th_class_W',
                              '4th_class_S')]

clusterize = KMeans(n_clusters=4, random_state=42)
output = clusterize.fit_predict(df_numbers)

"""first = []
second = []
third = []
fourth = []
for i, res in enumerate(output):
    if i < 20:
        # print('actual:', 1, 'predicted:', res)
        first.append(res)
    elif 19 < i < 40:
        # print('actual:', 2, 'predicted:', res)
        second.append(res)
    elif 39 < i < 60:
        # print('actual:', 3, 'predicted:', res)
        third.append(res)
    else:
        # print('actual:', 4, 'predicted:', res)
        fourth.append(res)

print(Counter(output))
print(Counter(first))
print(Counter(second))
print(Counter(third))
print(Counter(fourth))

y = [0 for i in range(20)] + [1 for i in range(20)] + [2 for i in range(20)] + [3 for i in range(20)]
#y = [0 for i in range(40)] + [1 for i in range(40)]"""
data_clusterization = []

data_clusterization.append(({
        'ARI': metrics.adjusted_rand_score(new_labels, clusterize.labels_),
        'AMI': metrics.adjusted_mutual_info_score(new_labels, clusterize.labels_),
        'Homogenity': metrics.homogeneity_score(new_labels, clusterize.labels_),
        'Completeness': metrics.completeness_score(new_labels, clusterize.labels_),
        'V-measure': metrics.v_measure_score(new_labels, clusterize.labels_),
        'Silhouette': metrics.silhouette_score(df_numbers, clusterize.labels_)}))

results = pd.DataFrame(data=data_clusterization, columns=['ARI', 'AMI', 'Homogenity',
                                           'Completeness', 'V-measure',
                                           'Silhouette'],
                       index=['K-means'])

print(results)

if vizualize:
    pca = PCA(n_components=2)
    X_top_plot = pca.fit_transform(df_numbers)
    plt.figure()
    clusters = new_labels
    plt.scatter(X_top_plot[:, 0], X_top_plot[:, 1], c=clusters)  # plot all points
    plt.show()
else:
    data['filename'] = [str(elem) + '.txt' for elem in text_name]
    data['text_label'] = new_labels
    data['cluster'] = clusterize.labels_
    data.to_csv('resulting_dfs\\for_tagged_december\\result_grouped_feats_for_analysis.csv')

"""col_names.append('predicted cluster')


def save_csv(dataframe, name, labels):
    dataframe['predicted cluster'] = labels
    dataframe.to_csv(name, encoding='utf-8', columns=col_names)"""


"""save_csv(first, 'first_df.csv', clusterize.labels_[:20])
save_csv(second, 'second_df.csv', clusterize.labels_[20:40])
save_csv(third, 'third_df.csv', clusterize.labels_[40:60])
save_csv(fourth, 'fourth_df.csv', clusterize.labels_[60:])"""
