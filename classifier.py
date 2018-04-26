import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from collections import Counter

data = pd.read_csv(r'dataset\\all_data\\result.csv')

expert_labels = []
for elem in data['filename']:
    expert_labels.append(re.findall('\d+', elem)[0])

#for i in range(len(data['filename'])):
#    print(data['filename'][i], expert_labels[i])

data = data.drop(labels=['filename'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, expert_labels, test_size=0.33, random_state=42,
                                                    stratify=expert_labels)

"""print('X_train:', Counter(expert_labels))
print('y_train:', Counter(y_train))
print('y_test:', Counter(y_test))"""

classifier = RandomForestClassifier(n_estimators=42, criterion='gini', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=27,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                    bootstrap=True, oob_score=False, n_jobs=-1, random_state=42, verbose=0,
                                    warm_start=False, class_weight='balanced')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))

"""for i in range(len(y_test)):
    if y_test[i] == '2':
        print('y_test:', y_test[i],
              'y_pred:', y_pred[i])"""

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
plt.figure(figsize = (10,7))
sb.heatmap(conf_matrix, annot=True, xticklabels=['2 класс true','3 класс true','4 класс true'],
           yticklabels=['2 класс pred','3 класс pred','4 класс pred'])
plt.show()

importances = classifier.feature_importances_
features_names = [name for name in data]

std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# print the feature ranking
print("Feature ranking:")

for f in range(data.shape[1]):
    print("%d. %s (%f)" % (f + 1, features_names[indices[f]], importances[indices[f]]))
