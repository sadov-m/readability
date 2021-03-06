import pandas as pd
import re
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from collections import Counter

data = pd.read_csv(r'dataset\\all_data\\result.csv')

expert_labels = []
for elem in data['filename']:
    expert_labels.append(int(re.findall('\d+', elem)[0]))

data = data.drop(labels=['filename'], axis=1)

two_class = False
if two_class:
    for j in range(len(expert_labels)):
        if expert_labels[j] == 4:
            expert_labels[j] = 3

to_be_dropped = []

normalization = True
if normalization:
    for feature in data:
        delimiter = data[feature].max()-data[feature].min()

        if delimiter == 0:
            to_be_dropped.append(feature)
        else:
            data[feature] = (data[feature] - data[feature].mean())/(delimiter)

formal = data.loc[:, ['avg_len_in_chars', 'len_in_chars', 'len_in_words']]

data = data.drop(labels=['len_in_chars', 'len_in_words'], axis=1)

sylls_morphs_accents = data.loc[: ,['stressed_first_v_W', 'c_in_the_end_W', 'c_in_the_beginning_W', 'two_syl_open_syls_W',
                                    'three_syl_open_syls_W', 'one_syl_W', 'two_syl_W', 'one_syl_cvc_W', 'one_syl_begin_cc_W',
                                    'two_syl_begin_cc_W', 'two_syl_1th_stressed_W', 'three_syl_2nd_stressed_W', 'two_syl_2nd_stressed_W',
                                    'three_syl_1th_stressed_W', 'three_syl_cv_pattern_W', 'four_syl_cv_pattern_W', 'nom_W',
                                    'acc_W', 'dat_W', 'abl_W', 'verbs_pers_S', 'parenth_S', 'one_syl_end_cc_W', 'two_syl_middle_cc_W',
                                    'three_syl_begin_cc_W', 'three_syl_middle_cc_W', 'three_syl_end_cc_W', 'four_syl_cc_on_the_edge_W',
                                    'five_syl_cv_pattern_W', 'adv_W', 'gen_W', 'ins_W', 'numeral_W', 'a_pro_W', 'coord_conjs_num_S',
                                    's_pro_S', 'three_syl_3rd_stressed_W', 'three_syl_cc_on_the_edge_W', 'five_syl_cc_on_the_edge_W',
                                    'alt_conjs_num_S', 'abstr_nouns_rate_S', 'foreign_W']]

lexs = data.loc[:, ['rare_obsol_W', 'avg_W_freq_S', 'avg_W_Rs_S', 'avg_W_Ds_S', 'avg_W_Docs_S', 'oov_words_rate_S',
                    'N_top_200_rate_S', 'N_top_400_rate_S', 'N_top_600_rate_S', 'N_top_800_rate_S', 'N_top_1000_rate_S',
                    'V_top_200_rate_S', 'V_top_400_rate_S', 'V_top_600_rate_S', 'V_top_800_rate_S', 'V_top_1000_rate_S',
                    'A_top_200_rate_S', 'A_top_400_rate_S', 'A_top_600_rate_S', 'A_top_800_rate_S', 'A_top_1000_rate_S']]

syntax = data.loc[:, ['sent_simple_S', 'sent_two_homogen_S', 'sent_three_homogen_S', 'no_predic_S', 'sent_complic_soch_S',
                    'sent_complic_depend_S', 'inverse_S']]

if normalization:
    print(to_be_dropped)
    if to_be_dropped:
        data = data.drop(labels=to_be_dropped, axis=1)
        sylls_morphs_accents = sylls_morphs_accents.drop(labels=['foreign_W'], axis=1)

#data = data.loc[:, ['inverse_S', 'abstr_nouns_rate_S', 'parenth_S', 'acc_W', 'two_syl_W', 'three_syl_cc_on_the_edge_W',
#       'avg_len_in_chars', 'gen_W', 'sent_complic_depend_S', 'three_syl_3rd_stressed_W', 'two_syl_middle_cc_W',
#       'a_pro_W', 'five_syl_cc_on_the_edge_W', 'one_syl_W', 'dat_W', 'three_syl_2nd_stressed_W', 'N_top_800_rate_S']]

for data_type in [data]:#[formal, sylls_morphs_accents, lexs, syntax, data]:

    X_train, X_test, y_train, y_test = train_test_split(data_type, expert_labels, test_size=1/3, random_state=42,
                                                        stratify=expert_labels)

    """print('X_train:', Counter(expert_labels))
    print('y_train:', Counter(y_train))
    print('y_test:', Counter(y_test))"""

    #classifier = RidgeClassifier(random_state=42)#,alpha=0.8, fit_intercept=True, normalize=True)
    classifier = RandomForestClassifier(random_state=42, n_jobs=-1, max_features=0.72,
                                        class_weight='balanced', n_estimators=48)

    """predicted = cross_val_predict(classifier, data_type, expert_labels, cv = 3)
    print(classification_report(expert_labels, predicted))
    conf_matrix = confusion_matrix(expert_labels, predicted)
    print(conf_matrix)"""

    classifier.fit(X_train, y_train)

    save_model = False
    if save_model:
        joblib.dump(classifier, 'trained_model')

    y_pred = classifier.predict(X_test)

    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_pred, y_test)
    print(conf_matrix)

    spec_score = 0
    full_match = 0
    part_match = 0
    penalty = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            spec_score += 1
            full_match += 1
        elif abs(y_pred[i]-y_test[i]) == 1:
            if y_test[i] == 3:
                pass
            else:
                spec_score += 0.25
                part_match += 1
        elif abs(y_pred[i]-y_test[i]) == 2:
            spec_score -= 0.5
            penalty += 1

    print('our_score', spec_score/len(y_pred), spec_score)
    print(full_match, part_match, penalty)

    print_heatmap = False
    if print_heatmap:
        plt.figure(figsize = (10,7))
        sb.heatmap(conf_matrix, annot=True, xticklabels=['2 класс true','3 класс true','4 класс true'],
                   yticklabels=['2 класс pred','3 класс pred','4 класс pred'])
        plt.show()

    print_importances = False
    if print_importances:
        importances = classifier.feature_importances_
        features_names = [name for name in data_type]

        indices = np.argsort(importances)[::-1]

        # print the feature ranking
        print("Feature ranking:")

        for f in range(data_type.shape[1]):
            print("%d. %s (%f)" % (f + 1, features_names[indices[f]], importances[indices[f]]))
