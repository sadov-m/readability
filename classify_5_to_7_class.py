import IB_metrics_readability
from readability_io_api import extracting_texts_paths
import numpy as np

path_for_pipeline = input('type in the path to a folder with texts to analyze: ')

paths = extracting_texts_paths(path_for_pipeline)
#x = []


def classify_5_to_7(paths_list):
    results = []
    for path in paths_list:
        with open(path, encoding='utf-8') as file_opener:
            text = file_opener.read().strip()

            metrics = IB_metrics_readability.calc_readability_metrics(text)
            avg = np.mean([metrics['FK'], metrics['CL'], metrics['DC'], metrics['SMOG'], metrics['ARI']])
            #x.append(avg)

            if avg < 3:
                results.append(4)  # too easy for such pupils to read
            elif 3 <= avg < 8.5:
                results.append(5)
            elif 8.5 <= avg < 11:
                results.append(6)
            elif 11 <= avg < 13.5:
                results.append(7)
            else:
                results.append(8)  # too hard for such pupils to read
    return results


"""import matplotlib.pyplot as plt

num_bins = 10
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()"""
