import IB_metrics_readability
from readability_io_api import extracting_texts_paths
import numpy as np

path_for_pipeline = input('type in the path to a folder with texts to analyze: ')

paths = extracting_texts_paths(path_for_pipeline)


def classify_5_to_7(paths_list):
    results = []
    for path in paths_list:
        with open(path, encoding='utf-8') as file_opener:
            text = file_opener.read().strip()

            metrics = IB_metrics_readability.calc_readability_metrics(text)
            avg = np.mean([metrics['FK'], metrics['CL'], metrics['DC'], metrics['SMOG'], metrics['ARI']])

            if avg < 2:
                results.append(4)  # too easy for such pupils to read
            elif 2 <= avg < 4:
                results.append(5)
            elif 4 <= avg < 7:
                results.append(6)
            elif 7 <= avg < 9:
                results.append(7)
            else:
                results.append(8)  # too hard for such pupils to read
    return results


#print(classify_5_to_7(paths))
