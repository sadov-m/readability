import IB_metrics_readability
from readability_io_api import extracting_texts_paths
import numpy as np
import re
from subprocess import call
import os

path_for_pipeline = input('type in the path to a folder with texts to analyze: ')

paths = extracting_texts_paths(path_for_pipeline)
texts = []

# latin finder
regexp_latin_words_finder = re.compile(r"[A-Za-z]+-*[A-Za-z]*")


# a func to open txt wordlists
def open_wordlist(path_to_txt):

    with open(path_to_txt, encoding='utf-8') as list_opener:
        wordlist = list_opener.read().split('\n')

    return wordlist


def classify_5_to_7(paths_list):
    results = []
    for path in paths_list:
        with open(path, encoding='utf-8') as file_opener:
            text = file_opener.read().strip()
            texts.append(text)

            # latin words + swear words
            latin_words_qty = regexp_latin_words_finder.findall(text)
            swear_flag = False

            output_path = path.split('\\')[-1]
            call_string = r'C:/Users/Mike/PycharmProjects/ru-syntax/bin/mystem.exe -cgnid {}' \
                          r' tmp\{}'.format(path, output_path)

            call(call_string)
            mystem_result = open_wordlist('tmp/' + output_path)

            for word_gr in mystem_result:

                try:
                    gr = re.findall('\w+', word_gr.split('|')[0])
                    if 'обсц' in gr:
                        swear_flag = True
                        break
                except:
                    pass

            metrics = IB_metrics_readability.calc_readability_metrics(text)
            avg = np.mean([metrics['FK'], metrics['CL'], metrics['DC'], metrics['SMOG'], metrics['ARI']])

            if swear_flag:
                results.append('swear')
            elif len(latin_words_qty) > 1:
                results.append('latin')
            elif metrics['# of sentences'] <= 3 or metrics['# of words'] <= 12 or metrics['# of chars'] <= 50 \
                    or metrics['# of chars']/metrics['# of words'] <= 2.5:
                results.append('short')
            elif metrics['# of sentences'] >= 50 or metrics['# of words'] >= 1200:
                results.append('long')
            elif avg < 3:
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


results_classified = classify_5_to_7(paths)
print(results_classified)

tmp_files = list(os.walk(os.path.dirname(__file__)+'/tmp'))[0][2]

for tmp_file in tmp_files:
    os.remove(os.path.dirname(__file__)+'/tmp/'+tmp_file)

"""with open('science_texts_with_labels.csv', 'w', encoding='utf-8') as writer:
    for i in range(len(texts)):
        writer.write(texts[i]+'<delim>'+'5-7'+'<delim>'+str(results_classified[i])+'\n')"""

"""import matplotlib.pyplot as plt

num_bins = 10
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()"""
