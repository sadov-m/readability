# coding: utf-8

import requests
import os
import time
from tqdm import tqdm

header = 'filename, audience, readability level (SMOG), Flesch-Kincaid, Coleman-Liau index, Dale-Chale readability formula, ' \
                 'Automated Readability Index, # of chars, # of spaces, # of letters, # of words, # of sentences, ' \
                 '# of complex words, # of simple words, average # of words per sentence, average # of syllables per sentence, ' \
                 '% of complex words'


def estimations_for_text(text_to_estimate):
    response = requests.post("http://api.plainrussian.ru/api/1.0/ru/measure/", data={"text": text_to_estimate})
    time.sleep(0.5)

    # dict_keys(['metrics', 'status', 'lang', 'debug', 'indexes'])
    grades = response.json()['indexes']
    text_metrics = response.json()['metrics']

    text_characteristics = [grades['grade_SMOG'], grades['index_SMOG'], grades['index_fk'], grades['index_cl'], grades['index_dc'],
                            grades['index_ari'], text_metrics['chars'], text_metrics['spaces'], text_metrics['letters'], text_metrics['n_words'], text_metrics['n_sentences'],
                            text_metrics['n_complex_words'], text_metrics['n_simple_words'], text_metrics['avg_slen'], text_metrics['avg_syl'],
                            text_metrics['c_share']]

    return [str(elem) for elem in text_characteristics]


def extracting_path(path_dir):
    container = []

    for d, dirs, files in os.walk(path_dir):
        for f in files:
            filepath = os.path.join(d, f)  # формирование адреса
            container.append(filepath)  # добавление адреса в список

    return container


def create_an_output_table(list_of_paths, func, header_of_table):

    with open(path+'\output_for_'+path.split('\\')[-1]+'.csv', 'w', encoding='utf-8') as writer:
        writer.write(header_of_table + '\n')

        for i in tqdm(range(len(list_of_paths))):
            length = len(list_of_paths)

            with open(list_of_paths[i], 'r', encoding='utf-8') as opener:
                readability_result = func(opener.read())
                readability_result.insert(0, list_of_paths[i].split('\\')[-1])

                if i == length-1:
                    writer.write(', '.join(readability_result))
                else:
                    writer.write(', '.join(readability_result)+'\n')

if __name__ == '__main__':
    path = input('type in the path to a folder which contains corpus: ')

    create_an_output_table(extracting_path(path), estimations_for_text, header)