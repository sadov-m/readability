# coding: utf-8

import requests
import os
import time
from tqdm import tqdm


def estimations_for_text(text_to_estimate):
    response = requests.post("http://api.plainrussian.ru/api/1.0/ru/measure/", data={"text": text_to_estimate})
    time.sleep(0.5)

    # dict_keys(['metrics', 'status', 'lang', 'debug', 'indexes'])
    grades = response.json()['indexes']
    text_metrics = response.json()['metrics']

    text_characteristics = [grades['grade_SMOG'], grades['index_SMOG'], grades['index_fk'], grades['index_cl'],
                            grades['index_dc'], grades['index_ari'], text_metrics['chars'], text_metrics['spaces'],
                            text_metrics['letters'], text_metrics['n_words'], text_metrics['n_sentences'],
                            text_metrics['n_complex_words'], text_metrics['n_simple_words'], text_metrics['avg_slen'],
                            text_metrics['avg_syl'], text_metrics['c_share']]

    return [str(elem) for elem in text_characteristics]


def extracting_texts_paths(path_dir):
    container = []

    for d, dirs, files in os.walk(path_dir):
        for f in files:
            filepath = os.path.join(d, f)  # формирование адреса
            container.append(filepath)  # добавление адреса в список

    return container


def create_an_output_table(list_of_paths, func_for_estimations, header_of_table):

    with open(path_for_api+ '\output_for_'+path_for_api.split('\\')[-1]+ '.csv', 'w', encoding='utf-8') as writer:
        writer.write(header_of_table + '\n')

        for i in tqdm(range(len(list_of_paths))):
            length = len(list_of_paths)

            with open(list_of_paths[i], 'r', encoding='utf-8') as opener:
                readability_result = func_for_estimations(opener.read())
                readability_result.insert(0, list_of_paths[i].split('\\')[-1])

                if i == length-1:
                    writer.write(', '.join(readability_result))
                else:
                    writer.write(', '.join(readability_result)+'\n')


if __name__ == '__main__':
    header = 'filename, audience, readability level (SMOG), Flesch-Kincaid, Coleman-Liau index, ' \
             'Dale-Chale readability formula, Automated Readability Index, # of chars, # of spaces, # of letters, ' \
             '# of words, # of sentences, # of complex words, # of simple words, average # of words per sentence, ' \
             'average # of syllables per sentence, % of complex words'

    path_for_api = input('type in the path to a folder with texts to analyze: ')

    create_an_output_table(extracting_texts_paths(path_for_api), estimations_for_text, header)
