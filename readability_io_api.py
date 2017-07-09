# coding: utf-8

import requests
import os
import time

text = """Итальянский форвард Джузеппе Росси в новом сезоне не будет выступать за «Сельту», сообщает официальный сайт клуба.

Арендное соглашение «Сельты» с «Фиорентиной» истекло в конце июня, и клуб решил его не продлевать.

Причины такого решения испанцев не озвучиваются, но, вероятно, дело в очередном разрыве крестообразных связок у 30-летнего форварда, который выбил его из игры как минимум на полгода.

Напомним, что у Росси закончился не только арендный договор с «Сельтой», но и контракт с «Фиорентиной», так что в новый клуб он сможет перебраться как свободный агент.

В минувшем сезоне Джузеппе Росси сыграл за «Сельту» 29 матчей и забил шесть голов во всех турнирах."""
path = input('type in the path to a folder whhich contains corpus: ')# r"C:\Users\Ольга\Desktop\test_collection"

def estimations_for_text(text_to_estimate):
    response = requests.post("http://api.plainrussian.ru/api/1.0/ru/measure/", data={"text": text_to_estimate})
    time.sleep(0.1)

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


def create_an_output_table(list_of_paths):

    with open(path+'\output_for_'+path.split('\\')[-1]+'.csv', 'w', encoding='utf-8') as writer:
        header = 'filename, audience, readability level (SMOG), Flesch-Kincaid, Coleman-Liau index, Dale-Chale readability formula, ' \
                 'Automated Readability Index, # of chars, # of spaces, # of letters, # of words, # of sentences, ' \
                 '# of complex words, # of simple words, average # of words per sentence, average # of syllables per sentence, ' \
                 '% of complex words'
        writer.write(header+'\n')

        for ind, link in enumerate(list_of_paths):
            length = len(list_of_paths)

            with open(link, 'r', encoding='utf-8') as opener:
                readability_result = estimations_for_text(opener.read())
                readability_result.insert(0, link.split('\\')[-1])

                if ind == length:
                    writer.write(', '.join(readability_result))
                else:
                    writer.write(', '.join(readability_result)+'\n')

create_an_output_table(extracting_path(path))
