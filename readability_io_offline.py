import IB_metrics_readability
from readability_io_api import extracting_texts_paths
from tqdm import tqdm


def create_an_output_table_offline(list_of_paths, func, header_of_table):

    with open(path_for_offline_api+ '\output_for_'+path_for_offline_api.split('\\')[-1]+ '.csv', 'w', encoding='utf-8') as writer:
        writer.write(header_of_table + '\n')

        for i in tqdm(range(len(list_of_paths))):

            if list_of_paths[i][-3:] == 'csv':
                pass
            else:
                length = len(list_of_paths)

                with open(list_of_paths[i], 'r', encoding='utf-8') as opener:
                    readability_result = func(opener.read())
                    name = list_of_paths[i].split('\\')[-1]

                    if i == length-1:
                        writer.write(name + ', ' + ', '.join([str(round(value, 3))
                                                              for value in readability_result.values()]))
                    else:
                        writer.write(name + ', ' + ', '.join([str(round(value, 3))
                                                              for value in readability_result.values()]) + '\n')


if __name__ == '__main__':
    header = 'filename, Flesch-Kincaid, Coleman-Liau index, Dale-Chale readability formula, readability level (SMOG), ' \
             'Automated Readability Index, # of chars, # of spaces, # of letters, # of words, # of sentences, ' \
             '# of complex words, # of simple words, average # of words per sentence, ' \
             'average # of syllables per sentence, % of complex words'

    path_for_offline_api = input("type in the path to a folder with texts to analyze: ")

    create_an_output_table_offline(extracting_texts_paths(path_for_offline_api), IB_metrics_readability.calc_readability_metrics, header)
