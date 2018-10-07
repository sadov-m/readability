from readability_io_api import extracting_texts_paths
import os
from subprocess import call
import get_tokens_and_sent_segmentation
import re
import numpy as np
from math import log

path_for_pipeline = input('type in the path to a folder with texts to analyze: ')
paths = extracting_texts_paths(path_for_pipeline)
data = []


def open_wordlist(path_to_txt):

    with open(path_to_txt, encoding='utf-8') as list_opener:
        wordlist = list_opener.read().split('\n')

    return wordlist


# loading lists of conjunctions according to frequency
first_100, second_100, third_100, fourth_100 = open_wordlist(os.path.dirname(__file__)+'/conjs_acc_to_freqs/one_hundred'),\
                                               open_wordlist(os.path.dirname(__file__)+'/conjs_acc_to_freqs/two_hundred'),\
                                               open_wordlist(os.path.dirname(__file__)+'/conjs_acc_to_freqs/three_hundred'), \
                                               open_wordlist(os.path.dirname(__file__) + '/conjs_acc_to_freqs/four_hundred')
other_conjs = ['разве', 'сиречь', 'якобы']

# loading frequency dict by Sharov and Lyashevskaya
# 1st string - header, last - null
freq_dict_rnc = open_wordlist(os.path.dirname(__file__) + '/freq_rnc/freqrnc2011.csv')[1:-1]
freq_dict_rnc = [string.split('\t') for string in freq_dict_rnc]
freq_dict_lemmas = [string[0] for string in freq_dict_rnc]
freq_dict_freqs = [float(string[2]) for string in freq_dict_rnc]
freq_dict_Rs = [float(string[3]) for string in freq_dict_rnc]
freq_dict_Ds = [float(string[4]) for string in freq_dict_rnc]
freq_dict_Docs = [float(string[5]) for string in freq_dict_rnc]
min_dict_freq = min(freq_dict_freqs)

# loading top-n word lists
nouns_top_1000 = open_wordlist(os.path.dirname(__file__) + '/freq_rnc/nouns_top_1000.csv')[1:-1]
nouns_top_1000 = [string.split(';')[0] for string in nouns_top_1000]
verbs_top_1000 = open_wordlist(os.path.dirname(__file__) + '/freq_rnc/verbs_top_1000.csv')[1:-1]
verbs_top_1000 = [string.split(';')[0] for string in verbs_top_1000]
adjs_top_1000 = open_wordlist(os.path.dirname(__file__) + '/freq_rnc/adjs_top_1000.csv')[1:-1]
adjs_top_1000 = [string.split(';')[0] for string in adjs_top_1000]

file_names = []
texts = []
avg_chars_lens = []
total_chars_qty = []
total_words_qty = []
sentences_qty = []

regexp_abbr_finder = re.compile(r"\b(?:[A-ZА-Я][a-zа-я-]*){2,}")
regexp_latin_words_finder = re.compile(r"[A-Za-z]+-*[A-Za-z]*")
regexp_latin_nums_finder = re.compile(r"[IVXLCDM]+")
regexp_tough_number_finder = re.compile(r"[0-9]+[.,\/]{1}[0-9]+|[0-9]{5,}")
regexp_qty_marker_finder = re.compile(r"[А-Яа-яA-Za-z]+[\.\/][А-Яа-яA-Za-z]+")
regexp_punct_finder = re.compile(r"[^a-zA-Zа-яёА-ЯЁ0-9\s]")

sent_two_homogen_all = []
sent_three_homogen_all = []
sent_simple_all = []
sent_complic_soch_all = []
sent_complic_depend_all = []
no_predic_all = []
particip_clause_all = []
inverse_all = []

# sets with syntax roles for certain features
syntax_roles_partcp = ['релят', 'опред', 'оп-опред']

for path in paths:

    with open(path, encoding='utf-8') as file_opener:
        text = file_opener.read().strip()
        file_names.append(path)

        # sentense splitting + tokenization
        tokenizer = get_tokens_and_sent_segmentation.Text(fname=r'', text_in_string=text, path_input=False)
        texts.append(text.replace(',', ' <comma> ').replace('\n', '\t'))
        tokenizer.process()

        abbrs_qty = regexp_abbr_finder.findall(tokenizer.text)
        foreign_words = regexp_latin_words_finder.findall(tokenizer.text)
        latin_nums = regexp_latin_nums_finder.findall(tokenizer.text)
        compound_nums = regexp_tough_number_finder.findall(tokenizer.text)
        qty_markers = regexp_qty_marker_finder.findall(tokenizer.text)
        punctuation = regexp_punct_finder.findall(tokenizer.text)

        # estimating difficulty of the punctuation
        punct_score = 0

        for elem in punctuation:
            if elem in set('.,?!-…:;—'):
                punct_score += 1
            elif elem in set('()\"\"\\\/«»'):
                punct_score += 2
            elif elem in set('&<>@*#§ˆ¦ǀǁ©®™¤€$¢¥£•№°πƒ∫øØ∏∑√∞⊥∴≈≠≡=≤≥∧∨ʘ±+~'):
                punct_score += 5
            else:
                punct_score += 2.5

        # removing all the punctuation from tokens so as to count number of words in text
        tokenized_sents = [[token for token in sent if token.isalnum()] for sent in
                           tokenizer.get_sentence_segmentation()]

        # plain formal features: see variables names
        n_of_sents = len(tokenized_sents)
        n_of_words = sum([len(sent) for sent in tokenized_sents])
        list_of_words_len = [len(word) for sent in tokenized_sents for word in sent]  # not feature var
        total_chars_len = sum(list_of_words_len)  # also not feature var
        avg_chars_len = total_chars_len / len(list_of_words_len)

        avg_chars_lens.append(avg_chars_len)
        total_chars_qty.append(total_chars_len)
        total_words_qty.append(n_of_words)
        sentences_qty.append(n_of_sents)

        # lex features
        named_entities = abbrs_qty[:]  # Имена, географические названия
        parenth = []  # Вводные слова
        rare_obsol = []  # Редко употребляемые/устаревшие слова

        W_freqs = []  # Частоты слов в тексте
        W_Rs = []  # коэф. R слов в тексте
        W_Ds = []  # коэф. Жуйана слов в тексте
        W_Docs = []  # Док. частота слов в тексте

        oov_words = []  # Кол-во слов, не найденных в словаре
        abstr_nouns = []  # Кол-во абстрактных существительных

        # Списки топ-n частотных слов по частям речи
        num_of_top_200_nouns = []
        num_of_top_500_nouns = []
        num_of_top_1000_nouns = []

        num_of_top_200_verbs = []
        num_of_top_500_verbs = []
        num_of_top_1000_verbs = []

        num_of_top_200_adjs = []
        num_of_top_500_adjs = []
        num_of_top_1000_adjs = []

        # morph features
        verbs_pers = 0  # Глаголы в личной форме
        participles = 0  # Причастия, деепричастия
        nom = 0  # слова в номинативе
        gen = 0  # слова в генитиве
        acc = 0  # слова в аккузативе
        dat = 0  # слова в дативе
        ins = 0  # слова в творительном
        abl = 0  # слова в предложном

        abstr_endings = ['тье', 'ьё', 'ние', 'вие', 'ство', 'ация', 'ость', 'есть',
                         'изм', 'изна', 'ота', 'тика', 'тива']

        # counters for normalizing some morph features defined above
        nouns = 0
        verbs_gen = 0
        adjs = 0
        advs = 0
        conjs = 0

        conjs_weights = []
        conjs_inds = []

        # lex and morph features retrieval
        output_path = path.split('\\')[-1]
        call_string = r'C:/Users/Mike/PycharmProjects/ru-syntax/bin/mystem.exe -cgnid {}' \
                      r' tmp\{}'.format(path, output_path)

        call(call_string)
        mystem_result = open_wordlist('tmp/' + output_path)

        for ind, word_gr in enumerate(mystem_result):

            try:
                # word_gr['analysis']:
                gr = re.findall('\w+', word_gr.split('|')[0])

                if gr[1] in freq_dict_lemmas:
                    W_freqs.append(freq_dict_freqs[freq_dict_lemmas.index(gr[1])])
                    W_Rs.append(freq_dict_Rs[freq_dict_lemmas.index(gr[1])])
                    W_Ds.append(freq_dict_Ds[freq_dict_lemmas.index(gr[1])])
                    W_Docs.append(freq_dict_Docs[freq_dict_lemmas.index(gr[1])])
                else:  # если слово не было найдено в частот. словаре
                    W_freqs.append(min_dict_freq ** -1)
                    W_Rs.append(1)
                    W_Ds.append(1)
                    W_Docs.append(1)
                    oov_words.append(gr[1])

                if 'редк' in gr or 'затр' in gr or 'устар' in gr or 'обсц' in gr or 'искаж' in gr:
                    rare_obsol.append(gr[1])

                if 'вводн' in gr and gr[1] != 'хорошо':
                    parenth.append(gr[0])

                if 'гео' in gr or 'имя' in gr or 'фам' in gr or 'отч' in gr:
                    named_entities.append(gr[1])

                if 'S' in gr:
                    nouns += 1
                    last_2_chars = gr[1][-2:]
                    last_3_chars = gr[1][-3:]
                    last_4_chars = gr[1][-4:]
                    last_n_chars = [last_2_chars, last_3_chars, last_4_chars]

                    for n_chars in last_n_chars:
                        if n_chars in abstr_endings:
                            abstr_nouns.append(gr[1])
                            break

                    if gr[1] in nouns_top_1000:
                        N_ordinal = nouns_top_1000.index(gr[1])
                        if N_ordinal < 200:
                            num_of_top_200_nouns.append(gr[1])
                            num_of_top_500_nouns.append(gr[1])
                            num_of_top_1000_nouns.append(gr[1])
                        elif N_ordinal < 500:
                            num_of_top_500_nouns.append(gr[1])
                            num_of_top_1000_nouns.append(gr[1])
                        elif N_ordinal < 1000:
                            num_of_top_1000_nouns.append(gr[1])

                elif 'A' in gr:
                    adjs += 1
                    if gr[1] in adjs_top_1000:
                        A_ordinal = adjs_top_1000.index(gr[1])
                        if A_ordinal < 200:
                            num_of_top_200_adjs.append(gr[1])
                            num_of_top_500_adjs.append(gr[1])
                            num_of_top_1000_adjs.append(gr[1])
                        elif A_ordinal < 500:
                            num_of_top_500_adjs.append(gr[1])
                            num_of_top_1000_adjs.append(gr[1])
                        elif A_ordinal < 1000:
                            num_of_top_1000_adjs.append(gr[1])
                elif 'V' in gr:
                    verbs_gen += 1
                    if 'прич' not in gr and 'деепр' not in gr:
                        if gr[1] in verbs_top_1000:
                            if 'инф' not in gr:
                                verbs_pers += 1

                            V_ordinal = verbs_top_1000.index(gr[1])
                            if V_ordinal < 200:
                                num_of_top_200_verbs.append(gr[1])
                                num_of_top_500_verbs.append(gr[1])
                                num_of_top_1000_verbs.append(gr[1])
                            elif V_ordinal < 500:
                                num_of_top_500_verbs.append(gr[1])
                                num_of_top_1000_verbs.append(gr[1])
                            elif V_ordinal < 1000:
                                num_of_top_1000_verbs.append(gr[1])
                    elif 'прич' in gr or 'деепр' in gr:
                        participles += 1
                elif 'CONJ' in gr:
                    if gr[1] in first_100:
                        conjs_weights.append(1)
                        conjs_inds.append(ind)
                    elif gr[1] in second_100:
                        conjs_weights.append(1 + log(2, 100))
                        conjs_inds.append(ind)
                    elif gr[1] in third_100:
                        conjs_weights.append(1 + log(3, 100))
                        conjs_inds.append(ind)
                    elif gr[1] in fourth_100:
                        conjs_weights.append(1 + log(4, 100))
                        conjs_inds.append(ind)
                    elif gr[1] in other_conjs:
                        conjs_weights.append(1 + log(5, 100))
                        conjs_inds.append(ind)
                    else:
                        conjs_weights.append(1 + log(6, 100))
                        conjs_inds.append(ind)
                    conjs += 1
                elif 'ADV' in gr:
                    advs += 1

            except:
                pass

        avg_W_freq = np.mean(W_freqs)
        avg_W_Rs = np.mean(W_Rs)
        avg_W_Ds = np.mean(W_Ds)
        avg_W_Docs = np.mean(W_Docs)

        # searching for complicated conjs
        complicated_inds = {}
        beginning = -1
        for i in range(len(conjs_inds)):
            if conjs_inds[i] - 2 == conjs_inds[i - 1]:
                if beginning == -1:
                    beginning = i - 1
                else:
                    pass
            else:
                if beginning != -1:
                    complicated_inds[beginning] = (beginning, i - 1)
                beginning = -1

        new_conjs_weights = []
        last_ind = -1
        if complicated_inds:
            for i in range(len(conjs_weights)):
                if i in complicated_inds.keys():
                    new_conjs_weights.append(np.prod(conjs_weights[complicated_inds[i][0]: complicated_inds[i][1] + 1]))
                elif i <= last_ind:
                    pass
                else:
                    new_conjs_weights.append(conjs_weights[i])

        if new_conjs_weights == []:
            new_conjs_weights = conjs_weights

        # ru-syntax
        rusyntax_call_str = r'python C:\Users\Mike\PycharmProjects\ru-syntax\ru-syntax.py {}'.format(path)
        call(rusyntax_call_str)

        syntax_result = r'C:\Users\Mike\PycharmProjects\ru-syntax\out\{}'.format(path.split('\\')[-1].split('.')[0]
                                                                                 + '.conll')

        with open(syntax_result, encoding='utf-8') as syntax_opener:
            lines = syntax_opener.read().split('\n')

        # header: 0 - id, 1 - form, 2 - lemma, 3 - upostag, 4 - xpostag, 5 - feats, 6 - head
        # 7 deprel, 8 - deps, 9 - misc
        sents = []

        temp_sent = []
        for line in lines:
            if line == '':
                sents.append(temp_sent)
                temp_sent = []
            else:
                temp_sent.append(line.split('\t'))

        # synt features
        sent_two_homogen = 0  # Предложение с двумя однородными членами
        sent_three_homogen = 0  # Предложение с тремя однородными членами
        sent_simple_synt = 0  # Предложение с простым синтаксисом
        sent_complic_soch = 0  # Сложносочиненное предложение
        sent_complic_depend = 0  # Сложноподчиненное предложение
        no_predic = 0  # Безличное предложение
        particip_clauses_qty = 0  # Причастный оборот
        inverse = 0  # Обратный порядок слов

        sent_two_homogen_all.append([])
        sent_three_homogen_all.append([])
        sent_simple_all.append([])
        sent_complic_soch_all.append([''])
        sent_complic_depend_all.append([''])
        no_predic_all.append([])
        particip_clause_all.append([])
        inverse_all.append([])

        for sent in sents:
            sent_in_str_comma_repl = ' '.join([elem[1] for elem in sent]).replace(',', ' <comma> ')
            sent_in_str = ' '.join([elem[1] for elem in sent])
            words = re.findall('\w+', sent_in_str)

            predic = False
            predic_ids = []
            root_ids = []
            simple_syntax = True
            soch = 0
            heads_ids = [elem[6] for elem in sent]


            for i, elem in enumerate(sent):

                # case features extraction starts here
                if 'nom' in elem[5]:
                    nom += 1
                elif 'gen' in elem[5] or 'part' in elem[5]:
                    gen += 1
                elif 'acc' in elem[5]:
                    acc += 1
                elif 'dat' in elem[5]:
                    dat += 1
                elif 'ins' in elem[5]:
                    ins += 1
                elif 'abl' in elem[5] or 'loc' in elem[5]:
                    abl += 1

                if elem[7] == 'предик' and 'nom' in elem[5]:
                    predic = True
                    predic_ids.append(int(elem[0]))
                    root_ids.append(int(elem[6]))
                elif elem[7] == 'ROOT':
                    pass
                elif elem[7] in syntax_roles_partcp + ['PUNC', 'аппрокс-порядк']:
                    pass
                else:
                    simple_syntax = False

                if elem[7] in ['сент-соч', 'соч-союзн']:
                    if sent_complic_soch_all[-1][-1] != sent_in_str_comma_repl:
                        sent_complic_soch += 1
                        sent_complic_soch_all[-1].append(sent_in_str_comma_repl)
                elif elem[7] == 'подч-союзн' or elem[7] == 'изъясн' or elem[7] == 'релят':
                    if sent_complic_depend_all[-1][-1] != sent_in_str_comma_repl:
                        sent_complic_depend += 1
                        sent_complic_depend_all[-1].append(sent_in_str_comma_repl)
                elif elem[7] == 'сочин':
                    soch += 1

                elif elem[3] == 'PARTCP' and elem[7] in syntax_roles_partcp and elem[0] in heads_ids:
                    particip_clauses_qty += 1
                    particip_clause_all[-1].append(sent_in_str_comma_repl)

            for i in range(len(predic_ids)):
                if root_ids[i] < predic_ids[i]:
                    inverse += 1
                    inverse_all[-1].append(sent_in_str_comma_repl)
                    break

            if predic and simple_syntax:
                sent_simple_synt += 1
                sent_simple_all[-1].append(sent_in_str_comma_repl)

            if not predic and len(words) > 0:
                no_predic += 1
                no_predic_all[-1].append(sent_in_str_comma_repl)

            if soch == 1:
                sent_two_homogen += 1
                sent_two_homogen_all[-1].append(sent_in_str_comma_repl)
            elif soch == 2:
                sent_three_homogen += 1
                sent_three_homogen_all[-1].append(sent_in_str_comma_repl)

        w_n_str = str(n_of_words)
        s_n_str = str(n_of_sents)

        if adjs == 0:
            adjs = 1
        if nouns == 0:
            nouns = 1
        if verbs_gen == 0:
            verbs_gen = 1
        if conjs == 0:
            conjs = 1

        final_list_of_features = [avg_chars_len, total_chars_len, n_of_words, len(foreign_words)/n_of_words,
                                  len(latin_nums)/n_of_words, len(compound_nums)/n_of_words,
                                  len(qty_markers)/n_of_words, punct_score/len(punctuation), sum(new_conjs_weights)/conjs,
                                  len(named_entities)/nouns, len(parenth)/n_of_sents, len(rare_obsol)/n_of_words,
                                  avg_W_freq, avg_W_Rs, avg_W_Ds, avg_W_Docs, len(oov_words)/n_of_words,
                                  len(abstr_nouns)/nouns, participles/verbs_gen, verbs_pers/verbs_gen, advs/verbs_gen,
                                  len(num_of_top_200_nouns)/nouns, len(num_of_top_500_nouns)/nouns,
                                  len(num_of_top_1000_nouns)/nouns, len(num_of_top_200_adjs)/adjs,
                                  len(num_of_top_500_adjs)/adjs, len(num_of_top_1000_adjs)/adjs,
                                  len(num_of_top_200_verbs)/verbs_gen, len(num_of_top_500_verbs)/verbs_gen,
                                  len(num_of_top_1000_verbs)/verbs_gen, inverse/n_of_sents,
                                  particip_clauses_qty/n_of_sents, no_predic/n_of_sents, sent_two_homogen/n_of_sents,
                                  sent_three_homogen/n_of_sents, sent_simple_synt/n_of_sents,
                                  sent_complic_soch/n_of_sents, sent_complic_depend/n_of_sents]
        data.append(final_list_of_features)


header = """file_name, avg_chars_len, total_chars_len, n_of_words, foreign_words, latin_nums,
                                  compound_nums, qty_markers, punct_score, new_conjs_weights, named_entities,
                                  parenth, rare_obsol, avg_W_freq, avg_W_Rs, avg_W_Ds, avg_W_Docs,
                                  oov_words, abstr_nouns, participles, verbs_pers, advs, num_of_top_200_nouns,
                                  num_of_top_500_nouns, num_of_top_1000_nouns, num_of_top_200_adjs,
                                  num_of_top_500_adjs, num_of_top_1000_adjs, num_of_top_200_verbs,
                                  num_of_top_500_verbs, num_of_top_1000_verbs, inverse, particip_clauses_qty,
                                  no_predic, sent_two_homogen, sent_three_homogen, sent_simple_synt,
                                  sent_complic_soch, sent_complic_depend""".replace('\n', '')

save_output = True
if save_output:
    with open(path_for_pipeline+r'/result.csv', 'w', encoding='utf-8') as writer:
        writer.write(header + '\n')

        length = len(file_names) - 1
        for m in range(len(file_names)):

            string_to_write = file_names[m] + ',' + ','.join(list(map(str, data[m])))
            if m != length:
                writer.write(string_to_write + '\n')
            else:
                writer.write(string_to_write)

tmp_files = list(os.walk(os.path.dirname(__file__)+'/tmp'))[0][2]

for tmp_file in tmp_files:
    os.remove(os.path.dirname(__file__)+'/tmp/'+tmp_file)
