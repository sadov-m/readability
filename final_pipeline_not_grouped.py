from readability_io_api import extracting_texts_paths
import get_tokens_and_sent_segmentation
from accent_lstm import text_accentAPI
import syllable_segmentation
from subprocess import call
import re
import os

path_for_pipeline = input('type in the path to a folder with texts to analyze: ')

file_names = []
texts = []

num_of_1st_class = []

num_of_2nd_class_W = []
num_of_2nd_class_S = []

num_of_3rd_class_W = []
num_of_3rd_class_S = []

num_of_4th_class_W = []
num_of_4th_class_S = []

avg_chars_lens = []
total_chars_lens = []
total_words_nums = []

# lists for all the features that were extracted acc to category
stressed_first_v_all = []
c_in_the_end_all = []
c_in_the_beginning_all = []
two_syl_open_syls_all = []
three_syl_open_syls_all = []
one_syl_all = []
two_syl_all = []

one_syl_cvc_all = []
one_syl_begin_cc_all = []
two_syl_begin_cc_all = []
two_syl_1th_stressed_all = []
three_syl_2nd_stressed_all = []
two_syl_2nd_stressed_all = []
three_syl_1th_stressed_all = []
three_syl_cv_pattern_all = []
four_syl_cv_pattern_all = []
nom_all = []
acc_all = []
dat_all = []
abl_all = []
sent_simple_all = []
sent_two_homogen_all = []
sent_three_homogen_all = []
no_predic_all = []
sent_complic_soch_all = []
verbs_pers_all = []
parenth_all = []

one_syl_end_cc_all = []
two_syl_middle_cc_all = []
three_syl_begin_cc_all = []
three_syl_middle_cc_all = []
three_syl_end_cc_all = []
four_syl_cc_on_the_edge_all = []
five_syl_cv_pattern_all = []
adv_all = []
gen_all = []
ins_all = []
coord_conjs_num_all = []
sent_complic_depend_all = []
inverse_all = []
numeral_all = []
a_pro_all = []
s_pro_all = []

three_syl_3rd_stressed_all = []
three_syl_cc_on_the_edge_all = []
five_syl_cc_on_the_edge_all = []
alt_conjs_num_all = []
rare_obsol_all = []
foreign_all = []
particip_clause_all = []

paths = extracting_texts_paths(path_for_pipeline)


# a func to transform a word to a mask of type CVC... where C is a consonant and V is a vowel
def get_word_mask(word):
    mask = []

    for sym in word:
        mask.append('V' if sym in text_accentAPI.VOWELS else 'C')

    return mask


def get_accent_syl_id(accentuated_char_id, list_of_syls_lengths):
    last = len(list_of_syls_lengths) - 1

    for k in range(len(list_of_syls_lengths)):
        if k == 0:
            if 0 <= accentuated_char_id < list_of_syls_lengths[0]:
                return 0
        elif k == last:
            if sum(list_of_syls_lengths[:k]) <= accentuated_char_id:
                return last
        else:
            if sum(list_of_syls_lengths[:k]) <= accentuated_char_id < sum(list_of_syls_lengths[:k+1]):
                return k


# a func to open txt's
def open_wordlist(path_to_txt):

    with open(path_to_txt, encoding='utf-8') as list_opener:
        wordlist = list_opener.read().split('\n')

    return wordlist


# loading lists of conjunctions for tokenization
alt_conjs, coord_conjs = open_wordlist('lex_dicts/противительные_союзы.txt'),\
                         open_wordlist('lex_dicts/сочинительные_союзы.txt')

# main loop
for path in paths:

    with open(path, encoding='utf-8') as file_opener:
        text = file_opener.read().strip()
        texts.append(text)
        file_names.append(path)

    # sentense splitting + tokenization
    tokenizer = get_tokens_and_sent_segmentation.Text(fname=r'', text_in_string=text, path_input=False)
    tokenizer.process()
    # removing all the punctuation from tokens so as to count number of words in text
    tokenized_sents = [[token for token in sent if token.isalnum()] for sent in tokenizer.get_sentence_segmentation()]

    # plain formal features: see variables names
    n_of_sents = len(tokenized_sents)
    n_of_words = sum([len(sent) for sent in tokenized_sents])
    list_of_words_len = [len(word) for sent in tokenized_sents for word in sent]  # auxiliary
    total_chars_len = sum(list_of_words_len)
    avg_chars_len = total_chars_len/len(list_of_words_len)

    avg_chars_lens.append(avg_chars_len)
    total_chars_lens.append(total_chars_len)
    total_words_nums.append(n_of_words)

    # print('num of sentences:', number_of_sents, 'num of words:', number_of_words, 'total chars:', total_chars_len)

    # accent_lstm
    accentuated = text_accentAPI.main([' '.join(sent) for sent in tokenized_sents])
    accent_positions = []
    words_only = []

    for line in accentuated:
        # print(line)
        line_clean = line.replace('_', '').split()

        for word in line_clean:

            if "'" in word:
                words_only.append(''.join(word.split("'")))

                accent_positions.append(word.index("'") - 1)  # ' stands after accentuated vowel
            else:
                words_only.append(word)

                word_mask = get_word_mask(word.lower())

                try:
                    accent_positions.append(word_mask.index('V'))
                except ValueError:
                    accent_positions.append('No')  # not found won't be passed for syllable segmentation script

    # syllable_segmentation

    # features listing
    one_syl = 0  # Слова из одного слога
    one_syl_begin_cc = 0  # Односложные слова с сочетанием согласных в начале слова
    one_syl_end_cc = 0  # Односложные слова с сочетанием согласных в конце слова
    one_syl_cvc = 0  # Слова с одним закрытым трехбуквенным слогом?

    one_syl_all.append([])
    one_syl_begin_cc_all.append([])
    one_syl_end_cc_all.append([])
    one_syl_cvc_all.append([])

    two_syl = 0  # Слова из двух слогов
    two_syl_1th_stressed = 0  # Ударение на первый слог в двусложных словах
    two_syl_2nd_stressed = 0  # Ударение на второй слог в двусложных словах
    two_syl_begin_cc = 0  # Двусложные слова с сочетанием согласных в начале слова
    two_syl_open_syls = 0  # Двусложные слова с открытым слогом
    two_syl_middle_cc = 0  # Двусложные слова с сочетанием согласных в середине слова

    two_syl_all.append([])
    two_syl_1th_stressed_all.append([])
    two_syl_2nd_stressed_all.append([])
    two_syl_begin_cc_all.append([])
    two_syl_open_syls_all.append([])
    two_syl_middle_cc_all.append([])

    three_syl_1th_stressed = 0  # Ударение на первый слог в трехсложных словах
    three_syl_2nd_stressed = 0  # Ударение на второй слог в трехсложных словах
    three_syl_3rd_stressed = 0  # Ударение на третий слог в трехсложных словах
    three_syl_open_syls = 0  # Трехсложные слова с открытым слогом
    three_syl_begin_cc = 0  # Трехсложные слова с сочетанием согласных в начале слова
    three_syl_middle_cc = 0  # Трехсложные слова с сочетанием согласных в середине слова
    three_syl_end_cc = 0  # Трехсложные слова с сочетанием согласных в конце слова
    three_syl_cv_pattern = 0  # Слова из трех слогов (чередование гласных и согласных)
    three_syl_cc_on_the_edge = 0  # Слова из трех слогов (сочленение согласных букв)

    three_syl_1th_stressed_all.append([])
    three_syl_2nd_stressed_all.append([])
    three_syl_3rd_stressed_all.append([])
    three_syl_open_syls_all.append([])
    three_syl_begin_cc_all.append([])
    three_syl_middle_cc_all.append([])
    three_syl_end_cc_all.append([])
    three_syl_cv_pattern_all.append([])
    three_syl_cc_on_the_edge_all.append([])

    four_syl_cv_pattern = 0  # Слова из четырех слогов (чередование гласных и согласных)
    four_syl_cc_on_the_edge = 0  # Слова из четырех слогов (сочленение согласных букв)

    four_syl_cv_pattern_all.append([])
    four_syl_cc_on_the_edge_all.append([])

    five_syl_cv_pattern = 0  # Слова из пяти слогов (чередование гласных и согласных)
    five_syl_cc_on_the_edge = 0  # Слова из пяти слогов (сочленение согласных букв

    five_syl_cv_pattern_all.append([])
    five_syl_cc_on_the_edge_all.append([])

    stressed_first_v = 0  # Ударные гласные в начале слова
    c_in_the_end = 0  # Согласные в конце слова
    c_in_the_beginning = 0  # Согласные в начале слова

    stressed_first_v_all.append([])
    c_in_the_end_all.append([])
    c_in_the_beginning_all.append([])

    # segmentation itself and syllable features retrieval
    for ind, word in enumerate(words_only):

        if accent_positions[ind] != 'No':

            if '-' in word:
                syllables = []
                word_parts = word.split('-')

                for part in word_parts:
                    syllables_part = syllable_segmentation.get_syllables(part).split('-')
                    syllables.extend(syllables_part)

                accent_pos = accent_positions[ind] - get_accent_syl_id(accent_positions[ind],
                                                                       [len(part) for part in word_parts])
                num_of_syls = len(syllables)

            else:
                syllables = syllable_segmentation.get_syllables(word).split('-')
                num_of_syls = len(syllables)
                accent_pos = accent_positions[ind]

            accent_syl_id = get_accent_syl_id(accent_pos, [len(syl) for syl in syllables])
            accent_id_in_syl = accent_pos - sum([len(syl) for syl in syllables[:accent_syl_id]])

            result = [''.join(get_word_mask(syl)) for syl in syllables]
            # print(words_only[ind], accent_positions[ind])
            # print(result, result[accent_syl_id], accent_syl_id, accent_id_in_syl)
            # print()

            # here comes the features!
            whole_mask = ''.join(result)

            if accent_pos == 0:
                stressed_first_v += 1
                stressed_first_v_all[-1].append(word)
            elif whole_mask[0] == 'C':
                c_in_the_beginning += 1
                c_in_the_beginning_all[-1].append(word)

            if whole_mask[-1] == 'C':
                c_in_the_end += 1
                c_in_the_end_all[-1].append(word)

            if num_of_syls == 1:
                one_syl += 1
                one_syl_all[-1].append(word)

                for syl in result:
                    if syl[:2] == 'CC':
                        one_syl_begin_cc += 1
                        one_syl_begin_cc_all[-1].append(word)
                    if syl[-2:] == 'CC':
                        one_syl_end_cc += 1
                        one_syl_end_cc_all[-1].append(word)
                    if syl == 'CVC':
                        one_syl_cvc += 1
                        one_syl_cvc_all[-1].append(word)

            if num_of_syls == 2:
                two_syl += 1
                two_syl_all[-1].append(word)

                if accent_syl_id == 0:
                    two_syl_1th_stressed += 1
                    two_syl_1th_stressed_all[-1].append(word)
                else:
                    two_syl_2nd_stressed += 1
                    two_syl_2nd_stressed_all[-1].append(word)

                if result[0][:2] == 'CC':
                    two_syl_begin_cc += 1
                    two_syl_begin_cc_all[-1].append(word)

                if result[0][-1]+result[1][-1] == 'VV':
                    two_syl_open_syls += 1
                    two_syl_open_syls_all[-1].append(word)

                if 'CC' in whole_mask[1:-1]:
                    two_syl_middle_cc += 1
                    two_syl_middle_cc_all[-1].append(word)

            if num_of_syls == 3:

                if accent_syl_id == 0:
                    three_syl_1th_stressed += 1
                    three_syl_1th_stressed_all[-1].append(word)
                elif accent_syl_id == 1:
                    three_syl_2nd_stressed += 1
                    three_syl_2nd_stressed_all[-1].append(word)
                else:
                    three_syl_3rd_stressed += 1
                    three_syl_3rd_stressed_all[-1].append(word)

                if result[0][-1]+result[1][-1]+result[2][-1] == 'VVV':
                    three_syl_open_syls += 1
                    three_syl_open_syls_all[-1].append(word)
                else:
                    if result[2][-2:] == 'CC':
                        three_syl_end_cc += 1
                        three_syl_end_cc_all[-1].append(word)

                prev_char = whole_mask[0]
                for j, char in enumerate(whole_mask[1:]):

                    if char == prev_char:
                        break
                    else:
                        prev_char = char

                else:
                    three_syl_cv_pattern += 1
                    three_syl_cv_pattern_all[-1].append(word)

                if result[0][:2] == 'CC':
                    three_syl_begin_cc += 1
                    three_syl_begin_cc_all[-1].append(word)

                if 'CC' in whole_mask[1:-1]:
                    three_syl_middle_cc += 1
                    three_syl_middle_cc_all[-1].append(word)

                if result[0][-1]+result[1][0] == 'CC' or result[1][-1]+result[2][0] == 'CC':
                    three_syl_cc_on_the_edge += 1
                    three_syl_cc_on_the_edge_all[-1].append(word)

            if num_of_syls == 4:
                prev_char = whole_mask[0]
                for j, char in enumerate(whole_mask[1:]):

                    if char == prev_char:
                        break
                    else:
                        prev_char = char

                else:
                    four_syl_cv_pattern += 1
                    four_syl_cv_pattern_all[-1].append(word)

                if result[0][-1]+result[1][0] == 'CC' or result[1][-1]+result[2][0] == 'CC' or \
                                        result[2][-1]+result[3][0] == 'CC':
                    four_syl_cc_on_the_edge += 1
                    four_syl_cc_on_the_edge_all[-1].append(word)

            if num_of_syls == 5:
                prev_char = whole_mask[0]
                for j, char in enumerate(whole_mask[1:]):

                    if char == prev_char:
                        break
                    else:
                        prev_char = char

                else:
                    five_syl_cv_pattern += 1
                    five_syl_cv_pattern_all[-1].append(word)

                if result[0][-1] + result[1][0] == 'CC' or result[1][-1] + result[2][0] == 'CC' or \
                                        result[2][-1] + result[3][0] == 'CC' or result[3][-1] + result[4][0] == 'CC':
                    five_syl_cc_on_the_edge += 1
                    five_syl_cc_on_the_edge_all[-1].append(word)

    # mystem analyzer
    analyzed_sents = []

    # lex features
    parenth = 0  # Вводные слова
    rare_obsol = 0  # Редко употребляемые/устаревшие слова
    alt_conjs_num = 0  # Противительные союзы
    coord_conjs_num = 0  # Сочинительные союзы
    foreign = 0  # Иностранные слова

    parenth_all.append([])
    rare_obsol_all.append([])
    alt_conjs_num_all.append([])
    coord_conjs_num_all.append([])
    foreign_all.append([])

    # morph features
    verbs_pers = 0  # Глаголы в личной форме
    nom = 0  # слова в номинативе
    gen = 0  # слова в генитиве
    acc = 0  # слова в аккузативе
    dat = 0  # слова в дативе
    ins = 0  # слова в творительном
    abl = 0  # слова в предложном
    numeral = 0  # числительные
    a_pro = 0  # adj-pron
    s_pro = 0  # subj-pron
    adv = 0  # наречия

    verbs_pers_all.append([])
    nom_all.append([])
    gen_all.append([])
    acc_all.append([])
    dat_all.append([])
    ins_all.append([])
    abl_all.append([])
    numeral_all.append([])
    a_pro_all.append([])
    s_pro_all.append([])
    adv_all.append([])

    # features for normalizing
    nouns = 0
    verbs = 0

    # lex and morph features retrieval
    output_path = path.split('\\')[-1]
    call_string = r'C:\Users\Mike\PycharmProjects\ru-syntax\bin\mystem.exe -n -l -i -d {}' \
                  r' tmp\{}'.format(path, output_path)

    call(call_string)
    mystem_result = open_wordlist('tmp\\'+output_path)

    for word_gr in mystem_result:

        try:
            # word_gr['analysis']:
            gr = re.findall('\w+', word_gr.split('|')[0])

            if 'редк' in gr or 'устар' in gr or 'гео' in gr:  # words with 'гео' tag are considered to be rare
                rare_obsol += 1
                rare_obsol_all[-1].append(gr[0])

            if 'вводн' in gr:
                parenth += 1
                parenth_all[-1].append(gr[0])
            elif 'S' in gr:
                nouns += 1
            elif 'V' in gr:
                verbs += 1
                if 'л' in gr:
                    verbs_pers += 1
                    verbs_pers_all[-1].append(gr[0])
            elif 'ADV' in gr:
                adv += 1
                adv_all[-1].append(gr[0])
            elif 'NUM' in gr:
                numeral += 1
                numeral_all[-1].append(gr[0])
            elif 'APRO' in gr:
                a_pro += 1
                s_pro_all[-1].append(gr[0])
            elif 'SPRO' in gr:
                s_pro += 1
                s_pro_all[-1].append(gr[0])
            elif 'CONJ' in gr:
                if gr[0] in coord_conjs:
                    coord_conjs_num += 1
                    coord_conjs_num_all[-1].append(gr[0])
                elif gr[0] in alt_conjs:
                    alt_conjs_num += 1
                    alt_conjs_num_all[-1].append(gr[0])

            if 'им' in gr:
                nom += 1
                nom_all[-1].append(gr[0])
            elif 'род' in gr:
                gen += 1
                gen_all[-1].append(gr[0])
            elif 'вин' in gr:
                acc += 1
                acc_all[-1].append(gr[0])
            elif 'дат' in gr:
                dat += 1
                dat_all[-1].append(gr[0])
            elif 'твор' in gr:
                ins += 1
                ins_all[-1].append(gr[0])
            elif 'пр' in gr:
                abl += 1
                abl_all[-1].append(gr[0])

        except:
            pass

    # ru-syntax
    rusyntax_call_str = r'python C:\Users\Mike\PycharmProjects\ru-syntax\ru-syntax.py {}'.format(path)
    call(rusyntax_call_str)

    syntax_result = r'C:\Users\Mike\PycharmProjects\ru-syntax\out\{}'.format(path.split('\\')[-1].split('.')[0] + '.conll')

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
    sent_simple = 0  # Предложение с простым синтаксисом
    sent_complic_soch = 0  # Сложносочиненное предложение
    sent_complic_depend = 0  # Сложноподчиненное предложение
    no_predic = 0  # Безличное предложение
    particip_clause = 0  # Причастный оборот
    inverse = 0  # Обратный порядок слов

    sent_two_homogen_all.append([])
    sent_three_homogen_all.append([])
    sent_simple_all.append([])
    sent_complic_soch_all.append([])
    sent_complic_depend_all.append([])
    no_predic_all.append([])
    particip_clause_all.append([])
    inverse_all.append([])

    for sent in sents:
        sent_in_str = ' '.join([elem[1] for elem in sent])
        predic = False
        predic_ids = []
        root_ids = []
        not_simple = False
        soch = 0

        for i, elem in enumerate(sent):
            if elem[7] == 'предик':
                predic = True
                predic_ids.append(int(elem[0]))
                root_ids.append(int(elem[6]))
            elif elem[7] == 'ROOT':
                pass
            elif elem[7] == 'опред' or 'PUNC':
                pass
            else:
                not_simple = True

            if elem[7] == 'сент-соч':
                sent_complic_soch += 1
                sent_complic_soch_all[-1].append(sent_in_str)
            elif elem[7] == 'подч-союзн' or elem[7] == 'изъясн' or elem[7] == 'релят':
                sent_complic_depend += 1
                sent_complic_depend_all[-1].append(sent_in_str)
            elif elem[7] == 'сочин':
                soch += 1

            if elem[3] == 'NID':
                foreign += 1
                foreign_all[-1].append(elem[1])
            elif elem[3] == 'PARTCP' and elem[7] == 'опред':
                particip_clause += 1
                particip_clause_all[-1].append(sent_in_str)

        for i in range(len(predic_ids)):
            if root_ids[i] < predic_ids[i]:
                inverse += 1
                inverse_all[-1].append(sent_in_str)
                break

        if len(sent) < 5 and predic and not not_simple:
            sent_simple += 1
            sent_simple_all[-1].append(sent_in_str)

        if not predic:
            no_predic += 1
            no_predic_all[-1].append(sent_in_str)

        if soch == 1:
            sent_two_homogen += 1
            sent_two_homogen_all[-1].append(sent_in_str)
        elif soch == 2:
            sent_three_homogen += 1
            sent_three_homogen_all[-1].append(sent_in_str)

    # print(number_of_sents, number_of_words, len(list_of_words_len), total_chars_len, avg_chars_len)

    # 7
    first_level = [stressed_first_v / n_of_words, c_in_the_end / n_of_words, c_in_the_beginning / n_of_words,
                   two_syl_open_syls / n_of_words, three_syl_open_syls / n_of_words, one_syl / n_of_words,
                   two_syl / n_of_words]
    num_of_1st_class.append(first_level)

    # 20
    second_level = [one_syl_cvc / n_of_words, one_syl_begin_cc / n_of_words, two_syl_begin_cc / n_of_words,
                    two_syl_1th_stressed / n_of_words, three_syl_2nd_stressed / n_of_words,
                    two_syl_2nd_stressed / n_of_words, three_syl_1th_stressed / n_of_words,
                    three_syl_cv_pattern / n_of_words, four_syl_cv_pattern / n_of_words, nom / n_of_sents,
                    acc / n_of_sents, dat / n_of_sents, abl / n_of_sents, sent_simple / n_of_sents,
                    sent_two_homogen / n_of_sents, sent_three_homogen / n_of_sents, no_predic / n_of_sents,
                    sent_complic_soch / n_of_sents, verbs_pers / n_of_sents, parenth / n_of_sents]
    second_level_W_norm = second_level[:][:9]
    second_level_S_norm = second_level[:][9:]
    num_of_2nd_class_W.append(second_level_W_norm)
    num_of_2nd_class_S.append(second_level_S_norm)

    # 16
    third_level = [one_syl_end_cc / n_of_words, two_syl_middle_cc / n_of_words, three_syl_begin_cc / n_of_words,
                   three_syl_middle_cc / n_of_words, three_syl_end_cc / n_of_words, four_syl_cc_on_the_edge / n_of_words,
                   five_syl_cv_pattern / n_of_words, adv / verbs, gen / n_of_sents, ins / n_of_sents,
                   coord_conjs_num / n_of_sents, sent_complic_depend / n_of_sents, inverse / n_of_sents,
                   numeral / n_of_words, a_pro / nouns, s_pro / n_of_sents]
    third_level_W_norm = third_level[:][:8] + third_level[:][13:15]
    third_level_S_norm = third_level[:][8:13] + [third_level[:][15]]
    num_of_3rd_class_W.append(third_level_W_norm)
    num_of_3rd_class_S.append(third_level_S_norm)

    # 7
    fourth_level = [three_syl_3rd_stressed / n_of_words, three_syl_cc_on_the_edge / n_of_words,
                    five_syl_cc_on_the_edge / n_of_words, alt_conjs_num / n_of_sents, rare_obsol / n_of_words,
                    foreign / n_of_words, particip_clause / n_of_sents]
    fourth_level_W_norm = fourth_level[:][:3] + fourth_level[:][4:6]
    fourth_level_S_norm = [fourth_level[:][3]] + [fourth_level[:][6]]
    num_of_4th_class_W.append(fourth_level_W_norm)
    num_of_4th_class_S.append(fourth_level_S_norm)

first_level_names = """stressed_first_v c_in_the_end c_in_the_beginning two_syl_open_syls three_syl_open_syls
     one_syl two_syl""".split()
first_level_names = [string+'_W' for string in first_level_names]

second_level_names = """one_syl_cvc one_syl_begin_cc two_syl_begin_cc two_syl_1th_stressed
                    three_syl_2nd_stressed two_syl_2nd_stressed three_syl_1th_stressed
                    three_syl_cv_pattern four_syl_cv_pattern nom acc dat abl
                    sent_simple sent_two_homogen sent_three_homogen no_predic sent_complic_soch
                    verbs_pers parenth""".split()
second_level_W_names = second_level_names[:][:9]
second_level_W_names = [string + '_W' for string in second_level_W_names]
second_level_S_names = second_level_names[:][9:]
second_level_S_names = [string + '_S' for string in second_level_S_names]

third_level_names = """one_syl_end_cc two_syl_middle_cc three_syl_begin_cc three_syl_middle_cc
                   three_syl_end_cc four_syl_cc_on_the_edge five_syl_cv_pattern adv gen
                   ins coord_conjs_num sent_complic_depend inverse numeral a_pro s_pro""".split()
third_level_W_names = third_level_names[:][:8] + third_level_names[:][13:15]
third_level_W_names = [string + '_W' for string in third_level_W_names]
third_level_S_names = third_level_names[:][8:13] + [third_level_names[:][15]]
third_level_S_names = [string + '_S' for string in third_level_S_names]

fourth_level_names = """three_syl_3rd_stressed three_syl_cc_on_the_edge five_syl_cc_on_the_edge
                    alt_conjs_num rare_obsol foreign particip_clause""".split()
fourth_level_W_names = fourth_level_names[:][:3] + fourth_level_names[:][4:6]
fourth_level_W_names = [string + '_W' for string in fourth_level_W_names]
fourth_level_S_names = [fourth_level_names[:][3]] + [fourth_level_names[:][6]]
fourth_level_S_names = [string + '_S' for string in fourth_level_S_names]

with open(path_for_pipeline+r'/result.csv', 'w', encoding='utf-8') as writer:
    writer.write('filename' + 'text' + '\t' + '\t'.join(first_level_names) + '\t' + '\t'.join(second_level_W_names) +
                 '\t' + '\t'.join(second_level_S_names) + '\t' + '\t'.join(third_level_W_names) + '\t' +
                 '\t'.join(third_level_S_names) + '\t' + '\t'.join(fourth_level_W_names) + '\t' +
                 '\t'.join(fourth_level_S_names) + '\t' + 'avg_len_in_chars' + '\t' + 'len_in_chars' + '\t' +
                 'len_in_words' + '\n')

    length = len(file_names) - 1
    for m in range(len(file_names)):

        string_to_write = file_names[m] + '\t' + texts[m] + '\t' + '\t'.join(list(map(str, num_of_1st_class[m]))) + '\t' +\
                         '\t'.join(list(map(str, num_of_2nd_class_W[m]))) + '\t' +\
                         '\t'.join(list(map(str, num_of_2nd_class_S[m]))) + '\t' +\
                         '\t'.join(list(map(str, num_of_3rd_class_W[m]))) + '\t' +\
                         '\t'.join(list(map(str, num_of_3rd_class_S[m]))) + '\t' +\
                         '\t'.join(list(map(str, num_of_4th_class_W[m]))) + '\t' +\
                         '\t'.join(list(map(str, num_of_4th_class_S[m]))) + '\t' + str(avg_chars_lens[m]) +\
                         '\t' + str(total_chars_lens[m]) + '\t' + str(total_words_nums[m])

        if m != length:
            writer.write(string_to_write + '\n')
        else:
            writer.write(string_to_write)

header_for_detailed = 'text' + '\t' + '\t'.join(first_level_names) + '\t' + '\t'.join(second_level_W_names) + '\t' +\
    '\t'.join(second_level_S_names) + '\t' + '\t'.join(third_level_W_names) + '\t' + '\t'.join(third_level_S_names) +\
    '\t' + '\t'.join(fourth_level_W_names) + '\t' + '\t'.join(fourth_level_S_names) + '\n'

os.mkdir(os.path.join(path_for_pipeline, 'detailed_report'))

for j in range(len(texts)):
    with open(path_for_pipeline+r'/detailed_report/{}.csv'.format(str(j)), 'w', encoding='utf-8') as file:
        file.write(header_for_detailed)
        file.write(texts[j] + '\t' + ';'.join(stressed_first_v_all[j]) + '\t' + ';'.join(c_in_the_end_all[j]) + '\t' +
                   ';'.join(c_in_the_beginning_all[j]) + '\t' + ';'.join(two_syl_open_syls_all[j]) + '\t' +
                   ';'.join(three_syl_open_syls_all[j]) + '\t' + ';'.join(one_syl_all[j]) + '\t' +
                   ';'.join(two_syl_all[j]) + '\t' + ';'.join(one_syl_cvc_all[j]) + '\t' +
                   ';'.join(one_syl_begin_cc_all[j]) + '\t' + ';'.join(two_syl_begin_cc_all[j]) + '\t' +
                   ';'.join(two_syl_1th_stressed_all[j]) + '\t' + ';'.join(three_syl_2nd_stressed_all[j]) + '\t' +
                   ';'.join(two_syl_2nd_stressed_all[j]) + '\t' + ';'.join(three_syl_1th_stressed_all[j]) + '\t' +
                   ';'.join(three_syl_cv_pattern_all[j]) + '\t' + ';'.join(four_syl_cv_pattern_all[j]) + '\t' +
                   ';'.join(nom_all[j]) + '\t' + ';'.join(acc_all[j]) + '\t' + ';'.join(dat_all[j]) + '\t' +
                   ';'.join(abl_all[j]) + '\t' + ';'.join(sent_simple_all[j]) + '\t' +
                   ';'.join(sent_two_homogen_all[j]) + '\t' + ';'.join(sent_three_homogen_all[j]) + '\t' +
                   ';'.join(no_predic_all[j]) + '\t' + ';'.join(sent_complic_soch_all[j]) + '\t' +
                   ';'.join(verbs_pers_all[j]) + '\t' + ';'.join(parenth_all[j]) + '\t' +
                   ';'.join(one_syl_end_cc_all[j]) + '\t' + ';'.join(two_syl_middle_cc_all[j]) + '\t' +
                   ';'.join(three_syl_begin_cc_all[j]) + '\t' + ';'.join(three_syl_middle_cc_all[j]) + '\t' +
                   ';'.join(three_syl_end_cc_all[j]) + '\t' + ';'.join(four_syl_cc_on_the_edge_all[j]) + '\t' +
                   ';'.join(five_syl_cv_pattern_all[j]) + '\t' + ';'.join(adv_all[j]) + '\t' +
                   ';'.join(numeral_all[j]) + '\t' + ';'.join(a_pro_all[j]) + '\t' + ';'.join(gen_all[j]) + '\t' +
                   ';'.join(ins_all[j]) + '\t' + ';'.join(coord_conjs_num_all[j]) +
                   ';'.join(sent_complic_depend_all[j]) + '\t' + ';'.join(inverse_all[j]) + '\t' +
                   ';'.join(s_pro_all[j]) + '\t' + ';'.join(three_syl_3rd_stressed_all[j]) + '\t' +
                   ';'.join(three_syl_cc_on_the_edge_all[j]) + '\t' + ';'.join(five_syl_cc_on_the_edge_all[j]) + '\t' +
                   ';'.join(rare_obsol_all[j]) + '\t' + ';'.join(foreign_all[j]) + '\t' +
                   ';'.join(alt_conjs_num_all[j]) + '\t' + ';'.join(particip_clause_all[j]))
