from readability_io_api import extracting_texts_paths
import get_tokens_and_sent_segmentation
from pymystem3 import Mystem
from accent_lstm import text_accentAPI
import syllable_segmentation
from subprocess import call
import re

mystem = Mystem()
mystem.start()

path_for_pipeline = input('type in the path to a folder with texts to analyze: ')
# C:\Users\Mike\Desktop\test

paths = extracting_texts_paths(path_for_pipeline)


# transform a word to a mask of type CVC... where C is a consonant and V is a vowel
def get_word_mask(word):
    mask = []

    for sym in word:
        mask.append('V' if sym in text_accentAPI.VOWELS else 'C')

    return mask


def get_accent_syl_id(accentuated_char_id, list_of_syls_lengths):
    last = len(list_of_syls_lengths) - 1

    for i in range(len(list_of_syls_lengths)):
        if i == 0:
            if 0 <= accentuated_char_id < list_of_syls_lengths[0]:
                return 0
        elif i == last:
            if sum(list_of_syls_lengths[:i]) <= accentuated_char_id:
                return last
        else:
            if sum(list_of_syls_lengths[:i]) <= accentuated_char_id < sum(list_of_syls_lengths[:i+1]):
                return i


for path in paths:
    with open(path, encoding='utf-8') as file_opener:
        text = file_opener.read()

    # sentense splitting + tokenization
    tokenizer = get_tokens_and_sent_segmentation.Text(fname=r'', text_in_string=text, path_input=False)
    tokenizer.process()
    tokenized_sents = [[token for token in sent if token.isalnum()] for sent in tokenizer.get_sentence_segmentation()]

    # plain formal features: see variables names
    number_of_sents = len(tokenized_sents)
    number_of_words = sum([len(sent) for sent in tokenized_sents])
    list_of_words_len = [len(word) for sent in tokenized_sents for word in sent]
    total_chars_len = sum(list_of_words_len)
    avg_chars_len = total_chars_len/len(list_of_words_len)

    # accent_lstm
    accentuated = text_accentAPI.main([' '.join(sent) for sent in tokenized_sents[:2]])
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

    two_syl = 0  # Слова из двух слогов
    two_syl_1th_stressed = 0  # Ударение на первый слог в двусложных словах
    two_syl_2nd_stressed = 0  # Ударение на второй слог в двусложных словах
    two_syl_begin_cc = 0  # Двусложные слова с сочетанием согласных в начале слова
    two_syl_open_syls = 0  # Двусложные слова с открытым слогом
    two_syl_middle_cc = 0  # Двусложные слова с сочетанием согласных в середине слова

    three_syl_1th_stressed = 0  # Ударение на первый слог в трехсложных словах
    three_syl_2nd_stressed = 0  # Ударение на второй слог в трехсложных словах
    three_syl_3rd_stressed = 0  # Ударение на третий слог в трехсложных словах
    three_syl_open_syls = 0  # Трехсложные слова с открытым слогом
    three_syl_begin_cc = 0  # Трехсложные слова с сочетанием согласных в начале слова
    three_syl_middle_cc = 0  # Трехсложные слова с сочетанием согласных в середине слова
    three_syl_end_cc = 0  # Трехсложные слова с сочетанием согласных в конце слова
    three_syl_cv_pattern = 0  # Слова из трех слогов (чередование гласных и согласных)
    three_syl_cc_on_the_edge = 0  # Слова из трех слогов (сочленение согласных букв)

    four_syl_cv_pattern = 0  # Слова из четырех слогов (чередование гласных и согласных)
    four_syl_cc_on_the_edge = 0  # Слова из четырех слогов (сочленение согласных букв)

    five_syl_cv_pattern = 0  # Слова из пяти слогов (чередование гласных и согласных)
    five_syl_cc_on_the_edge = 0  # Слова из пяти слогов (сочленение согласных букв

    stressed_first_v = 0  # Ударные гласные в начале слова
    c_in_the_end = 0  # Согласные в конце слова
    c_in_the_beginning = 0  # Согласные в начале слова

    # segmentation itself and syllable features retrieval
    for ind, word in enumerate(words_only[:10]):

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
            print(words_only[ind], accent_positions[ind])
            print(result, result[accent_syl_id], accent_syl_id, accent_id_in_syl)
            print()
            # here comes the features!
            whole_mask = ''.join(result)

            if accent_syl_id == 0:
                stressed_first_v += 1
            elif whole_mask[0] == 'C':
                c_in_the_beginning += 1

            if whole_mask[-1] == 'C':
                c_in_the_end += 1

            if num_of_syls == 1:
                one_syl += 1

                for syl in result:
                    if syl[:2] == 'CC':
                        one_syl_begin_cc += 1
                    if syl[-2:] == 'CC':
                        one_syl_end_cc += 1
                    if syl == 'CVC':
                        one_syl_cvc += 1

            if num_of_syls == 2:
                two_syl += 1

                if accent_syl_id == 0:
                    two_syl_1th_stressed += 1
                else:
                    two_syl_2nd_stressed += 1

                if result[0][:2] == 'CC':
                    two_syl_begin_cc += 1

                if result[0][-1]+result[1][-1] == 'VV':
                    two_syl_open_syls += 1

                if 'CC' in whole_mask[1:-1]:
                    two_syl_middle_cc += 1

            if num_of_syls == 3:

                if accent_syl_id == 0:
                    three_syl_1th_stressed += 1
                elif accent_syl_id == 1:
                    three_syl_2nd_stressed += 1
                else:
                    three_syl_3rd_stressed += 1

                if result[0][-1]+result[1][-1]+result[2][-1] == 'VVV':
                    three_syl_open_syls += 1
                else:
                    if result[2][-2:] == 'CC':
                        three_syl_end_cc += 1

                prev_char = whole_mask[0]
                for j, char in enumerate(whole_mask[1:]):

                    if char == prev_char:
                        break
                    else:
                        prev_char = char

                else:
                    three_syl_cv_pattern += 1

                if result[0][:2] == 'CC':
                    three_syl_begin_cc += 1

                if 'CC' in whole_mask[1:-1]:
                    three_syl_middle_cc += 1

                if result[0][-1]+result[1][0] == 'CC' or result[1][-1]+result[2][0] == 'CC':
                    three_syl_cc_on_the_edge += 1

            if num_of_syls == 4:
                prev_char = whole_mask[0]
                for j, char in enumerate(whole_mask[1:]):

                    if char == prev_char:
                        break
                    else:
                        prev_char = char

                else:
                    four_syl_cv_pattern += 1

                if result[0][-1]+result[1][0] == 'CC' or result[1][-1]+result[2][0] == 'CC' or\
                                        result[2][-1]+result[3][0] == 'CC':
                    four_syl_cc_on_the_edge += 1

            if num_of_syls == 5:
                prev_char = whole_mask[0]
                for j, char in enumerate(whole_mask[1:]):

                    if char == prev_char:
                        break
                    else:
                        prev_char = char

                else:
                    five_syl_cv_pattern += 1

                if result[0][-1] + result[1][0] == 'CC' or result[1][-1] + result[2][0] == 'CC' or \
                                        result[2][-1] + result[3][0] == 'CC' or result[3][-1] + result[4][0] == 'CC':
                    five_syl_cc_on_the_edge += 1

    # mystem analyzer
    analyzed_sents = []

    # lex features
    parenth = 0  # Вводные слова
    rare_obsol = 0  # Редко употребляемые/устаревшие слова
    protiv_conj = 0  # Противительные союзы?
    soch_conj = 0  # Сочинительные союзы?
    foreign = 0  # Иностранные слова?

    # morph features
    verbs_pers = 0  # Глаголы в личной форме
    nom = 0  # слова в номинативе
    gen = 0  # слова в генитиве
    acc = 0  # слова в аккузативе
    dat = 0  # слова в дативе
    ins = 0  # слова в творительном
    abl = 0  # слова в предложном
    numeral = 0  #
    a_pro = 0  #
    s_pro = 0  #
    adv = 0  #

    # lex and morph features retrieval
    mystem_result = mystem.analyze('Я люблю ходить гулять, ты любишь тоже, а они любят?')

    for word_gr in mystem_result:

        try:
            # word_gr['analysis']:
            gr = re.findall('\w+', word_gr['analysis'][0]['gr'])

            if 'редк' in gr or 'устар' in gr:
                rare_obsol += 1

            if 'вводн' in gr:
                parenth += 1
            elif 'V' in gr and 'л' in gr:
                verbs_pers += 1
            elif 'ADV' in gr:
                adv += 1

            if 'им' in gr:
                nom += 1
            elif 'род' in gr:
                gen += 1
            elif 'вин' in gr:
                acc += 1
            elif 'дат' in gr:
                dat += 1
            elif 'твор' in gr:
                ins += 1
            elif 'пр' in gr:
                abl += 1

            if 'NUM' in gr:
                numeral += 1
            elif 'APRO' in gr:
                a_pro += 1
            elif 'SPRO' in gr:
                s_pro += 1
        except KeyError:
            pass

    # ru-syntax

    line = r'python C:\Users\Mike\PycharmProjects\ru-syntax\ru-syntax.py {}'.format(path)
    call(line)

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

    for sent in sents:
        predic = False
        root_id = 0
        predic_id = 0
        not_simple = False
        soch = 0

        for i, elem in enumerate(sent):
            if elem[7] == 'предик':
                predic = True
                predic_id = int(elem[0])
            elif elem[7] == 'ROOT':
                root_id = int(elem[0])
            elif elem[7] == 'опред' or 'PUNC':
                pass
            else:
                not_simple = True

            if elem[7] == 'сент-соч':
                sent_complic_soch += 1
            elif elem[7] == 'подч-союзн' or elem[7] == 'изъясн' or elem[7] == 'релят':
                sent_complic_depend += 1
            elif elem[7] == 'сочин':
                soch += 1

            if elem[3] == 'NID':
                foreign += 1
            elif elem[3] == 'PARTCP':
                particip_clause += 1

        if root_id < predic_id:
            inverse += 1

        if len(sent) < 5 and predic and not not_simple:
            sent_simple += 1

        no_predic += 1 if not predic else 0

        if soch == 1:
            sent_two_homogen += 1
        elif soch == 2:
            sent_three_homogen += 1

    #print(number_of_sents, number_of_words, len(list_of_words_len), total_chars_len, avg_chars_len)
