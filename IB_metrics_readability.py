"""
This code by Ivan Begtin is taken from
https://github.com/infoculture/plainrussian/blob/master/textmetric/metric.py

Upgraded for Python-3 by Konstantin Druzhkin.
"""
__author__ = 'ibegtin'
from math import sqrt
import csv

from numpy import mean, arange

# Russian sounds and characters
RU_CONSONANTS_LOW = [u'к', u'п', u'с', u'т', u'ф', u'х', u'ц', u'ч', u'ш', u'щ']
RU_CONSONANTS_HIGH = [u'б', u'в', u'г', u'д', u'ж', u'з']
RU_CONSONANTS_SONOR = [u'л', u'м', u'н', u'р']
RU_CONSONANTS_YET = [u'й']
RU_CONSONANTS = RU_CONSONANTS_HIGH + RU_CONSONANTS_LOW + RU_CONSONANTS_SONOR + RU_CONSONANTS_YET
RU_VOWELS = [u'а', u'е', u'и', u'у', u'о', u'я', u'ё', u'э', u'ю', u'я', u'ы']
RU_MARKS = [u'ь', u'ъ']
SENTENCE_SPLITTERS = [u'.', u'?', u'!']
RU_LETTERS = RU_CONSONANTS + RU_MARKS + RU_VOWELS
SPACES = [u' ', u'\t']

######################################################################
# Адаптированные для русского языка формулы                          #
######################################################################

# Flesh Kinkaid Grade константы. Подробнее http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
FLG_X_GRADE = 0.318
FLG_Y_GRADE = 14.2
FLG_Z_GRADE = 30.5

def calc_Flesh_Kincaid_Grade_rus_flex(n_syllabes, n_words, n_sent):
    """Метрика Flesh Kincaid Grade для русского языка с константными параметрами"""
    if n_words == 0 or n_sent == 0: return 0
    n = FLG_X_GRADE * (float(n_words) / n_sent) + FLG_Y_GRADE * (float(n_syllabes) / n_words) - FLG_Z_GRADE
    return n if n > 0 else 0

# Coleman Liau константы. Подробнее http://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
CLI_X_GRADE = 0.055
CLI_Y_GRADE = 0.35
CLI_Z_GRADE = 20.33

def calc_Coleman_Liau_index(n_letters, n_words, n_sent):
    """ Метрика Coleman Liau для русского языка с константными параметрами """
    if n_words == 0: return 0
    n = CLI_X_GRADE * (n_letters * (100.0 / n_words)) - CLI_Y_GRADE * (n_sent * (100.0 / n_words)) - CLI_Z_GRADE
    return n if n > 0 else 0

# Константы SMOG Index http://en.wikipedia.org/wiki/SMOG
SMOG_X_GRADE = 1.1
SMOG_Y_GRADE = 64.6
SMOG_Z_GRADE = 0.05

def calc_SMOG_index(n_psyl, n_sent):
    """Метрика SMOG для русского языка с константными параметрами"""
    n = SMOG_X_GRADE * sqrt((float(SMOG_Y_GRADE) / n_sent) * n_psyl) + SMOG_Z_GRADE
    return n if n > 0 else 0

DC_X_GRADE = 0.552
DC_Y_GRADE = 0.273

def calc_Dale_Chale_index(n_psyl, n_words, n_sent):
    """Метрика Dale Chale для русского языка с константными параметрами"""
    n = DC_X_GRADE * (100.0 * n_psyl / n_words) + DC_Y_GRADE * (float(n_words) / n_sent)
    return n if n > 0 else 0

ARI_X_GRADE = 6.26
ARI_Y_GRADE = 0.2805
ARI_Z_GRADE = 31.04

def calc_ARI_index(n_letters, n_words, n_sent):
    """ Метрика Automated Readability Index (ARI) для русского языка с константными параметрами """
    if n_words == 0 or n_sent == 0: return 0
    n = ARI_X_GRADE * (float(n_letters) / n_words) + ARI_Y_GRADE * (float(n_words) / n_sent) - ARI_Z_GRADE
    return n if n > 0 else 0

# Number of syllabes for long words
COMPLEX_SYL_FACTOR = 4

def calc_readability_metrics(text):
    sentences = 0
    chars = 0
    spaces = 0
    letters = 0
    syllabes = 0
    words = 0
    complex_words = 0
    simple_words = 0
    wsyllabes = {}

    wordStart = False
    for l in text.splitlines():
        chars += len(l)
        for ch in l:
            if ch in SENTENCE_SPLITTERS:
                sentences += 1
            if ch in SPACES:
                spaces += 1

        for w in l.split():
            has_syl = False
            wsyl = 0
            for ch in w:
                if ch in RU_LETTERS:
                    letters += 1
                if ch in RU_VOWELS:
                    syllabes += 1
                    has_syl = True
                    wsyl += 1
            if wsyl > COMPLEX_SYL_FACTOR:
                complex_words += 1
            elif wsyl < COMPLEX_SYL_FACTOR+1 and wsyl > 0:
                simple_words += 1
            if has_syl:
                words += 1
                v = wsyllabes.get(str(wsyl), 0)
                wsyllabes[str(wsyl)] = v + 1

    # KD: avoid division by zero
    if sentences == 0: sentences = 1
    if words == 0: words = 1

    metrics = {
               'FK': calc_Flesh_Kincaid_Grade_rus_flex(syllabes, words, sentences),
               'CL' : calc_Coleman_Liau_index(letters, words, sentences),
               'DC' : calc_Dale_Chale_index(complex_words, words, sentences),
               'SMOG' : calc_SMOG_index(complex_words, sentences),
               'ARI' : calc_ARI_index(letters, words, sentences),
                  '# of chars': chars,
                  '# of spaces': spaces,
                  '# of letters': letters,
                '# of words':words,
    '# of sentences':sentences,
    '# of complex words': complex_words,
    '# of simple words': simple_words,
    'average # of words per sentence': words/sentences,
    'average # of syllables per sentence': syllabes/words,
    '% of complex words': complex_words/words*100
    }
    del text
    return metrics

######################################################################
# Внешний код будет вызывать эту функцию                             #
######################################################################

def calc_metrics_with_average (text):
    """
    КД: обёртка вокруг `calc_readability_metrics`, которая вычисляет среднее значение всех метрик.
    """
    metrics = calc_readability_metrics (text)
    metrics['avg_grade_index'] = mean (list (metrics.values()))
    return metrics