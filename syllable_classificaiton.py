"""# coding=utf-8

vowels = set(u'аеёиоуыэюя')
sign_chars = set(u'ъь')
pattern_str = u"(c*[ьъ]?vc+[ьъ](?=v))|(c*[ьъ]?v(?=v|cv))|(c*[ьъ]?vc[ъь]?(?=cv|ccv))|(c*[ьъ]?v[cьъ]*(?=$))"
pattern = re.compile(pattern_str)
tokenizing = re.compile('\w+')


def get_syllables(word):
    mask = ''.join(['v' if c in vowels else c if c in sign_chars else 'c' for c in word.lower()])
    return [word[m.start():m.end()] for m in pattern.finditer(mask)]

string = 'капремонт'

for word in tokenizing.findall(string.lower()):

    if len(word) != len(''.join(get_syllables(word))):
        print(word)
        print(print('-'.join(get_syllables(word))))"""

import re
import morfessor

io = morfessor.MorfessorIO()
prep_finder = re.compile('\w+')
prepositions = """без-/бес- · в-/во- · вз-/взо-/вс- · вне- · внутри- · воз-/возо-/вос- · все- · вы- · до- · за- · из-/изо-/ис- · испод- · к- · кое-/кой- · меж-/междо-/между- · на- · над-/надо- · наи- · не- · небез-/небес- · недо- · ни- · низ-/низо-/нис- · о- · об-/обо- · обез-/обес- · около- · от-/ото- · па- · пере- · по- · под-/подo- · поза- · после- · пра- · пре- · пред-/предо- · преди- · при- · про- · противо- · раз-/разо-/рас- (роз-/рос-) · с-/со- · сверх- · среди- · су- · сыз- · тре- · у- · чрез-/через-/черес-

а- · анти- · архи- · би- · вице- · гипер- · де-/дез- · дис- · им- · интер- · ир- · квази- · контр- · макро- · микро- · обер- · пост- · пре- · прото- · псевдо- · ре- · суб- · супер- · транс- · ультра- · экзо- · экс- · экстра-

агит- · глав- · гор- · гос- · деп- · дет- · диа- · здрав- · ино- · кол- · ком- · лик- · маг- · мат- · маш- · мин- · мол- · об-/обл- · окруж- · орг- · парт- · полит- · потреб- · прод- · пром- · проп- · рай- · рег- · ред- · род- · рос- · сек- · сель- · со- · сов- · сот- · соц- · студ- · тер- · фед- · фин- · хоз- · хос-"""

# print(prep_finder.findall(prepositions))

vowel = set('аеёиоуыэюя')
voiced = set('бвгдзлмнрхц')
deaf = set('кпстф')
brief = 'й'
other = set('ьъ')
cons = set('бвгджзйклмнпрстфхцчшщ')
sizz = set('жшчщ')


def check_vowel_presence(string):
    presence = False
    for letter in string:
        if letter in vowel:
            presence = True
            break

    return presence


def get_syllables(string):
    result = []
    current_syllable = ''
    last_ind = len(string) - 1

    # Проверка на признаки конца слогов
    for ind, letter in enumerate(string):
        current_syllable += letter

        # если буква - "й", если она не первая, не последняя, и это - не последний слог
        if ind != 0 and ind != last_ind and letter == brief and check_vowel_presence(string[ind+1:]):
            result.append(current_syllable)
            current_syllable = ''

        # если эта буква - не последняя, текущая и следующая - гласные
        if ind != last_ind and letter in vowel and string[ind+1] in vowel:
            result.append(current_syllable)
            current_syllable = ''

        # если эта буква - не последняя и не предпоследняя, текущая - гласная, следующая - согласная, а после неё - гласная
        if ind < last_ind - 1 and letter in vowel and string[ind+1] in cons and string[ind+2] in vowel:
            result.append(current_syllable)
            current_syllable = ''

        # если эта буква - не последняя и не предпоследняя, текущая - гласная, следующая - глухая согласная,
        # а после неё - согласная, и это - не последний слог
        if ind < last_ind - 1 and letter in vowel and string[ind+1] in deaf and string[ind+2] in cons and check_vowel_presence(string[ind+1:]):
            result.append(current_syllable)
            current_syllable = ''

        # если это - не первая и не последняя буква, если текущая - звонкая или шипящая согласная, перед ней - гласная,
        # следующая - не гласная и не "ь"/"ъ", и это - не последний слог
        if ind != 0 and ind != last_ind and letter in voiced.union(sizz) and string[ind-1] in vowel and string[ind+1] not in vowel.union(other) and check_vowel_presence(string[ind+1:]):
            result.append(current_syllable)
            current_syllable = ''

        # если текущая - "ь"/"ъ" и при этом не последняя, если следующая - гласная или первый слог
        if ind != last_ind and letter in other and (string[ind+1] not in vowel and result == []):
            result.append(current_syllable)
            current_syllable = ''

    result.append(current_syllable)

    return '-'.join(result)

with open('сложные-слова.txt', 'r', encoding='utf-8') as opener:
    word_list = opener.read().split('\n')

model_types = io.read_binary_model_file(r'C:\Users\Ольга\PycharmProjects\DSM_morphology\morfessor\types')

for word in word_list:
    syllabs = get_syllables(word.lower())
    print(syllabs)