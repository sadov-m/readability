import re

# prepositions were supposed to be used for improving the segmentation results, but currently they are not used
prep_finder = re.compile('\w+')
prepositions = """без-/бес- · в-/во- · вз-/взо-/вс- · вне- · внутри- · воз-/возо-/вос- · все- · вы- · до- · за- · из-/изо-/ис- · испод- · к- · кое-/кой- · меж-/междо-/между- · на- · над-/надо- · наи- · не- · небез-/небес- · недо- · ни- · низ-/низо-/нис- · о- · об-/обо- · обез-/обес- · около- · от-/ото- · па- · пере- · по- · под-/подo- · поза- · после- · пра- · пре- · пред-/предо- · преди- · при- · про- · противо- · раз-/разо-/рас- (роз-/рос-) · с-/со- · сверх- · среди- · су- · сыз- · тре- · у- · чрез-/через-/черес-

а- · анти- · архи- · би- · вице- · гипер- · де-/дез- · дис- · им- · интер- · ир- · квази- · контр- · макро- · микро- · обер- · пост- · пре- · прото- · псевдо- · ре- · суб- · супер- · транс- · ультра- · экзо- · экс- · экстра-

агит- · глав- · гор- · гос- · деп- · дет- · диа- · здрав- · ино- · кол- · ком- · лик- · маг- · мат- · маш- · мин- · мол- · об-/обл- · окруж- · орг- · парт- · полит- · потреб- · прод- · пром- · проп- · рай- · рег- · ред- · род- · рос- · сек- · сель- · со- · сов- · сот- · соц- · студ- · тер- · фед- · фин- · хоз- · хос-"""

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
        if ind < last_ind - 1 and letter in vowel and string[ind+1] in deaf and string[ind+2] in cons and \
                check_vowel_presence(string[ind+1:]):
            result.append(current_syllable)
            current_syllable = ''

        # если это - не первая и не последняя буква, если текущая - звонкая или шипящая согласная, перед ней - гласная,
        # следующая - не гласная и не "ь"/"ъ", и это - не последний слог
        if ind != 0 and ind != last_ind and letter in voiced.union(sizz) and string[ind-1] in vowel and \
                string[ind+1] not in vowel.union(other) and check_vowel_presence(string[ind+1:]):
            result.append(current_syllable)
            current_syllable = ''

        # если текущая - "ь"/"ъ" и при этом не последняя, если следующая - гласная или первый слог
        # или если следуют от 1 до 3 согласных, за которыми - гласная
        # ъ/ь не должны быть второй буквой (въехать), перед ъ/ь не должно быть двух согласных подряд
        try:        
            if ind != last_ind and ind != 1 and string[ind-2] in vowel and letter in other and ((string[ind+1] in vowel) or result == [] or (string[ind+1] in cons and string[ind+2] in vowel) or (string[ind+1] in cons and string[ind+2] in cons and string[ind+3] in vowel) or (string[ind+1] in cons and string[ind+2] in cons and string[ind+3] in cons and string[ind+4] in vowel)):
                result.append(current_syllable)
                current_syllable = ''
        except IndexError:
            pass
        
    result.append(current_syllable)

    return '-'.join(result)


if __name__ == "__main__":

    with open('ьъ.txt', 'r', encoding='utf-8') as opener:
        word_list = opener.read().split('\n')

    for word in word_list:
        print(word)
        syllabs = get_syllables(word.lower())
        print(syllabs)
