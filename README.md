# readability
This is the project dedicated to readability measurement for texts written in russian.
Prerequisites: python 3

## code
readability_io_api.py - the script that analyzes your text collection in terms of their readability by requesting API at https://github.com/ivbeg/readability.io/wiki/API giving the output in .csv format. The output will appear at the same directory in which the text collection is situated. File type supported: .txt. Encoding supported: utf-8.

syllable_classificaiton.py - a demo version of the script that segments words by syllables. Beware: due to the fact that it is the demo version, certain pieces of code inside might cause confusion.

## files
test_collection.7z - .txt files to give readability_io_api.py a try

output_for_test_collection.csv - the output for files in test_collection.7z

output_for_1.csv, output_for_3.csv, output_for_4.csv, output_for_biblioschool.csv, output_for_general.csv - the output for files gathered by project's team (not recommended to use them if you are not the member of this project's team)

сложные-слова.txt - list of words for testing syllable_classificaiton.py
