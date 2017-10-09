# readability
This is the project dedicated to readability measurement for texts written in russian.
Prerequisites: python 3 + libraries (numpy, requests, tqdm)

## code
readability_io_api.py - the script that analyzes your text collection in terms of their readability by requesting [API](https://github.com/ivbeg/readability.io/wiki/API) made by [Ivan Begtin](https://github.com/ivbeg) giving the output in .csv format. The output will appear at the same directory in which the text collection is situated. File type supported: .txt. Encoding supported: utf-8.

readability_io_offline.py - does almost the same job as readability_io_api.py, but is offline and doesn't provide the output with the explicit audience description.

IB_metrics_readability.py - the script to make readability_io_offline.py work. Upgraded for Python-3 by [Konstantin Druzhkin](https://bitbucket.org/KDruzhkin/).

syllable_segmentation.py - the script that segments words by syllables.

## files
test_collection.7z - .txt files to give readability_io_api.py a try

output_for_test_collection.csv - the output for files in test_collection.7z

output_for_readability_team_only - the output for files gathered by project's team (not recommended to use them if you are not the member of this project's team)

syllable_segmentation_testing - to demonstrate how syllable_segmentation.py works
