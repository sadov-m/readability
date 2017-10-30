# readability
This is the project dedicated to measure readability of texts written in russian.
Prerequisites:
* Python 3.x (developed on 3.6.2)
* libraries
  * numpy
  * requests
  * tqdm (*not obligatory, but adds some convenience*)
 * [ru-syntax](https://github.com/tiefling-cat/ru-syntax) up and running (*check the link to see how to install it and beware: if you are using Windows, it needs some changes in code to make it work; see the issues of the project*)

## code
readability_io_api.py - the script that analyzes your text collection in terms of their readability levels by requesting [API](https://github.com/ivbeg/readability.io/wiki/API) made by [Ivan Begtin](https://github.com/ivbeg) giving the output in .csv format. The output will appear at the same directory in which the text collection is situated. File type supported: .txt. Encoding supported: utf-8.

readability_io_offline.py - does almost the same job as readability_io_api.py, but is offline and doesn't provide the output with the explicit audience description.

IB_metrics_readability.py - the script to make readability_io_offline.py work. Upgraded for Python 3 by [Konstantin Druzhkin](https://bitbucket.org/KDruzhkin/).

syllable_segmentation.py - the script that segments words by syllables.

get_tokens_and_sent_segmentation.py - the script that segments a text into sentences and further - sentences to tokens; reworked version of [RusTokenizer](https://github.com/elmiram/RusTokenizer).

final_pipeline.py - the beta-version of script that analyzes your text collection in terms of their readability levels (*not finished and, consequently, not recommended to use now*).

## files
trial_output - a demonstration of how readability_io_api.py works.

output_for_readability_team_only - the output for files gathered by project's team (*not recommended to use them if you are not the member of this project's team*).

syllable_segmentation_testing - to demonstrate how syllable_segmentation.py works.

accent_lstm - the script that segments words by syllables. Taken from [here](https://github.com/MashaPo/accent_lstm).

lex dicts and lists - required for .py files.
