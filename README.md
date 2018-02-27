# readability
This is the project dedicated to measure readability of texts written in russian.
Prerequisites:
* Python 3.x (developed on 3.6.2)
* libraries
  * numpy
  * requests
  * tqdm (*not obligatory, but adds some convenience*)
 * [ru-syntax](https://github.com/tiefling-cat/ru-syntax) up and running (*check the link to see how to install it and beware: if you are using Windows, it needs some changes in code to make it work; see the issues of the project*)

## main code
readability_io_api.py - the script that analyzes your text collection in terms of their readability features by requesting [API](https://github.com/ivbeg/readability.io/wiki/API) made by [Ivan Begtin](https://github.com/ivbeg) giving the output in .csv format. The output will appear at the same directory in which the text collection is situated. File type supported: .txt. Encoding supported: utf-8.

readability_io_offline.py - does almost the same job as readability_io_api.py, but is available offline and doesn't provide the output with the explicit audience description.

IB_metrics_readability.py - the script to make readability_io_offline.py work. Upgraded for Python 3 by [Konstantin Druzhkin](https://bitbucket.org/KDruzhkin/).

syllable_segmentation.py - the script that segments words by syllables.

get_tokens_and_sent_segmentation.py - the script that segments a text into sentences and further - sentences to tokens; reworked version of [RusTokenizer](https://github.com/elmiram/RusTokenizer).

final_pipeline.py - the beta-version of our pipeline that analyzes your text collection in terms of their readability features.
