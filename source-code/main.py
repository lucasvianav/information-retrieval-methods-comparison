from nltk import download, word_tokenize
from index_class import Index
from util import *
from methods import probabilisticModel
from functools import reduce

# FILTER_STOPWORDS = False
# STEM_WORDS = False

FILTER_STOPWORDS = True
STEM_WORDS = True

database_contents = get_db_content()
index = Index(database_contents, FILTER_STOPWORDS, STEM_WORDS)

query = parse_text('Clashes between the Gurjars and Meenas', FILTER_STOPWORDS, STEM_WORDS)

print(probabilisticModel(query, index))
