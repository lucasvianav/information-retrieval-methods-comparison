from index_class import Index
from methods import probabilisticModel, vectorialModel
from util import *

FILTER_STOPWORDS = True
STEM_WORDS = True

database_contents = get_db_content()
index = Index(database_contents, FILTER_STOPWORDS, STEM_WORDS)

queries = get_queries()
truth_set = get_queries_truth_set()

results = [
    {
        'query': query['id'],
        'results': {
            'probabilistic': probabilisticModel(query['query'], index),
            'vectorial': vectorialModel(query['query'], index)
        }
    } for query in queries
]

for result in results:
    # performs the evaluation here
    pass
