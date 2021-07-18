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

map_probabilistic = 0.0
map_vectorial = 0.0
for result in results:
    # performs the evaluation here
    eval_prob = Evaluation(
        result['results']['probabilistic'],
        truth_set[result['query']])
    eval_vec = Evaluation(
        result['results']['vectorial'],
        truth_set[result['query']])
    map_probabilistic = eval_prob.get_MAP_i()
    map_vectorial = eval_vec.get_MAP_i()

map_probabilistic = map_probabilistic/len(results)
map_vectorial = map_vectorial/len(results)
