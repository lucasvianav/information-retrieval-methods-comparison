
from evaluation import Evaluation
from index_class import Index
from methods import probabilisticModel, vectorialModel
from query_expansion import implicit_feedback
from util import *

database_contents = get_db_content()
queries = get_queries()
truth_sets = get_queries_truth_set()

def getMetrics(filter_stopwords: bool, stem_words: bool, expand_queries: bool):
    index = Index(database_contents, filter_stopwords, stem_words)

    # interpol: first list is precision and second is recall
    # dcg: first item is DCG and second is IDCG
    metrics = {
        'probab': {
            'precision': 0.,
            'recall': 0.,
            'map': 0.,
            'interpol': ([], []),
            'dcg': ([], [])
        },

        'vectorial': {
            'precision': 0.,
            'recall': 0.,
            'map': 0.,
            'interpol': ([], []),
            'dcg': ([], [])
        }
    }

    # loops through all queries
    for query in queries:
        truth_set = truth_sets[query['id']]
        query_vector = parse_text(query['query'], filter_stopwords, stem_words)

        # the returned rankings
        probabilistic = probabilisticModel(query_vector, index)
        vectorial = vectorialModel(query_vector, index)

        # exands queries if necessary
        if expand_queries:
            new_query = implicit_feedback(index, query_vector, probabilistic, 2)
            probabilistic = probabilisticModel(new_query, index)

            new_query = implicit_feedback(index, query_vector, vectorial, 2)
            vectorial = vectorialModel(new_query, index)

        evalProb = Evaluation(probabilistic, truth_set)
        evalVect = Evaluation(vectorial, truth_set)

        # saves this query's probabilistic evaluation metrics
        metrics['probab']['precision'] += evalProb.getPrecision()
        metrics['probab']['recall']    += evalProb.getRecall()
        metrics['probab']['map']       += evalProb.getMAP()

        dcg = metrics['probab']['dcg'][0]
        interpol = metrics['probab']['interpol'][0]

        metrics['probab']['dcg'] = (
            get_sum(dcg, evalProb.getDCG()[0]),
            get_sum(dcg, evalProb.getDCG()[1])
        )

        metrics['probab']['interpol'] = (
            get_sum(interpol, evalProb.getInterpol()[0]),
            evalProb.getInterpol()[1]
        )


        # saves this query's vectorial evaluation metrics
        metrics['vectorial']['precision'] += evalVect.getPrecision()
        metrics['vectorial']['recall']    += evalVect.getRecall()
        metrics['vectorial']['map']       += evalVect.getMAP()

        dcg = metrics['vectorial']['dcg'][0]
        interpol = metrics['vectorial']['interpol'][0]

        metrics['vectorial']['dcg'] = (
            get_sum(dcg, evalVect.getDCG()[0]),
            get_sum(dcg, evalVect.getDCG()[1])
        )

        metrics['vectorial']['interpol'] = (
            get_sum(interpol, evalVect.getInterpol()[0]),
            evalVect.getInterpol()[1]
        )


    no_queries = len(queries)

    # calculates the precisions' mean values
    probabilistic_precisions = get_division(metrics['probab']['interpol'],
                                            number=no_queries)
    vectorial_precisions = get_division(metrics['vectorial']['interpol'],
                                        number=no_queries)

    # calculates the probabilistic model's DCG and IDCG
    probabilistic_dcg = get_division(metrics['probab']['dcg'][0],
                                     number=no_queries)
    probabilistic_idcg = get_division(metrics['probab']['dcg'][1],
                                      number=no_queries)

    # calculates the vectorial model's DCG and IDCG
    vectorial_dcg = get_division(metrics['vectorial']['dcg'][0],
                                     number=no_queries)
    vectorial_idcg = get_division(metrics['vectorial']['dcg'][1],
                                      number=no_queries)

    return {
        'probab': {
            'precision': metrics['probab']['precision']/no_queries,
            'recall': metrics['probab']['recall']/no_queries,
            'map': metrics['probab']['map']/no_queries,
            'interpol': {
                'precision': probabilistic_precisions,
                'recall': metrics['probab']['interpol'][1]
            },
            'ndcg': get_division(probabilistic_dcg, list2=probabilistic_idcg)
        },

        'vectorial': {
            'precision': metrics['vectorial']['precision']/no_queries,
            'recall': metrics['vectorial']['recall']/no_queries,
            'map': metrics['vectorial']['map']/no_queries,
            'interpol': {
                'precision': vectorial_precisions,
                'recall': metrics['probab']['interpol'][1]
            },
            'ndcg': get_division(vectorial_dcg, list2=vectorial_idcg)
        }
    }

