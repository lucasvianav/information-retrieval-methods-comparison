from index_class import Index
import numpy as np
from util import extract_lists

def implicit_feedback(index: Index, query: list, ranking: dict):
    """
    Method to expand the received query by implicit feedback.
    
    Parameters:
        index (Index): the database index.
        query (list): the original query as a list of words/tokens.
        ranking (dict): the applied retrieval information method's result as a dict containing the doc names as keys and their similarity to the query as values (float).
    
    Return value:
        list: the new query as a list of word/tokens.
    """
    
    retrieved_docs = [ doc for doc, _ in sorted(ranking.items(), key=lambda doc: doc[1]) ] # D_l
    retrieved_vocabulary = index.get_all_words_in_docs(retrieved_docs) # V_l

    # matrix with the frequency of each word in each doc    
    term_doc_matrix = np.array([  # M_l
        [ index.get_frequency_in_doc(word, doc) for word in retrieved_vocabulary ] for doc in retrieved_docs 
    ])

    # matrix correlating the words with each other
    term_term_correlation_matrix = term_doc_matrix * term_doc_matrix.T # C_l


    # normalized matrix correlating the words with each other
    normalized_correlation_matrix = np.zeros(term_term_correlation_matrix.shape) # C_l'
    c = term_term_correlation_matrix # c_(u, v)
    for u in range(normalized_correlation_matrix.shape[0]):
        for v in range(normalized_correlation_matrix.shape[1]):
            normalized_correlation_matrix[u, v] = c[u, v]/(c[u, u] + c[v, v] - c[u, v])

    def get_expansions(N: int):
        """
        This method returns a given number of similar words to each token in the query.
        
        Parameters:
            N (int): number of similar words to be returned for each token in the query.
        
        Return value:
            list: contains a list of similar words to each token in the query.
        """
        
        # gets all correlations for each token in the query
        raw = [ 
            [ 
                { 'word': word, 'correlation': normalized_correlation_matrix[retrieved_vocabulary.index(q), v] } 
                for v, word in enumerate(retrieved_vocabulary) if word != q 
            ] for q in query 
        ]

        return extract_lists([ 
            list(map(
                lambda e: e['word'], # selects only the actual word
                sorted(l, key=lambda e: e['correlation']) # sorts by correlation
            ))[:N] # gets only the topmost N words

            for l in raw # for each token in the query
        ])

    # the expanded query
    return sorted(query + get_expansions(2))