import scipy.sparse as sp_sparse
from index_class import Index
from util import extract_lists


def implicit_feedback(index: Index, query: list, ranking: list, N: int) -> list:
    """
    Method to expand the received query by implicit feedback.

    Parameters:
        index (Index): the database index.
        query (list): the original query as a list of words/tokens.
        ranking (list<str>): the applied retrieval information method's result
                             as a list of documents.
        N (int): number of similar words to be returned for each token in the query.

    Return value:
        list: the new query as a list of word/tokens.
    """

    retrieved_vocabulary = index.get_all_words_in_docs(ranking) # V_l

    # matrix with the frequency of each word in each doc
    term_doc_matrix = sp_sparse.lil_matrix((len(retrieved_vocabulary),
                                            len(ranking)))

    # populates the TDM
    for i, word in enumerate(retrieved_vocabulary):
        for j, doc in enumerate(ranking):
            term_doc_matrix[i, j] = index.get_frequency_in_doc(word, doc)

    term_doc_matrix = term_doc_matrix.tocsr()
    transposed_tdm = term_doc_matrix.transpose()

    # matrix correlating the words with each other
    term_term_correlation_matrix = term_doc_matrix.dot(transposed_tdm) # C_l


    c = term_term_correlation_matrix.tolil() # c_(u, v)

    # normalized matrix correlating the words with each other
    normalized_correlation_matrix = sp_sparse.lil_matrix(c.shape) # C_l'

    # populates the normalized correlation matrix
    for u in range(normalized_correlation_matrix.shape[0]):
        for v in range(normalized_correlation_matrix.shape[1]):
            normalized_correlation_matrix[u, v] = c[u, v]/(c[u, u] + c[v, v] - c[u, v])

    def get_expansions(N: int) -> list:
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
                {
                    'word': word,
                    'correlation': normalized_correlation_matrix[
                        retrieved_vocabulary.index(token), v
                    ]
                }
                for v, word in enumerate(retrieved_vocabulary) if word != token
            ] for token in query if token in retrieved_vocabulary
        ]

        return extract_lists([
            [
                e['word'] for e in
                sorted(token, key=lambda e: e['correlation'])
            ][:N] # gets only the topmost N words

            for token in raw # for each token in the query
        ])

    # the expanded query
    return sorted(query + get_expansions(N))
