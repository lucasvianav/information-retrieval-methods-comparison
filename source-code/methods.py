import math
from functools import reduce

import scipy.sparse as sp_sparse
from index_class import Index
from util import get_intersection


def probabilisticModel(query: list, index: Index, relevant_docs = []) -> list:
    """
    Applies the probabilistic model for information retrieval to calculate the similarity between
    the query and the indexed documents (considering the passed docs as relevant).

    Parameters:
        query (list): list of query's words/tokens.
        index (Index): database index (instance of the Index class).
        relevante_docs (list | []): names of relevant docs for that query (default empty list).

    Return value:
        list: list containing the doc names sorted by similarity.
    """

    # key --> word/token
    # value --> names of docs containing that word
    containing_word_docs = { word: index.get_doc_list(word) for word in query }

    # key --> word/token
    # value --> names of relevant docs containing that word
    relevant_containing_word_docs = {
        word: get_intersection(relevant_docs, docs) for word,
        docs in containing_word_docs.items()
    }

    # n_ == number of
    n_docs = index.get_n_docs() # N
    n_containing_word = { word: index.get_n_docs_containing(word) for word in query } # n_i
    n_relevant = len(relevant_docs) # R
    n_relevant_containing_word = { # r_i
        word: len(docs) for word,
        docs in relevant_containing_word_docs.items()
    }

    def similarity(doc: list) -> float:
        """
        Function to calculate similarity between the query and a document.

        Parameters:
            doc (list): list of words/tokens (str) contained by a doc.

        Return value:
            float: similarity between the query and the passed list of words.
        """

        def similarity_i(acc, cur):
            """
            Function to be used in a reduce() call to calculate the similarity
            between the doc and a single word from the query,
            then sum it to the similarity calculated for the previous words.

            Parameters:
                acc (float): result from the sum between the similarity
                             calculated for all the previous query token.
                cur (str): current word/token whose's similarity to the doc
                           will be calculated.

            Return value:
                float: sum between "acc" and the "cur"'s similarity with
                the doc
            """

            # if n_i > N/2 remove n_i from the numerator
            n_containing_word_num = n_containing_word[cur] if n_containing_word[cur] <= n_docs/2 else 0

            # applies the probabilistic model equation
            return acc + math.log10(
                ((n_relevant_containing_word[cur] + 0.5) * (n_docs - n_containing_word_num - n_relevant + n_relevant_containing_word[cur] + 0.5))
                /
                ((n_relevant - n_relevant_containing_word[cur] + 0.5) * (n_containing_word[cur] - n_relevant_containing_word[cur] + 0.5))
            )

        # uses the reduce() function to calculate the similarity between the query and the doc
        return reduce(similarity_i, get_intersection(query, doc), 0)

    # calculates the similarities between the query an each of the indexed docs
    answer = [
        { "doc": doc, "sim": similarity(words) }
        for doc, words in index.get_all_docs() if similarity(words) > 0
    ]
    answer.sort(key=lambda element: element["sim"], reverse=True)

    # returns only the doc names
    return [ doc['doc'] for doc in answer ]

def vectorialModel(query: list, index: Index, tdm: sp_sparse.csr_matrix = None) -> list:
    """
    Applies the vectorial model for information retrieval to calculate the
    similarity between the query and the indexed documents (considering the
    passed docs as relevant).

    Parameters:
        query (list): list of query's words/tokens.
        index (Index): database index (instance of the Index class).

    Return value:
        list: tuples with first element being doc name and the second
              being the rank value of document with given query.
    """

    documents                       = index.get_all_docs_names()
    number_of_documents_in_database = index.get_n_docs()
    unique_words                    = index.get_all_words()
    number_of_unique_words          = len(unique_words)

    # creating query vector
    query_vector = sp_sparse.lil_matrix((1, number_of_unique_words))

    # populates the query vector
    query_set = set(query)
    for word in query_set:
        if not word in unique_words: continue

        # number of documents containg word word
        ni = index.get_n_docs_containing(word)
        idf = math.log2(number_of_documents_in_database/ni)

        # word's index on the vocabulary
        i = unique_words.index(word)

        query_vector[0, i] = (1 + math.log2(query.count(word)))*idf

    # creating norm
    norm = tdm.power(2).sum(axis=0).A[0]

    # converting to more efficient type of sparse matrix
    query_vector_csr = query_vector.tocsr()

    # ranking documents for answer
    answer = []

    for j, doc in enumerate(documents):
        similarity = query_vector_csr.dot(tdm.getcol(j)).A[0][0]
        similarity  /= math.sqrt(norm[j])

        if similarity > 10**-2:
            answer.append({ "doc": doc, "sim": float(similarity) })

    # sorts the answer by similarity
    answer.sort(key=lambda element: element["sim"], reverse=True)

    # returns only the doc names
    return [ doc['doc'] for doc in answer ]
