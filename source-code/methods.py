from functools import reduce
from util import get_intersection
from index_class import Index
import scipy.sparse as sp_sparse
import numpy as np
import math

def probabilisticModel(query: list, index: Index, relevant_docs = []) -> dict:
    """
    Applies the probabilistic model for information retrieval to calculate the similarity between
    the query and the indexed documents (considering the passed docs as relevant).

    Parameters:
        query (list): list of query's words/tokens.
        index (Index): database index (instance of the Index class).
        relevante_docs (list | []): names of relevant docs for that query (default empty list).

    Return value:
        dict: doc names as keys and their similarity to the query as values (float).
    """

    # key --> word/token
    # value --> names of docs containing that word
    containing_word_docs = { word: index.get_doc_list(word) for word in query }

    # key --> word/token
    # value --> names of relevant docs containing that word
    relevant_containing_word_docs = { word: get_intersection(relevant_docs, docs) for word, docs in containing_word_docs.items() }

    # n_ == number of
    n_docs = index.get_n_docs() # N
    n_containing_word = { word: index.get_n_docs_containing(word) for word in query } # n_i
    n_relevant = len(relevant_docs) # R
    n_relevant_containing_word = { word: len(docs) for word, docs in relevant_containing_word_docs.items() } # r_i

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
    return { doc: similarity(words) for doc, words in index.get_all_docs() }

def vectorialModel(query: list, index: Index) -> list:
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

    number_of_documents_in_database = index.get_n_docs()
    unique_words = index.get_all_words()
    number_of_unique_words = len(unique_words)
    doc_mapping = index.get_all_docs_names()

    # creating TDM base (Term Document Matrix)
    tdm = sp_sparse.lil_matrix((number_of_unique_words,
                                number_of_documents_in_database))


    for i in range(number_of_unique_words):
        word = unique_words[i]
        # list of document and frequency in document
        # of word word
        postings = index.get_posting_list(word)

        # Number of documents containg word word
        ni = index.get_n_docs_containing(word)

        idf = math.log2(number_of_documents_in_database/ni)

        # populate TDM
        for node in postings:
            docName = node["doc"]
            docId = index.get_doc_id(docName)
            frequency_in_doc = node["freq"]

            # Populating TDM
            tdm[i, docId] = (1 + math.log2(frequency_in_doc))*idf


    # creating norm
    norm = tdm.power(2).sum(axis=0).A[0]
    norm = [math.sqrt(norm[i]) for i in range(len(norm))]

    # creating query vector
    query_vector = np.zeros(number_of_unique_words)
    for i in range(number_of_unique_words):
        word = unique_words[i]
        if word in query:
            ni = index.get_n_docs_containing(word)
            idf = math.log2(number_of_documents_in_database/ni)

            query_vector[i] = (1 + math.log2(query.count(word)))*idf

    # ranking documents for answer
    answer = []
    ranking = np.zeros(number_of_documents_in_database)
    for j in range(number_of_documents_in_database):
        ranking[j] = np.dot(query_vector) / norm[j]
        answer.append((doc_mapping[j], ranking[j]))

    return sorted(answer, key = lambda x:x[1])

