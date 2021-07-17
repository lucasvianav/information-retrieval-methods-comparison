from functools import reduce

from util import *


class Index:
    """
    This is a class for information retrieval index storing.

    With this, it's possible to store a list of words associated with it's "posting list" - list of documents from the database in which each word is present as well as it's frequency in that document.

    Parameters:
        database_contents (dict): a dictonary with the database documents' names as keys and it's full contents (string) as values.
        filter_stopwords (bool): if true, all of the query stopwords will be ignored.
        stem_words (bool): if true, all of the query words will be lemmatized and stemmed.

    Return value:
        Index, if the passed parameters are valid.
        None, if not.
    """

    def __init__(self, database_contents: dict, filter_stopwords: bool, stem_words: bool):
        """
        The constructor for the Index class.

        Parameters:
            database_contents (dict): a dictonary with the database documents' names as keys and it's full contents (string) as values.
            filter_stopwords (bool): if true, all of the query stopwords will be ignored.
            stem_words (bool): if true, all of the query words will be lemmatized and stemmed.

        Return value:
            Index, if the passed parameters are valid.
            None, if not.
        """

        # sorts $database_contents by name of doc and
        # creates list of tuples (docname, docwords)
        database = sorted(database_contents.items(), key=lambda e: e[0])

        # keys --> doc name
        # values --> list of all words in that doc (with repetitions)
        self.words_in_doc = {}

        # keys --> doc name
        # values --> doc numerical id
        self.doc_id = {}

        # set containing the whole vocabulary
        all_words = set()

        # keys --> words (sorted)
        # values --> list of dicts containing the doc name and the word frequency in that doc (sorted by doc)
        self.posting_list = {}

        # loops through each doc and it's contents
        for i, items in enumerate(database):
            # unpacks items
            doc, content = items

            # if the passed doc is invalid, return None
            if type(content) is not str: return None

            # parses the content text and stores it
            words = parse_text(content, filter_stopwords, stem_words)
            self.words_in_doc[doc] = words

            for word in set(words):
                # adds current word to the set
                all_words.add(word)

                # if the word is not yet present in the index, adds it
                if word not in self.posting_list.keys(): self.posting_list[word] = []

                # appends the current doc to this word's posting list
                self.posting_list[word].append({'doc': doc, 'freq': words.count(word)})

            # sets doc_id
            self.doc_id[doc] = i

        # list containing every word processed (sorted and no repetition)
        self.all_words = sorted(list(all_words))

        self.filter_stopwords = filter_stopwords
        self.stem_words = stem_words

    def add_docs_to_word(self, word: str, posting_list: list) -> None:
        """
        Method to include a list of docs into a word's posting list.

        Parameters:
            word (str): the target-word (if it's a stopword, it may be ignored).
            posting_list (list): a list containing dicts with the 'doc' as key and the doc name as value (str), as well as 'freq' as key and that word's frequency in the doc as value (int).

        Return value: None.
        """

        # if the word is a stopword, ignore it
        if word in STOP_WORDS and self.filter_stopwords: return

        # if the word is not yet present in the index, adds it
        elif word not in self.posting_list.keys(): self.posting_list[word] = []

        # the last doc in the list's numerical id
        last_id = list(self.doc_id.values())[-1]

        # loops through the posting list
        for i, element in enumerate(posting_list):
            doc = element['doc']
            freq = element['freq']

            # if the current dict is invalid, ignore it
            if type(doc) is not str or type(freq) is not int: continue

            # computes the doc and word into self.words_in_doc
            elif doc not in self.words_in_doc.keys():
                self.posting_list[doc] = [ word for _ in range(freq) ]
                self.doc_id[doc].append({ doc: last_id + 1 + i })

            else: self.words_in_doc[doc] = sorted(self.words_in_doc[doc] + [word for _ in range(freq)])

            # computes the doc and word into self.posting_list
            try: doc_freq = [ node for node in self.posting_list[word] if node['doc'] == doc ][0]
            except IndexError: self.posting_list[word].append(element.copy())
            else: doc_freq['freq'] += freq
            finally: self.posting_list[word].sort(key=lambda e: e['doc'])

    def get_posting_list(self, word: str) -> list:
        """
        The getter for the posting list of a word.

        Parameters:
            word (str): the target-word.

        Return value:
            list: the posting list contains dicts with the 'doc'
                  as key and the doc name as value (str), as well
                  as 'freq' as key and that word's frequency in the
                  doc as value (int) - lists all docs that contain
                  the target-word.
        """

        return self.posting_list[word] if word in self.posting_list.keys() else []

    def get_doc_list(self, word: str) -> list:
        """
        The getter for the list of docs that contains a word.

        Parameters:
            word (str): the target-word.

        Return value:
            list: docs that contain the target-word.
        """

        return [ e['doc'] for e in self.posting_list[word] ] if word in self.posting_list.keys() else []

    def get_words_in_doc(self, doc: str) -> list:
        """
        The getter for the list of words contained by a doc.

        Parameters:
            doc (str): the target-doc.

        Return value:
            list: all words contained by the target-doc (with repetitions).
        """

        return self.words_in_doc[doc] if doc in self.words_in_doc.keys() else []

    def get_n_docs(self) -> int:
        """
        The getter for the number of docs in the database.

        Return value:
            int: total number of docs indexed.
        """

        return len(self.words_in_doc.keys())

    def get_n_docs_containing(self, word: str) -> int:
        """
        The getter for the number of docs that contain a word.

        Parameters:
            word (str): the target-word.

        Return value:
            int: number of docs that contains the target-word.
        """

        return len(self.posting_list[word]) if word in self.posting_list.keys() else 0

    def get_n_different_words(self, doc: str) -> int:
        """
        The getter for the number of different words contained by a doc.

        Parameters:
            doc (str): the target-doc.

        Return value:
            int: number of words contained by the target-doc.
        """


        return len(set(self.words_in_doc[doc])) if doc in self.words_in_doc.keys() else 0

    def get_total_freq(self, word: str) -> int:
        """
        The getter for total frequency of a word (number of times it appears through all docs).

        Parameters:
            word (str): the target-word.

        Return value:
            int: the total frequency of the word.
        """

        return reduce(lambda acc, cur: acc + cur['freq'], self.posting_list[word], 0) if word in self.posting_list.keys() else 0

    def print_posting_list(self, word: str) -> None:
        """
        Prints a word's posting list into stdout.

        Parameters:
            word (str): the target-word.

        Return value: None.
        """

        print('\n'.join(map(lambda e: f'doc: {e["doc"]}, freq: {e["freq"]}', self.posting_list[word])))

    def print_words_in_doc(self, doc: str) -> None:
        """
        Prints a doc's list of words.

        Parameters:
            doc (str): the target-doc.

        Return value: None.
        """

        print('\n'.join(self.words_in_doc[doc]))

    def get_all_docs(self, dict=False) -> list:
        """
        The getter for all of the docs in the database (returning both the doc's name and words).

        Parameters:
            dict (bool): if true, the returned value will be a list of {'doc': DOC_NAME, 'words': DOC_WORDS}, if false it'll be a list of (DOC_NAME, DOC_WORDS).

        Return value:
            list: contains the docs's info - list of {'doc': DOC_NAME, 'words': DOC_WORDS} if dict is True, (DOC_NAME, DOC_WORDS) otherwise
        """

        return_list = sorted(self.words_in_doc.items(), key=lambda e: e[0])

        return return_list if not dict else list(map(lambda e: { 'doc': e[0], 'words': e[1] }, return_list))

    def get_all_docs_names(self) -> list:
        """
        The getter for all of the docs in the database (returning only their names).

        Return value:
            list: all names of documents in the Index.
        """

        return sorted(self.words_in_doc.keys())

    def get_all_words(self) -> list:
        """
        The getter for the Index's vocabulary (all words with no repetitions).

        Return value:
            list: containing every word processed (sorted and no repetition)
        """
        return self.all_words

    def get_doc_id(self, name: str) -> int:
        """
        The getter for a doc's numerical id.

        Parameters:
            name (str): the target-document's name.

        Return value:
            int: the document's numerical id (-1 if the doc is not found).
        """

        return self.doc_id[name] if name in self.doc_id.keys() else -1

    def get_doc_name(self, id: int) -> str:
        """
        The getter for a doc's name.

        Parameters:
            id (int): the document's numerical id.

        Return value:
            str: the target-document's name
        """

        return [ doc[0] for doc in self.doc_id.items() if doc[1] == id ][0] if int in self.doc_id.values() else ''

    def get_all_docs_ids(self) -> list:
        """
        The getter for all of the docs in the database (returning only their numerical ids).

        Return value:
            list: all numerical ids of documents in the Index.
        """

        return sorted(self.doc_id.values())

    def get_all_words_in_docs(self, docs: list) -> list:
        """
        The getter for the vocabulary (all different words) in a list of docs.

        Parameters:
            docs (list): list of doc names.

        Return value:
            list: all different words contained by the docs targeted.
        """

        return list(set(extract_lists([ words for doc, words in self.words_in_doc.items() if doc in docs ])))

    def get_frequency_in_doc(self, word: str, doc: str) -> int:
        """
        The getter for a word's frequency in a doc.

        Parameters:
            word (str): the target-word.
            doc (str): the queried doc.

        Return value:
            int: the target-word's frequency in the queried doc.
        """

        return self.words_in_doc[doc].count(word)
