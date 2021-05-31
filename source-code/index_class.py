from functools import reduce
from util import *

class Index:
    """ 
    This is a class for information retrieval index storing. 

    With this, it's possible to store a list of words associated with it's "posting list" - list of documents from the database in which each word is present as well as it's frequency in that document.

    Parameters:
        database_contents (dict): a dictonary with the database documents' names as keys and it's full contents (string) as values.

    Return value:
        Index, if the passed parameters are valid.
        None, if not.
    """

    def __init__(self, database_contents: dict): 
        """
        The constructor for the Index class.

        Parameters:
            database_contents (dict): a dictonary with the database documents' names as keys and it's full contents (string) as values.

        Return value:
            Index, if the passed parameters are valid.
            None, if not.
        """

        # keys --> doc name
        # values --> list of all words in that doc (with repetitions)
        words_in_doc = {}
        
        # list of documents in database
        self.doc_mapping = []
        

        # loops through each file and it's contents
        for file, content in database_contents.items():
            # if the passed file is invalid, return None
            if type(content) is not str: return None

            # parses the content text and stores it
            words = parse_text(content)
            words_in_doc[file] = words
            
            # append document to doc_mapping
            self.doc_mapping.append(file)

    

        # # sorts by doc
        # words_in_doc = dict(sorted(words_in_doc.items(), key=lambda e: e[0]))

        # list containing every word processed (sorted and no repetition)
        all_words = set(extract_lists(words_in_doc.values()))

        # keys --> words (sorted)
        # values --> list of dicts containing the doc name and the word frequency in that doc (sorted by doc)
        self.posting_list = { 
            word: sorted(
                list(
                    map(
                        lambda doc: { 'doc': doc[0], 'freq': doc[1].count(word) },
                        filter( lambda doc: word in doc[1], words_in_doc.items() )
                    )
                ), 

                key=lambda e: e['doc']
            )

            for word in all_words 
        }
        
        # keys --> doc name
        # values --> list of all words in that doc (with repetitions)
        self.words_in_doc = words_in_doc


    def add_docs_to_word(self, word: str, posting_list: list) -> None:
        """
        The constructor for the Index class.

        Parameters:
            word (str): the target-word.
            posting_list (list): a list containing dicts with the 'doc' as key and the doc name as value (str), as well as 'freq' as key and that word's frequency in the doc as value (int).

        Return value: None.
        """

        # if the word is not yet present in the index, adds it
        if word not in self.posting_list.keys(): self.posting_list[word] = []

        # loops through the posting list
        for e in posting_list:
            # if the current dict is invalid, ignore it
            if type(e['doc']) is not str or type(e['freq']) is not int: continue

            # computes the doc and word into self.words_in_doc
            elif e['doc'] not in self.words_in_doc.keys(): self.posting_list[e['doc']] = [ word for _ in range(e['freq']) ]
            else: self.words_in_doc[e['doc']] = sorted(self.words_in_doc[e['doc']] + [word for _ in range(e['freq'])])

            # computes the doc and word into self.posting_list
            try: doc_freq = next(filter(lambda d: e['doc'] == d['doc'], self.posting_list[word]))
            except StopIteration: self.posting_list[word].append(e.copy())
            else: doc_freq['freq'] += e['freq']
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

        return map(lambda e: e['doc'], self.posting_list[word]) if word in self.posting_list.keys() else []

    def get_words_in_doc(self, doc: str) -> list: 
        """
        The getter for the posting list of a word.

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

    def get_n_different_words(self, doc: str) -> str:
        """
        The getter for the number of different words contained by a doc.

        Parameters:
            doc (str): the target-doc.

        Return value:
            int: number of words contained by the target-doc.
        """
        !!!!!!!!!!!!!!!!!!!!!! documentacao errada... !!!!!!!!!!!!!!!!!!!!!

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
        The getter for all of the docs in the databse (returning both the doc's name and words),

        Parameters:
            dict (bool): if true, the returned value will be a list of {'doc': DOC_NAME, 'words': DOC_WORDS}, if false it'll be a list of (DOC_NAME, DOC_WORDS).

        Return value:
            list: contains the docs's info - list of {'doc': DOC_NAME, 'words': DOC_WORDS} if dict is True, (DOC_NAME, DOC_WORDS) otherwise
        """
        
        return_list = sorted(self.words_in_doc.items(), key=lambda e: e[0])

        return return_list if not dict else list(map(lambda e: { 'doc': e[0], 'words': e[1] }, return_list))

    def get_doc_mapping(self) -> list:
        """
        getter of doc_mapping list of index class

        Return value:
            list: all names of documents in index class
        """
        return self.doc_mapping
