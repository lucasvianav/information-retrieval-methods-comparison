import re
from functools import reduce
from glob import glob
from typing import Dict, List, Union

from nltk import download, word_tokenize
from nltk.corpus import stopwords as stpw
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# download('punkt')
# download('wordnet')
# download('stopwords')

DATA_PATH = './data/'
DATABASE_DIRECTORY_PATH = DATA_PATH + 'en.doc.2010/TELEGRAPH_UTF8/'
QUERIES_FILENAME = 'en.topics.76-125.2010.txt'
QUERIES_RESULT_FILENAME = 'en.qrels.76-125.2010.txt'
STOP_WORDS = stpw.words('english')

Number = Union[float, int]

def get_db_content() -> Dict[str, str]:
    """
    Loops through all files in the database (data/) and reads their contents,
    returning it as a doc-content object.

    This method will parse documents marked up using the following tags:
        <DOC>: Starting tag of a document.
        <DOCNO> </DOCNO>: Contains document identifier.
        <TEXT> </TEXT>: Contains document text.
        </DOC>: Ending tag of a document.

    Return value:
        dict: object in which the keys are the doc's identifier and the value
              is the doc's full content (str).
    """

    contents = {}

    # DIRECTORY ORGANIZATION:
    # The top-level directory contains four directories, each corresponding
    # to the year of publication of the contained news articles. Each of these
    # directories is further divided into sub-directories corresponding to the
    # section/subject of the newspaper in which the various articles appeared (e.g.
    # nation, sports, business, etc). Total of 125586 documents.

    # list of all year directories in the database
    publication_years = glob(DATABASE_DIRECTORY_PATH + '*')

    # loops through all files
    for year in publication_years:
        publication_subjects = glob(year + '/*')

        for subject in publication_subjects:
            published_articles = glob(subject + '/*')

            for article in published_articles:
                with open(article, 'r') as file:
                    raw_content = file.read()
                    file_id = re.sub(r'.+?<DOCNO>(.+?)</DOCNO>.+', r'\g<1>', raw_content, flags=re.DOTALL)
                    file_text = re.sub(r'.+?<TEXT>(.+?)</TEXT>.+', r'\g<1>', raw_content, flags=re.DOTALL)

                    contents[file_id] = file_text

    return contents

def remove_special_characters(text: str) -> str:
    """
    Uses regex to remove characters that are neither alpha-numeric nor whitespace from a text.

    Parameters:
        text (str): the text to be filtered.

    Return value:
        str: the same text but with no special characters.
    """

    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def lemmatize(word: str, pos="") -> str:
    """
    Shortcut to nltk.stem.WordNetLemmatizer().lemmatize(word, pos).

    Parameters:
        word (str): the word to be lemmatized.
        pos (str): the word-type the result should be turned into (default is noun).

    Return value:
        str: the lemmatized word.
    """

    return str(WordNetLemmatizer().lemmatize((word, pos) if pos else word))

def stem(word: str, language='english') -> str:
    """
    Shortcut to nltk.stem.SnowballStemmer(language).stem().

    Parameters:
        word (str): the word to be lemmatized.
        pos (str): the word-type the result should be turned into (default is noun).

    Return value:
        str: the lemmatized word.
    """

    return SnowballStemmer(language).stem(word)

def get_intersection(list1: list, list2: list) -> list:
    """
    Uses the list comprehension to get the intersection between two lists (lists of lists are supported).

    Parameters:
        list1, list2: lists to intersect.

    Return value:
        list: intersection between list1 and list2.
    """

    return [ e for e in list1 if e in list2 ]

def get_sum(list1: List[Number], list2: List[Number]) -> List[Number]:
    """
    Performs the item-wise sum of two lists of numbers (either int or float).

    Parameters:
        list1 (list<int, float>), list2 (list<int, float>): lists to be summed.

    Return value:
        list<int, float>: item-wise sum between list1 and list2.
    """

    return [ a + b for a, b in zip(list1, list2) ]

def get_division(list1: List[Number], **kwargs) -> List[Number]:
    """
    Performs the item-wise division of two lists of numbers (either int or
    float) or between a list of number and a number.

    Either "list2" or "number" must be provided as second argument. If both are
    received, an error will be thrown.

    Parameters:
        list1 (list<int, float>): lists to be divided.
        list2 (list<int, float>): list do divide list1 by.
        number (int, float): number to divide list1 by.

    Return value:
        list<int, float>: item-wise division between list1 and list2.
    """

    result = []

    if 'list2' in kwargs and 'number' in kwargs:
        raise ValueError('Either a list of numbers or a number must be\
                          provided to get_division(), but no both.')

    elif 'list2' in kwargs:
        list2 = list(kwargs['list2'])
        result = [ a / b for a, b in zip(list1, list2) ]

    elif 'number' in kwargs:
        number = float(kwargs['number'])
        result = [ a / number for a in list1 ]

    else:
        raise ValueError('Either a list of numbers or a number must be\
                         provided to get_division().')

    return result

def parse_text(text: str, filter_stopwords: bool, stem_words: bool) -> list:
    """
    Removes all special characters from and tokenizes a text, then normalizes and lemmatizes each token (word).

    Parameters:
        text (str): the text to be parsed.
        filter_stopwords (bool): if true, all of the text stopwords will be ignored.
        stem_words (bool): if true, all of the words will be lemmatized and stemmed.

    Return value:
        list: parsed tokens/words.
    """

    return [
        stem(lemmatize(word)) if stem_words else word for word in word_tokenize(remove_special_characters(text).lower())
        if word not in ( STOP_WORDS if filter_stopwords else [] ) # ignores stopwords if the filter is activated
    ]

def extract_lists(list_of_lists: List[list]):
    """
    Uses the reduce() method to extract all inner elements of a list of lists
    into the outer list - turning the list of lists into a simple list. It also
    sorts the resulting list.

    Parameters:
        list_of_lists (list<list>): any list that contains other lists

    Return value:
        list: all inner elements from the passed list_of_lists sorted.
    """

    return sorted(reduce(lambda acc, cur: acc + cur, list_of_lists, []))

def get_queries() -> List[Dict[str, str]]:
    """
    Parses the query-list document and returns the list of queries.

    This method will parse queries marked up using the following tags:
        <topics>:         Contains the whole query list.
        <top> </top>:     Contains the query's data.
        <num> </num>:     Contains the query's id.
        <title> </title>: Contains the query.
        <desc> </desc>:   Contains the query's brief description.
        <narr> </narr>:   Contains the query's full description.

    Return value:
        list: list of dicts with the 'id' and 'query' keys and their respective
              values.
    """

    with open(DATA_PATH + QUERIES_FILENAME, 'r') as file:
        raw_content = file.read()
        queries = re.findall(r'<num>(\d+?)</num>.*?<title>(.+?)</title>',
                             raw_content,
                             re.DOTALL)

        contents = [
            { 'id': id, 'query': query.replace('\n', ' ') }
            for id, query in queries
        ]

    return contents

def get_queries_truth_set() -> Dict[str, List[str]]:
    """
    Parses the results document and returns each query's truth set (list of
    relevant documents for that query).

    This method will parse queries marked up using the following columns:
        1st column: query id.
        2nd column: irrelevant.
        3rd column: document title.
        4th column: relevancy ('1' for relevant, '0' for irrelevant).

    Return value:
        dict: objects in which each key is the query's numerical id and each
              value is it's truth set.
    """

    contents = {}

    with open(DATA_PATH + QUERIES_RESULT_FILENAME, 'r') as file:
        raw_content = file.read()
        results = re.findall(r'^(\d+?)\s\S{2}\s(\S+?)\s1$',
                             raw_content,
                             re.MULTILINE)


        for result in results:
            query_id, relevant_doc = result

            if query_id in contents.keys():
                contents[query_id].append(relevant_doc)

            else: contents[query_id] = [ relevant_doc ]

    return contents

