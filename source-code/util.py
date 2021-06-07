from glob import glob
from nltk.stem import WordNetLemmatizer
from nltk import download, word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords as stpw
from functools import reduce
import re

# download('punkt')
# download('wordnet')
# download('stopwords')

DATABASE_DIRECTORY_PATH = './data/en.doc.2010/TELEGRAPH_UTF8/'
STOP_WORDS = stpw.words('english')

def get_db_content() -> dict:
    """
    Loops through all files in the database (data/) and reads their contents, returning it as a doc-content object.

    This method will parse documents marked up using the following tags:
        <DOC>: Starting tag of a document.
        <DOCNO> </DOCNO>: Contains document identifier.
        <TEXT> </TEXT>: Contains document text.
        </DOC>: Ending tag of a document.

    Return value:
        dict: object in which the keys are the doc's identifier and the value is the doc's full content (str)
    """

    contents = {}

    # DIRECTORY ORGANIZATION:
    # The top-level directory contains four directories, each corresponding
    # to the year of publication of the contained news articles. Each of these
    # directories is further divided into sub-directories corresponding to the
    # section/subject of the newspaper in which the various articles appeared (e.g.
    # nation, sports, business, etc). Total of 125586 documents.

    # list of all year directories in the database
    # publication_years = sorted(glob(DATABASE_DIRECTORY_PATH + '*'))
    publication_years = ['./data/en.doc.2010/TELEGRAPH_UTF8/2004_utf8']

    # loops through all files
    for year in publication_years:
        publication_subjects = sorted(glob(year + '/*'))

        for subject in publication_subjects:
            published_articles = sorted(glob(subject + '/*'))

            for article in published_articles:
                print(article)
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

    return WordNetLemmatizer().lemmatize((word, pos) if pos else word)

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

def get_intersection(list1, list2) -> list:
    """
    Uses the list comprehension to get the intersection between two lists (lists of lists are supported).

    Parameters:
        list1, list2: lists to intersect.

    Return value:
        list: intersection between list1 and list2.
    """

    return [ e for e in list1 if e in list2 ]

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

def extract_lists(list_of_lists: list):
    """
    Uses the reduce() method to extract all inner elements of a list of lists into the outer list - turning the list of lists into a simple list. It also sorts the resulting list.

    Parameters:
        list_of_lists (list): any list that contains other lists

    Return value:
        list: all inner elements from the passed list_of_lists sorted.
    """

    return sorted(reduce(lambda acc, cur: acc + cur, list_of_lists, []))
