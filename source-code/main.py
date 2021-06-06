from index_class import Index
from util import *
from methods import probabilisticModel

words_in_doc = {}

database_contents = get_db_content()
index = Index(database_contents)

query = parse_text('to do')

print(probabilisticModel(query, index))
print(probabilisticModel(query, index, ['doc1.txt', 'doc3.txt']))
