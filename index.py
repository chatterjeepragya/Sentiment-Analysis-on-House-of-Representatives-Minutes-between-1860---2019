import os.path, glob
from whoosh import index, fields
from whoosh.fields import Schema, ID, TEXT
from whoosh.analysis import StemmingAnalyzer

if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
    schema = Schema(title=fields.TEXT(sortable=True), path=ID(unique=True,
    stored=True), content=TEXT(analyzer=StemmingAnalyzer(), stored=True))
    ind = index.create_in("indexdir", schema)  
    writer = ind.writer()  
    for path in glob.glob('corpustexts/*.txt'): 
        with open(path, 'r', encoding='utf-8', errors='ignore') as fhand:  
            content = fhand.read() 
            writer.add_document(path=path, content=content)  
    writer.commit()
    

