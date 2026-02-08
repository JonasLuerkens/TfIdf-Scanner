import nltk
import nltk.corpus
import sys

def select_corpus():
    # Corpus selection
    while True:
        user_input = input("Provide your desired NLTK corpus or just enter for default (reuters): ").strip()
        
        if user_input.lower() == 'exit':
            sys.exit(0)
        
        # Set default (reuters) for empty input
        corpus_name = user_input if user_input else 'reuters'
        corpus_name = corpus_name.lower()

        # Get data
        docs, ids = get_corpus_data(corpus_name)
        
        # Loading Feedback 
        if docs:
            return docs, ids, corpus_name
        else:
            print(f"Corpus not found, try again.")

def get_corpus_data(corpus_name):
    # Get the data of the corpus

    # Download check
    try: 
        nltk.data.find(f'corpora/{corpus_name}')
    except LookupError: 
        nltk.download(corpus_name, quiet=True)
    
    # Load corpus
    try:
        corpus = getattr(nltk.corpus, corpus_name)
        
        # Get all file ids from corpus
        doc_ids = corpus.fileids()
        
        # Load text of each document
        documents = [corpus.raw(doc_id) for doc_id in doc_ids]
        
        return documents, doc_ids
        
    except AttributeError:
        # Corpus not found
        return None, None