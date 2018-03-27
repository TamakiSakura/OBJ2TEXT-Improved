import numpy as np
from nltk.stem.porter import PorterStemmer

def cat2vocab(coco, vocab):
    print("Creating Converter...")
    converter = np.zeros((100, vocab.idx))
    lemmatizer = PorterStemmer()
    for cat_id in coco.cats.keys():
        categories = coco.cats[cat_id]['name'].split(" ")
        for category in categories:
            category = lemmatizer.stem(category)
            for vocab_id in range(vocab.idx):
                word = lemmatizer.stem(vocab.idx2word[vocab_id])
                if word == category:
                    converter[cat_id, vocab_id] = 1
    return converter
            
