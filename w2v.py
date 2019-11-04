from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import re, os
import numpy as np

def splitFile():
    paths = ['testing/', 'training/', 'validation/']
    outpath = ['test/', 'train/', 'validate/']

    for path in outpath:
        if not os.path(path):
            os.mkdir(path)
    
    path = paths[0]
    files = os.listdir(path)
    textfile = []
    truthfile = []
    


if __name__ == '__main__':
    all_words = []

    with open('dictionary.txt') as f:
        line = f.readline().strip().lower()
        while line != '':
            all_words.append(line)
            line = f.readline().strip().lower()

    model = Word2Vec([all_words], min_count = 1, workers = 4)
    model.save("word2vec.model")

    model = Word2Vec.load("word2vec.model")
    v = model.wv['computer']
    print (v)

