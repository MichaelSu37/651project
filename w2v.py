from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import re, os, json, nltk
import numpy as np

def splitFile():
    paths = ['testing/', 'training/', 'validation/']
    outpath = ['test/', 'train/', 'validate/']

    for path in outpath:
        if not os.path(path):
            os.mkdir(path)
    
    path = paths[0]
    outp = outpath[0]
    files = sorted(os.listdir(path))
    textfile = []
    truthfile = []
    for f in files:
        if (f.endswith(".truth")):
            truthfile.append(f)
        else:
            textfile.append(f)
    
    fid = 0
    for j in range(len(textfile)):
        textname = textfile[j]
        truthname = truthfile[j]
        textf = open(textname, 'r')
        truthf = open(truthname, 'r')

        # get if style is changed, and positions of changing 
        info = json.load(truthf)

        content = textf.readlines()
        c = ''
        for line in content:
            if (line != '\n'):
                c += ' ' + line.strip()
        
        textf.close()
        truthf.close()

        # only split the files, and create labeling with 0
        if (not info["changes"]):
            lastSentence = 0
            count = 0
            sentences = c.split('.')
            words = c.split()

            segment = ''
            while (count < len(words) - 200):
                for k in range(count, count + 200):
                    if ('.' in words[count + k]):
                        lastSentence = words[count + k]
                    segment += words[count + k].replace('.', '')

                textoutname = outp + 'file' + str(fid) + '.txt'
                truthoutname = outp + 'file' + str(fid) + '.truth'
                textout = open(textoutname, 'w')
                truthout = open(truthoutname, 'w')

                textout.write(segment + '\n')
                truthout.write('0\n')

                fid += 1
                count += (k + 1)






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

