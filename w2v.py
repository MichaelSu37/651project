from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import re, os, json, nltk
import numpy as np

wnl = nltk.WordNetLemmatizer()
unwanted = re.compile(r'[^a-z0-9-_\'\s]')

def segFile(fid, info, outp, words, c, styleChanged):
    if (styleChanged):
        positions = info["positions"]
        # partition the file based on style of change, each partition is written by one author
        partitions = []
        start = 0
        for position in positions:
            pos = sum(partitions) + len(c[start:position + 1])
            partitions.append(pos)
            start = position + 1
        pos = sum(partitions) + len(c[start:].split())
        partitions.append(pos)
        partitions = [x - 1 for x in partitions]

    lastSentence = 0

    done = False
    while not done:
        ylabel = 0
        segment = ''
        try:
            for k in range(lastSentence, lastSentence + 100):
                # a sequence of words has cross the point of style change.
                # meaning style has changed
                if (styleChanged and k in partitions):
                    ylabel = 1
                if ('.' in words[k]):
                    lastSentence = k
                segment += words[k].replace('.', '')
        except IndexError:
            for k in range(1, 101):
                segment += words[len(words) - k]
            done = True

        textoutname = outp + 'file' + str(fid) + '.txt'
        truthoutname = outp + 'file' + str(fid) + '.truth'
        textout = open(textoutname, 'w')
        truthout = open(truthoutname, 'w')

        # write the 100 words segment and the corresponding label to file
        textout.write(segment + '\n')
        truthout.write(str(ylabel) + '\n')

        textout.close()
        truthout.close()

        return fid


def splitFile():
    global wnl

    paths = ['testing/', 'training/', 'validation/']
    outpath = ['test/', 'train/', 'validate/']

    for path in outpath:
        if not os.path.exists(path):
            os.mkdir(path)

    allWords = []
    
    path = paths[0]
    outp = outpath[0]
    files = sorted(os.listdir(path))
    textfile = []
    truthfile = []
    for f in files:
        if (f.endswith(".truth")):
            truthfile.append(path + f)
        else:
            textfile.append(path + f)
    
    fid = 0
    for j in range(len(textfile)):
        textname = textfile[j]
        truthname = truthfile[j]
        textf = open(textname, 'r')
        truthf = open(truthname, 'r')

        # check if style is changed, and positions of changing 
        info = json.load(truthf)

        content = textf.readlines()
        c = ''
        for line in content:
            if (line != '\n'):
                c += ' ' + line.strip().lower()
        
        c = unwanted.sub(' ', c)
        textf.close()
        truthf.close()

        words = c.split()
        # TODO: remove plurals using WordNet
        for w in range(len(words)):
            #print (words[w])
            token = wnl.lemmatize(words[w])
            words[w] = token.encode('utf-8')

        allWords.extend(words)

        # only split the files, and create label with 0
        if (not info["changes"]):
            segFile(fid, info, outp, words, c, 0)
        # need to split the file as well as creating new labels for them
        else:
            segFile(fid, info, outp, words, c, 1)

        fid += 1

    return allWords





if __name__ == '__main__':
    d = []

    with open('dictionary.txt') as f:
        line = f.readline().strip().lower()
        while line != '':
            d.append(line)
            line = f.readline().strip().lower()

    words = splitFile()
    d.extend(words)
    wordList = list(set(d))
    model = Word2Vec([wordList], min_count = 1, workers = 4)
    model.save("word2vec.model")

    model = Word2Vec.load("word2vec.model")
    v = model.wv['computer']
    print (v)

