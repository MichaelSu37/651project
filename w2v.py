from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import re, os, json, nltk
import numpy as np

wnl = nltk.WordNetLemmatizer()
unwanted = re.compile('[^a-z0-9_!?\']')
#symbols = re.compile('[-()_.!?<>|+]')
symbols = re.compile('[^a-z0-9]')

# splits each file into 100-words segments
def segFile(fid, fnumber, info, outp, words, c, styleChanged):
    partitions = []
    if (styleChanged):
        positions = info["positions"]
        # partition the file based on style of change, each partition is written by one author
        partitions = [0]
        start = 0
        for position in positions:
            pos = partitions[-1] + len(c[start:position + 1].split())
            partitions.append(pos)
            start = position + 1
        pos = partitions[-1] + len(c[start:].split())
        partitions.append(pos)
        partitions.pop(0)
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
                #if ('.' in words[k]):
                segment += (' ' + symbols.sub('', words[k]))
            lastSentence = k
        except IndexError:
            segment = ''
            for k in range(1, 101):
                segment += (' ' + symbols.sub('', words[len(words) - k]))
            done = True

        textoutname = outp + 'problem_' + str(fnumber) + '_' + str(fid) + '.txt'
        truthoutname = outp + 'problem_' + str(fnumber) + '_' + str(fid) + '.truth'
        textout = open(textoutname, 'w')
        truthout = open(truthoutname, 'w')

        # write the 100 words segment and the corresponding label to file
        textout.write(segment + '\n')
        truthout.write(str(ylabel) + '\n')

        textout.close()
        truthout.close()
        fid += 1
    return fid


def splitFile():
    global wnl

    paths = ['testing/', 'training/', 'validation/']
    outpath = ['test/', 'train/', 'validate/']

    for path in outpath:
        if not os.path.exists(path):
            os.mkdir(path)

    allWords = []
    
    for i in range(len(paths)):
        path = paths[i]
        outp = outpath[i]
        files = sorted(os.listdir(path))
        textfile = []
        truthfile = []
        for f in files:
            if (f.endswith(".truth")):
                truthfile.append(path + f)
            else:
                textfile.append(path + f)

        textfile.sort()
        truthfile.sort()
        
        fid = 0
        fnumber = 1
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

            for w in range(len(words)):
                token = wnl.lemmatize(words[w])
                words[w] = symbols.sub('', token)

            allWords.append(words)
            
            # only split the files, and create label with 0
            if (not info["changes"]):
                fid = segFile(fid, fnumber, info, outp, words, c, 0)
            # need to split the file as well as creating new labels for them
            else:
                fid = segFile(fid, fnumber, info, outp, words, c, 1)
            
            fnumber += 1

    return allWords

def genModel():
    d = []

    wordList = splitFile()

    model = Word2Vec(wordList, min_count = 1, workers = 4)
    model.save("word2vec.model")

    model = Word2Vec.load("word2vec.model")
    v = model.wv['computer']
    print(len(v))

# each X[i] is a 100 * 100 matrix representing one file
def getSample(option):
    global wnl
    '''
    option: one of the values in ['train', 'test', 'validate']

    return value:
        X:  dimension: numfiles * 100 (words) * 100 (w2v dimension)

        y:  dimension: (numfiles * 1)
    '''
    model = Word2Vec.load("word2vec.model")
    paths = {'test':'test/', 'train':'train/', 'validate':'validate/'}
    path = paths[option]
    files = sorted(os.listdir(path))
    truthfile = []
    textfile = []

    for f in files:
        if (f.endswith(".truth")):
            truthfile.append(path + f)
        else:
            textfile.append(path + f)
    
    truthfile.sort()
    textfile.sort()

    y = np.zeros(len(textfile))
    X = np.zeros((len(textfile), 100, 100))

    for i in range(len(textfile)):
        textf = open(textfile[i], 'r')
        truthf = open(truthfile[i], 'r')
        words = textf.readline().strip().split()
        label = float(truthf.readline().strip())
        for j, word in enumerate(words):
            word = wnl.lemmatize(word)
            try:
                X[i][j] = model.wv[word]
            except:
                pass
        y[i] = label

    return X, y


if __name__ == '__main__':
    genModel()
    options = ['train', 'test', 'validate']
    getSample(options[0])

