from .PreProcessor import getDocuments
from .PreProcessor import processDocuments
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import math
import heapq
import os

def wordsDictionary(docAfterProcessing):
    doc = 1;
    wordsDictionary = {}
    for each_doc in docAfterProcessing:
        word_count = 1
        words = each_doc.split(" ")

        for each_word in words:
            if len(each_word) <= 2:
                continue

            if each_word not in wordsDictionary:
                wordsDictionary[each_word] = {}
            wordsDictionary[each_word][doc] = word_count
            word_count += 1

        doc += 1

    return wordsDictionary

def makeStemVectors(wordsDictionary,stemsDictionary,doc_length):

    stemVector = {}
    for each_stem in stemsDictionary:
        stemVector[each_stem] = {}

        for doc_id in range(1,doc_length+1):
            stemVector[each_stem][doc_id] = 0
            res = 0
            stemSet = stemsDictionary[each_stem]

            for each_word in stemSet:
                if doc_id in wordsDictionary[each_word]:
                    res+=1
            stemVector[each_stem][doc_id] = res

    return stemVector

def getDotProduct(stemVector,length):

    dotProduct = {}
    for each_stem1 in stemVector:
        dotProduct[each_stem1] = {}
        for each_stem2 in stemVector:
            prod = 0
            for each_doc in range(1,length+1):

                prod+= (stemVector[each_stem1][each_doc]*stemVector[each_stem2][each_doc])

            dotProduct[each_stem1][each_stem2] = prod

    return dotProduct

def getMagnitue(stemVector):

    magnitude = {}

    for each_stem in stemVector:
        sum = 0
        for doc_id in stemVector[each_stem]:
            sum+=(stemVector[each_stem][doc_id]*stemVector[each_stem][doc_id])

        sqrt = math.sqrt(sum)
        magnitude[each_stem] = sqrt

    return magnitude

def getScalarClustering(dotProduct,magnitude):

    scalarMatrix = {}

    for each1 in dotProduct:
        scalarMatrix[each1] = {}
        for each2 in dotProduct:

            scalarMatrix[each1][each2] = (dotProduct[each1][each2])/(magnitude[each1]*magnitude[each2])

    return scalarMatrix

def getFinalQuery(association,query,top,ps):

    querySet = set(); check = set()
    queryStems = getStemsFromSentence(query,ps)

    for each in queryStems:
        check.add(each)

    newQuery = query
    for each_stem in queryStems:
        if each_stem in association:
            topN = heapq.nlargest(top+10,association[each_stem],key=association[each_stem].get)
            curr = 1
            for new_stem in topN:
                if (new_stem in check) or len(new_stem)<=3:
                    continue

                if curr>top:
                    break
                elif new_stem not in querySet:
                    querySet.add(new_stem)
                    curr+=1

    for each in querySet:
        newQuery+=" "+each
    return newQuery


def getStemsFromSentence(sentence,ps):

    stems = []
    words = word_tokenize(sentence)

    for each_word in words:
        stems.append(ps.stem(each_word))

    return stems

def makeStems(wordsDictionary,ps):

    cwd = os.getcwd()
    with open(cwd + "/Climate_App/Services/stopwords", "r") as f:
        stop_list = set(ps.stem(line.rstrip('\n')) for line in f)
    stemsDictionary = {}

    for each_word in wordsDictionary:

        stem = ps.stem(each_word)
        if stem in stop_list:
            continue;
        if stem not in stemsDictionary:
            stemsDictionary[stem] = set()

        stemsDictionary[stem].add(each_word)

    return stemsDictionary

def getExpandedQuery(query,data):
    ps = PorterStemmer()
    documents = getDocuments(data)
    docAfterProcessing = processDocuments(documents);length = len(docAfterProcessing)
    words = wordsDictionary(docAfterProcessing)
    stemsDictionary = makeStems(words, ps)
    stemVector = makeStemVectors(words, stemsDictionary, length)
    dotProduct = getDotProduct(stemVector, length)
    magnitude = getMagnitue(stemVector)
    scalarClustering = getScalarClustering(dotProduct, magnitude)
    query = getFinalQuery(scalarClustering, query, 3, ps)
    return query