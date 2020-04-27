from .PreProcessor import getDocuments
from .PreProcessor import processDocuments
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import heapq
import os


def makeAssociationMatrix(docAfterProcessing,ps,length):
    associationMatrix = {};

    cwd = os.getcwd()
    with open(cwd + "/Climate_App/Services/stopwords", "r") as f:
        stop_list = set(ps.stem(line.rstrip('\n')) for line in f)

    for id in range(length):

        doc = docAfterProcessing[id]
        words = word_tokenize(doc)

        for eachWord in words:
            eachWord = ps.stem(eachWord)

            if eachWord in stop_list:
                continue
            if len(eachWord) <= 3:
                continue

            if eachWord not in associationMatrix:
                associationMatrix[eachWord] = {}
                for docId in range(length):
                    associationMatrix[eachWord][docId] = 0
            associationMatrix[eachWord][id] += 1

    return associationMatrix;


def createUnNormalized(associationMatrix,length):

    unNormalizedAssociation = {}
    for key1 in associationMatrix:
        unNormalizedAssociation[key1] = {}
        for key2 in associationMatrix:
            res = 0;
            for each_doc in range(length):
                res+=associationMatrix[key1][each_doc]*associationMatrix[key2][each_doc]

            unNormalizedAssociation[key1][key2] = res

    return unNormalizedAssociation;

def normalizedAssocaition(unNormalized):

    normalized = {}
    for key1 in unNormalized:
        normalized[key1] = {}
        for key2 in unNormalized:
            res = unNormalized[key1][key2]/(unNormalized[key1][key1]+unNormalized[key1][key2]+unNormalized[key2][key2])
            normalized[key1][key2] = res

    return normalized

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

def getExpandedQuery(query,data):

    ps = PorterStemmer()
    documents = getDocuments(data)
    docAfterProcessing = processDocuments(documents); length = len(docAfterProcessing)
    associationMatrix = makeAssociationMatrix(docAfterProcessing,ps,length)
    unNormalizedAssociation = createUnNormalized(associationMatrix, length)
    normalizedAssociation = normalizedAssocaition(unNormalizedAssociation)
    expandedQuery = getFinalQuery(normalizedAssociation, query, 3,ps)
    return expandedQuery
