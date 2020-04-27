from .PreProcessor import getDocuments
from .PreProcessor import processDocuments
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
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

def makeStems(wordsDictionary,ps):

    cwd = os.getcwd()
    with open(cwd + "/Climate_App/Services/stopwords", "r") as f:
        stop_list = set(ps.stem(line.rstrip('\n')) for line in f)
    stemsDictionary = {}

    for each_word in wordsDictionary:

        stem = ps.stem(each_word)

        if stem in stop_list:
            continue
        if stem not in stemsDictionary:
            stemsDictionary[stem] = set()

        stemsDictionary[stem].add(each_word)

    return stemsDictionary

def unNormalizedMetricClustering(stemsDictionary,wordsDictionary):

    unNormalizedMatrix = {}
    for stem1 in stemsDictionary:
        unNormalizedMatrix[stem1] = {}
        for stem2 in stemsDictionary:

            if stem1==stem2:
                continue

            res = 0;

            set1 = stemsDictionary[stem1]
            set2 = stemsDictionary[stem2]

            for each_word1 in set1:
                for each_word2 in set2:

                    wordMap1 = wordsDictionary[each_word1]
                    wordMap2 = wordsDictionary[each_word2]

                    for each_doc in wordMap1:
                        if each_doc in wordMap2:
                            res+=1/abs(wordMap1[each_doc]-wordMap2[each_doc])

            unNormalizedMatrix[stem1][stem2] = res

    return unNormalizedMatrix

def getNormalizedClustering(unNormalized,stemDictionary):

    normalized = {}

    for each_stem1 in unNormalized:
        normalized[each_stem1] = {}
        for each_stem2 in unNormalized:
            if each_stem1 == each_stem2:
                continue;
            normalized[each_stem1][each_stem2] = unNormalized[each_stem1][each_stem2]/(len(stemDictionary[each_stem1])*len(stemDictionary[each_stem2]))

    return normalized;

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
                if new_stem=="“global":
                    continue
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
    docAfterProcessing = processDocuments(documents);length = len(docAfterProcessing)
    words = wordsDictionary(docAfterProcessing)
    stemsDictionary = makeStems(words,ps)
    unNormalized = unNormalizedMetricClustering(stemsDictionary, words)
    normalized = getNormalizedClustering(unNormalized, stemsDictionary)
    query = getFinalQuery(normalized, query, 3, ps)
    return query
