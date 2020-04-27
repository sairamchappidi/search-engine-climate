from django.shortcuts import render
from django.http import HttpResponse
from googleapiclient.discovery import build
# Create your views here.
# Create your views here.
import json
import sklearn
import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import pandas as pd
from .hits import hits

from .Services import AssociativeClustering
from .Services import MetricClustering
from .Services import ScalarClustering
from django.http import JsonResponse
import os
import json

import requests
subscription_key = "7ba2665d89574bed9b0e887c41b882a5"
assert subscription_key

search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
search_term=""
activeTab="custom"
hits.get_url_map()
hits.get_adj_lis()
cwd = os.getcwd()

class clusterModel():
    model = joblib.load(cwd + '/Climate_App/clustering_model_zip.pkl')
    vectorizer = joblib.load(cwd + '/Climate_App/vectorizer_zip.pkl')

clusterData = clusterModel()


def getClimateData(request):

    return render(request, 'Climate_App/index.html')

def getResults(payload={}):
    global search_term
    search_term_list = search_term.split(" ")
    search_term_query = ""
    search_term_list_length = len(search_term_list)
    for i in range(0, search_term_list_length):
        search_term_query += search_term_list[i]
        if (search_term_list_length - 1) != i:
            search_term_query += " AND "
        search_term_query += search_term_list[i]
    url = "http://ec2-35-171-122-69.compute-1.amazonaws.com:8983/solr/nutch/select?q=content: (" + str(search_term) + ") OR title: (" + str(search_term) + ") OR id: (" + str(search_term) + ")"
    response = requests.get(url, params=payload)
    search_results = response.json()
    return search_results

def getGoogleResults(request):
     global activeTab
     my_api_key = "AIzaSyDbiy21NBP13V3FzvRe1LaF5E0UmwwfuPo"
     my_cse_id = "014871123708314990102:5dujllnfiee"
     service = build("customsearch", "v1", developerKey=my_api_key)
     res = service.cse().list(q=search_term, cx=my_cse_id).execute()
     activeTab = "google"
     return render(request, 'Climate_App/googleResults.html', {"google": json.dumps(res), "activeTab" : activeTab, "search_term" : search_term})

def getAssociativeExpansion(request):

    search_results = getResults()
    expandedQuery = AssociativeClustering.getExpandedQuery(search_term,search_results)
    return JsonResponse({"query":search_term,"expandedQuery":expandedQuery},safe=False)

def getMetricExpansion(request):

    search_results = getResults()
    expandedQuery = MetricClustering.getExpandedQuery(search_term, search_results)
    return JsonResponse({"query":search_term,"expandedQuery":expandedQuery},safe=False)

def getScalarExpansion(request):

    search_results = getResults()
    expandedQuery = ScalarClustering.getExpandedQuery(search_term, search_results)
    return JsonResponse({"query":search_term,"expandedQuery":expandedQuery},safe=False)

def getCustomResults(request):
    global activeTab
    activeTab = "custom"
    search_term=getSearchQuery(request)
    search_results = getResults()
    return render(request, 'Climate_App/customResults.html', {"results": search_results, "activeTab" : activeTab, "search_term" : search_term})

def getBingResults(request):
    global activeTab
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    y = json.dumps(search_results)
    activeTab="bing"
    return render(request, 'Climate_App/bingResults.html', {"bing": y,"activeTab" : activeTab, "search_term" : search_term})

def clusterResults(search_results):
    docs = search_results['response']['docs']
    search = clusterData.vectorizer.transform([search_term])
    results = clusterData.model.predict(search)
    final_res = []
    cluster_results = []
    normal_results = []
    urls = pd.read_csv(cwd + '/Climate_App/Clustered_results_final.csv')
    clusters = urls.cluster
    for i, val in enumerate(clusters.values):
        if val == results:
            final_res.append(urls['id'].values[i])
    for i in range(0, len(docs)):
        if docs[i]['url'] in final_res:
            cluster_results.append(docs[i])
        else:
            normal_results.append(docs[i])
    cluster_results.extend(normal_results)
    return cluster_results

def getClusterResults(request):
    global activeTab
    activeTab="cluster"
    global search_term
    search_results = getResults()
    cluster_results = clusterResults(search_results)
    return render(request, 'Climate_App/clusterResults.html', {"results": cluster_results,"activeTab" : activeTab, "search_term" : search_term})

def getAgglomerativeResults(request):
    global activeTab
    activeTab = "agglomerative"
    global search_term
    payload = {
        'start': 5,
        'rows': 10
    }
    search_results = getResults(payload)
    cluster_results = clusterResults(search_results)

    return render(request, 'Climate_App/agglomerativeResults.html',
                  {"results": cluster_results, "activeTab": activeTab, "search_term": search_term})


def getQueryExpansionResults(request):
    global activeTab
    activeTab="queryExpansion"
    return render(request, 'Climate_App/queryExpansionResults.html', {"activeTab" : activeTab, "search_term" : search_term})


def getSearchQuery(request):
    global search_term
    if request.method == 'GET' and 'searchResult' in request.GET:
        search_term = request.GET['searchResult']
    return search_term


def getHitsResults(request):
    global activeTab
    activeTab="hits"
    results = getResults()
    search_results = hits.get_hits(results)
    return render(request, 'Climate_App/hitsResults.html', { "results": search_results, "activeTab" : activeTab, "search_term" : search_term})

