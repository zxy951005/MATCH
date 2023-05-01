# encoding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import operator
import math
import json
import numpy as np

from collections import Counter
import random
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/home1/coref/stanford-corenlp-full-2018-10-05/')

#import nlpnet
#tagger = nlpnet.SRLTagger()

def flatten(l):
    return [item for sublist in l for item in sublist]
    
    
doc_example=[]
with open('/home1/coref/research/knowledge_large/3_Knowledge+/PMID/test4.conll') as f:
    for jsonline in f.readlines():
        doc_example.append(jsonline )    
    
start=[]
for i,l in enumerate(doc_example):
    if l.startswith("#begin"):
        start.append(i)

  
ds=[]
for j in range(len(start)):
    if j !=len(start)-1:    
        b=start[j]
        e=start[j+1]
        #print(b,e)
        ds.append(doc_example[b:e])    
    else:
        b=start[j]
        e=-1
        #print(b,e)
        ds.append(doc_example[b:e])    
         
    
for i,d in enumerate(ds):
    del ds[i][0]
    del ds[i][-1]
    del ds[i][-1]
    
  
dd_doc=[]
for i,d in enumerate(ds):
    dd=[]
    for j,l in enumerate(d):
        #print(l)
        if l!="\n":
            dd.append(l)
    dd_doc.append(dd)    
    

        
sindex_doc=[]
for docc in dd_doc:
    sindex=[]
    for i,ll in enumerate(docc):
        if ll.split(" ")[1]=="0":
            sindex.append(i)
    sindex_doc.append(sindex)

'''      
for i,ss in enumerate(sindex_doc):
    for j in sindex_doc[i]:
        print(j)
        print(dd_doc[i][j])    
'''        
 
with open('test4.conll.jsonlines') as f:
    train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
        

       
cs=[]
for t in train_examples:
    c=t["clusters"]
    c_fla=flatten(c)   
    cs.append(c_fla)
    
dep_doc=[]
for t in train_examples:
    ss=t["sentences"]
    dep=[]
    for s in ss:        
        dep.append(nlp.dependency_parse(" ".join(s)))
    dep_doc.append(dep)

 
depg_all=[]
depl_all=[]
for i,m_d in enumerate(cs):
    for m in m_d:
        #print(i)
        mb=m[0]
        me=m[1]
        print(mb,me)
        s_f=flatten(train_examples[i]["sentences"])
        for j,ins in enumerate(sindex_doc[i]):
            #print(ins)
            if me<=ins:
                sstart=sindex_doc[i][j-1]
                insb=mb-sstart+1
                inse=me-sstart+1
                s=train_examples[i]["sentences"][j-1]
                depp=dep_doc[i][j-1]         #sentence index
                #print(insb,inse) 
                #print(s)
                dep_l=[]
                dep_g=[]
                for dp in depp:
                    if insb<=dp[1]<=inse or insb<=dp[2]<=inse:   #local
                        dep_l.append((dp))
                        depl_all.append(dp[0])
                    if dp[1]==0 :
                        root=dp[2]
                for dp in depp:
                    if dp[1]==root or dp[2]==root:         #global
                        dep_g.append(dp)  
                        depg_all.append(dp[0])
                #print(dep_l)
                #print(dep_g)
                break
        
print(len(depl_all))    

result=Counter(depl_all)
dic=(dict(result))
dictt= sorted(dic.items(), key=lambda d:d[1], reverse = True)
print(dictt)
