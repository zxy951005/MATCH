from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import tempfile
import subprocess
import operator
import collections
from collections import Counter

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/home1/coref/stanford-corenlp-full-2018-10-05/')

import json
with open('CRAFT_test1.conti.jsonlines') as f:     #CRAFT_test1.conti.jsonlines
    train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    
dep_doc=[]
for t in train_examples:
    ss=t["sentences"]
    dep=[]
    for s in ss:        
        dep.append(nlp.dependency_parse(" ".join(s)))
    #print(len(dep))
    dep_doc.append(dep)
    
#print(len(dep_doc))
'''
for i,t in enumerate(train_examples):
    ss=t["sentences"]
    print(len(ss),len(dep_doc[i]))
    
    
'''
local=[]
globall=[]
for i,dep in enumerate(dep_doc):  #doc
  d_doc_local=[]
  d_doc_global=[]
  for dp in dep:   #sent
    d_sent_local=[]
    d_sent_global=[]
    #print(dp)   #sent_dp

    for d in dp:  #depedency
      if d[0] =="ROOT":
        d_sent_global.append(d)
        flag=d[2]
    for dd in dp:
      if dd[0] in ["compound","nmod","case","amod","det"]:  #local
        #print(dd)

        d_sent_local.append(dd)
      if dd[1]==flag:
        d_sent_global.append(dd)  
    d_doc_local.append(d_sent_local)
    d_doc_global.append(d_sent_global)
  local.append(d_doc_local)
  globall.append(d_doc_global)
  
'''
for i,t in enumerate(train_examples):
    ss=t["sentences"]
    print(len(ss),len(local[i]),len(globall[i]))
  
'''
print(len(globall))
print(len(local))


new_data = list()
for i, tmp_example in enumerate(train_examples):        
  new_example = tmp_example
  new_example['dep_local'] =local[i]
  new_example['dep_global'] =globall[i]
  new_data.append(new_example)

with open("CRAFT_test.depkb.jsonlines", 'w') as f:
    for tmp_example in new_data:
        f.write(json.dumps(tmp_example))
        f.write('\n')
