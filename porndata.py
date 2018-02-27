
# coding: utf-8

# In[ ]:


import pandas as pd

data = pd.read_csv('/Users/Lwmformula/Downloads/porndata/porndata.csv')
cat = data['categories']
tags = data['tags']
title = data['title']
views = data['nb_views']


# In[ ]:


from string import strip as trim
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

lmtzr = WordNetLemmatizer()
stop = set(stopwords.words('english'))

def strchange(key):
    return str(key)

def trimupperfunc(key):
    return str(trim(re.sub("[^a-zA-Z\d]", " ",key)).lower())

def split_underline(key):
    tmp = key.split('__')
    tmp2 = map(trimupperfunc,tmp)
    return tmp2

def wordprocess2(key):
    tmp = key.split('__')
    tmp2 = map(trimupperfunc,tmp)
    return tmp2

def title_processer(t):
    l = re.sub("[^a-zA-z]"," ",t)
    tokens = nltk.word_tokenize(l)
    lower = [l.lower() for l in tokens]
    return lower
    #lemmas = [lmtzr.lemmatize(i) for i in filtered]
    #return (' '.join(lemmas))


# In[ ]:


title_p = list(map(title_processer,title))
title_pred = []
for i in title_p:
    for j in i:
        title_pred.append(j)
        #title_pred.append(lmtzr.lemmatize(j,'n'))
title_count = Counter(title_pred)
freq = dict(title_count.most_common(20))
wordcloud = WordCloud(width=1500,height=1000, max_words=100,relative_scaling=1,max_font_size=500,
                      normalize_plurals=False).generate_from_frequencies(freq)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/Lwmformula/Downloads/porndata/title.png')
plt.show()

key = []
value = []
for key_,value_ in freq.items():
    key.append(key_)
    value.append(value_)
y_pos = np.arange(len(key))
plt.figure(figsize=(20, 3))
plt.bar(y_pos, value, align='center', alpha=0.5, width=0.8)
plt.xticks(y_pos,key)
plt.ylabel('keywords')
plt.title("__Title Keywords__")
plt.savefig('/Users/Lwmformula/Downloads/porndata/title_bar.png')
plt.show()
print freq


# In[ ]:


tags_pred = []
tagpre = map(split_underline,map(strchange,tags))
for i in tagpre:
    for j in i:
        tags_pred.append(j)
        #tags_pred.append(lmtzr.lemmatize(j,'n'))
tags_count = Counter(tags_pred)
freq = dict(tags_count.most_common(20))
wordcloud = WordCloud(width=1500,height=1000, max_words=100,relative_scaling=1,max_font_size=500,
                      normalize_plurals=False).generate_from_frequencies(freq)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/Lwmformula/Downloads/porndata/tags.png')
plt.show()

key = []
value = []
for key_,value_ in freq.items():
    key.append(key_)
    value.append(value_)
y_pos = np.arange(len(key))
plt.figure(figsize=(20, 3))
plt.bar(y_pos, value, align='center', alpha=0.5, width=0.8)
plt.xticks(y_pos,key)
plt.ylabel('keywords')
plt.title("__Tags Keywords__")
plt.savefig('/Users/Lwmformula/Downloads/porndata/tags_bar.png')
plt.show()
print freq


# In[ ]:


cat_pred = []
catpre = map(split_underline,map(strchange,cat))
for i in catpre:
    for j in i:
        cat_pred.append(j)
        #cat_pred.append(lmtzr.lemmatize(j,'n'))
cat_count = Counter(cat_pred)
freq = dict(cat_count.most_common(20))
wordcloud = WordCloud(width=1500,height=1000, max_words=100,relative_scaling=1,max_font_size=500,
                      normalize_plurals=False).generate_from_frequencies(freq)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/Lwmformula/Downloads/porndata/cat.png')
plt.show()

key = []
value = []
for key_,value_ in freq.items():
    key.append(key_)
    value.append(value_)
y_pos = np.arange(len(key))
plt.figure(figsize=(20, 3))
plt.bar(y_pos, value, align='center', alpha=0.5, width=0.8)
plt.xticks(y_pos,key)
plt.ylabel('keywords')
plt.title("__Category Keywords__")
plt.savefig('/Users/Lwmformula/Downloads/porndata/Category_bar.png')
plt.show()
print freq


# In[ ]:


import operator

title_p = list(map(title_processer,title))
countdict_t = {}
for i in range(len(title_p)):
    for j in title_p[i]:
        try:
            tmp = int(countdict[j])
            tmp2 = tmp + int(views[i])
            countdict_t.update({j:tmp2})
        except:
            countdict_t.update({j:int(views[i])})
top20 = dict(sorted(countdict_t.iteritems(), key=operator.itemgetter(1), reverse=True)[:20])
wordcloud = WordCloud(width=1500,height=1000, max_words=100,relative_scaling=1,max_font_size=500,
                      normalize_plurals=False).generate_from_frequencies(top20)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/Lwmformula/Downloads/porndata/title_views.png')
plt.show()

key = []
value = []
for key_,value_ in top20.items():
    key.append(key_)
    value.append(value_)
y_pos = np.arange(len(key))
plt.figure(figsize=(20, 3))
plt.bar(y_pos, value, align='center', alpha=0.5, width=0.8)
plt.xticks(y_pos,key)
plt.ylabel('keywords')
plt.title("__Title Keywords__views")
plt.savefig('/Users/Lwmformula/Downloads/porndata/title_bar_views.png')
plt.show()
print top20


# In[ ]:


tagpre = map(split_underline,map(strchange,tags))
countdict_tag = {}
for i in range(len(tagpre)):
    for j in tagpre[i]:
        try:
            tmp = int(countdict[j])
            tmp2 = tmp + int(views[i])
            countdict_tag.update({j:tmp2})
        except:
            countdict_tag.update({j:int(views[i])})
top20 = dict(sorted(countdict_tag.iteritems(), key=operator.itemgetter(1), reverse=True)[:20])
wordcloud = WordCloud(width=1500,height=1000, max_words=100,relative_scaling=1,max_font_size=500,
                      normalize_plurals=False).generate_from_frequencies(top20)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/Lwmformula/Downloads/porndata/tags_views.png')
plt.show()

key = []
value = []
for key_,value_ in top20.items():
    key.append(key_)
    value.append(value_)
y_pos = np.arange(len(key))
plt.figure(figsize=(20, 3))
plt.bar(y_pos, value, align='center', alpha=0.5, width=0.8)
plt.xticks(y_pos,key)
plt.ylabel('keywords')
plt.title("__Tags Keywords__views")
plt.savefig('/Users/Lwmformula/Downloads/porndata/tags_bar_views.png')
plt.show()
print top20


# In[ ]:


catpre = map(split_underline,map(strchange,cat))
countdict_cat = {}
for i in range(len(title_p)):
    for j in catpre[i]:
        try:
            tmp = int(countdict[j])
            tmp2 = tmp + int(views[i])
            countdict_cat.update({j:tmp2})
        except:
            countdict_cat.update({j:int(views[i])})
top20 = dict(sorted(countdict_cat.iteritems(), key=operator.itemgetter(1), reverse=True)[:20])
wordcloud = WordCloud(width=1500,height=1000, max_words=100,relative_scaling=1,max_font_size=500,
                      normalize_plurals=False).generate_from_frequencies(top20)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/Lwmformula/Downloads/porndata/cat_views.png')
plt.show()

key = []
value = []
for key_,value_ in top20.items():
    key.append(key_)
    value.append(value_)
y_pos = np.arange(len(key))
plt.figure(figsize=(20, 3))
plt.bar(y_pos, value, align='center', alpha=0.5, width=0.8)
plt.xticks(y_pos,key)
plt.ylabel('keywords')
plt.title("__Category Keywords__views")
plt.savefig('/Users/Lwmformula/Downloads/porndata/cat_bar_views.png')
plt.show()
print top20


# In[ ]:


concern = ['chile','rape','cumshot','incest','drug','family','harassment','sibling']

concern_dict_occ_tag = {'chile':0,'rape':0,'cumshot':0,'incest':0,'drug':0,
                    'family':0,'harassment':0,'sibling':0}
concern_dict_occ_title = {'chile':0,'rape':0,'cumshot':0,'incest':0,'drug':0,
                          'family':0,'harassment':0,'sibling':0}
concern_dict_view_tag = {'chile':0,'rape':0,'cumshot':0,'incest':0,'drug':0,
                         'family':0,'harassment':0,'sibling':0}
concern_dict_view_title = {'chile':0,'rape':0,'cumshot':0,'incest':0,'drug':0,
                           'family':0,'harassment':0,'sibling':0}

for i in concern:
    for j in tags_count:
        if j.find(i) != -1: 
            tmp = concern_dict_occ_tag[i]
            tmp2 = tmp + tags_count[j]
            concern_dict_occ_tag.update({i:tmp2})
        else: continue
    for j in countdict_tag:
        if j.find(i) != -1: 
            tmp = concern_dict_view_tag[i]
            tmp2 = tmp + countdict_tag[j]
            concern_dict_view_tag.update({i:tmp2})
        else: continue
    for j in title_count:
        if j.find(i) != -1: 
            tmp = concern_dict_occ_title[i]
            tmp2 = tmp + title_count[j]
            concern_dict_occ_title.update({i:tmp2})
        else: continue
    for j in countdict_t:
        if j.find(i) != -1: 
            tmp = concern_dict_view_title[i]
            tmp2 = tmp + countdict_t[j]
            concern_dict_view_title.update({i:tmp2})
        else: continue           
            
print concern_dict_occ_tag
print concern_dict_occ_title
print concern_dict_view_tag
print concern_dict_view_title

