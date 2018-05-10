# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:01:32 2017

@author: ajadhav
"""
import os

import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from matplotlib.pyplot import hist
from nltk import bigrams, pos_tag
from nltk import trigrams
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(('blue','dart'))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

df = pd.read_csv("mouthshut.com.csv", encoding='cp1252')
negDict = pd.read_csv("./dictionaries/negative-words-new.txt",header=None)
negDict.columns = ["NegWords"]
NegWords = list(negDict.NegWords.values.flatten())

negativeWords = []
NegCounter = []
for i in range(len(df)):
    tempNegwords = set(word_tokenize(df.Review[i]))&set(NegWords)
    NegCounter.append(len(tempNegwords))
    negativeWords.append(tempNegwords)

df["negWords"] = negativeWords
df["negCounter"] = NegCounter
  
Threshold = np.median(df.negCounter)
hist(df.negCounter)
df["Polarity"] = "NA"
df["Polarity"][df.negCounter>=Threshold] = "StronglyNegative"
df["Polarity"][df.negCounter<=Threshold] = "Negative"

 
def BiagramFreq(DataFrame): 
    Bigrams =[] 
    posTags = []
    for i in range(len(DataFrame)):
        text = DataFrame['Review'].iloc[i]
        word_tokens = tokenizer.tokenize(text)
        filtered_sentence =   [w for w in word_tokens if not w in stop_words]
        for (w1, t1), (w2, t2), (w3,t3) in trigrams(pos_tag(filtered_sentence)):
            if t1.startswith('JJ') and t2.startswith('NN') and t3.startswith('JJR'):
                if w1 in NegWords:
                    temp1 = w1, w2, w3
                    temp2 = t1, t2, t3
                    Bigrams.append(temp1)
                    posTags.append(temp2)
            
    filterDf = pd.DataFrame(Bigrams,index = posTags, columns = ["First","Second", "Third"])
    finalDf = filterDf[['First', 'Second','Third']].apply(lambda x: ' '.join(x), axis=1)           
    BigramFrequncy = finalDf.iloc[1:].value_counts()
    return BigramFrequncy

Neg = df[df.Polarity== "Negative" ]
NegativeBigramFrequncy = BiagramFreq(Neg)
NegativeBigramFrequncy.to_csv("NegativeBigramFrequncy.csv")

StrongNeg = df[df.Polarity== "StronglyNegative" ]
StrongNegativeBigramFrequncy = BiagramFreq(StrongNeg)
StrongNegativeBigramFrequncy.to_csv("StrongNegativeBigramFrequncy.csv")
