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
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(('blue','dart'))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#from scipy.misc import imread

df = pd.read_csv("./input_files/AllCombinedData.csv", encoding='cp1252')
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

newdf = pd.DataFrame()
UniqueCouriers = df.Courier.unique()
for i in range(len(UniqueCouriers)):
    dfName = "df_{}".format(i)
    dfName = df[df.Courier== UniqueCouriers[i]]
    Threshold = np.median(dfName.negCounter)
    dfName.negCounter.hist()
    dfName["Polarity"] = "NA"
    dfName.loc[:,'Polarity'][dfName.negCounter>Threshold] = "StronglyNegative"
    dfName.loc[:,'Polarity'][dfName.negCounter<=Threshold] = "Negative"
    newdf = pd.concat([newdf,dfName])

newdf= newdf[newdf.negCounter!=0]    
newdf.to_csv("./output/SentimentWithCounter.csv",index =False)

def BiagramFreq(DataFrame):
    Bigrams =[] 
    posTags = []
    dateTemp =[]
    CourierTemp = []
    ReviewTemp =[]
    PolarityTemp = []
    for i in range(len(DataFrame)):
        text = DataFrame['Review'].iloc[i].lower()
        word_tokens = tokenizer.tokenize(text)
        filtered_sentence =   [w for w in word_tokens if not w in stop_words]
        for (w1, t1), (w2, t2) in bigrams(pos_tag(filtered_sentence)):
            if t1.startswith('JJ') and t2.startswith('NN') or t1.startswith('RB') and t2.startswith('VB') or t1.startswith('RB') and t2.startswith('JJ') or t1.startswith('JJ') and t2.startswith('JJ'):
                if w1 in NegWords:
                    temp1 = w1, w2
                    temp2 = t1, t2
                    ReviewTemp.append(DataFrame['Review'].iloc[i])
                    PolarityTemp.append(DataFrame['Polarity'].iloc[i])
                    Bigrams.append(temp1)
                    posTags.append(temp2)
                    dateTemp.append(DataFrame['Date'].iloc[i])
                    CourierTemp.append(DataFrame['Courier'].iloc[i])
                    
    filterDf = pd.DataFrame({'Bigrams': Bigrams,'PosTags': posTags })    
    filterDf["Review"] = ReviewTemp
    filterDf["Polarity"] = PolarityTemp
    filterDf["Date"] = dateTemp
    filterDf["Courier"] = CourierTemp
    filterDf['Bigrams'] = filterDf['Bigrams'].apply(lambda x: ' '.join(x))
    return filterDf

Neg = newdf[newdf.Polarity== "Negative" ]
NegativeBigramFrequncy = BiagramFreq(Neg)
NegativeBigramFrequncy["NegCount"] = 1
NegativeBigramFrequncy.to_csv("./output/NegativeBigramFrequncy.csv", index=False)

StrongNeg = newdf[newdf.Polarity== "StronglyNegative" ]
StrongNegativeBigramFrequncy = BiagramFreq(StrongNeg)
StrongNegativeBigramFrequncy["StrongNegCount"] = 1
StrongNegativeBigramFrequncy.to_csv("./output/StrongNegativeBigramFrequncy.csv",index=False)
#---------------------------------------------------                       
#                   Step -3
#---------------------------------------------------
NegativeBigramFrequncy["Month"] = pd.DatetimeIndex(NegativeBigramFrequncy["Date"]).month
NegativeBigramFrequncy["Year"] =pd.DatetimeIndex(NegativeBigramFrequncy["Date"]).year

StrongNegativeBigramFrequncy["Month"] = pd.DatetimeIndex(StrongNegativeBigramFrequncy["Date"]).month
StrongNegativeBigramFrequncy["Year"] =pd.DatetimeIndex(StrongNegativeBigramFrequncy["Date"]).year

frames = [NegativeBigramFrequncy, StrongNegativeBigramFrequncy]
result = pd.concat(frames)                            

finalResult = result.groupby(['Review','Polarity','Courier','Month','Year']).Bigrams.unique().reset_index()      
finalResult["BigramCount"]=finalResult.Bigrams.apply(len)
finalResult.to_csv("./output/BigramCounter.csv",index=False)
#---------------------------------------------------     

# Step - 4
result2 = result[["Bigrams","Month"]]
finalResult2 = result.groupby(['Bigrams','Month','Polarity','Courier','Year']).size().reset_index().rename(columns={0:'count'})
finalResult2.to_csv("./output/MonthlyBigramCounter.csv",index=False)


