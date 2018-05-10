# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:30:40 2017

@author: ajadhav
"""

import pandas as pd
from sklearn import preprocessing
import seaborn as sns       
import os

NegWeightage = 0.3
StrongNegWeightage = 0.7 

mappingDf = pd.read_csv("./input_files/topic-mapping.csv") 
mappingDf = mappingDf.drop_duplicates()


NegativeDf = pd.read_csv("./output/NegativeBigramFrequncy.csv", encoding='cp1252') 
StrongNegativeDf = pd.read_csv("./output/StrongNegativeBigramFrequncy.csv", encoding='cp1252' ) 
MonthlyBigramCounterDf = pd.read_csv("./output/MonthlyBigramCounter.csv", encoding='cp1252' ) 

#MonthlyBigramCounterDf.Bigrams= MonthlyBigramCounterDf.Bigrams.str.upper()
#MonthlyBigramCounterDf.Bigrams = MonthlyBigramCounterDf.Bigrams.str.strip().str.replace(' ', '_')
#
#mappingDf.Bigrams= mappingDf.Bigrams.str.upper()
#mappingDf.Bigrams = mappingDf.Bigrams.str.strip().str.replace(' ', '_')


Supspecious = ['Bigrams','Polarity','Month','Year','Courier','count']
mergedMonthlyCounter = pd.merge(MonthlyBigramCounterDf,mappingDf,on ='Bigrams',how = 'left')
mergedMonthlyCounter.drop_duplicates(subset=Supspecious, keep='first', inplace=True)
#Step no 5 Output
mergedMonthlyCounter.to_csv("./output/TopicBigramMonthlyCounter.csv",index=False)

Supspecious2 = ['Bigrams','Review','PosTags','Polarity','Courier','Date']

mergedDf = pd.merge(NegativeDf,mappingDf,on ='Bigrams',how = 'left')
#mergedDf.drop_duplicates(subset=Supspecious2, keep='first', inplace=True)
mergedDf.NegCount =mergedDf.NegCount * NegWeightage

mergedDf2 = pd.merge(StrongNegativeDf,mappingDf,on ='Bigrams',how = 'left')
#mergedDf2.drop_duplicates(subset=Supspecious2, keep='first', inplace=True)
mergedDf2.StrongNegCount =mergedDf2.StrongNegCount * StrongNegWeightage

frames = [mergedDf,mergedDf2]
finalDf = pd.concat(frames)
finalDf["TotalCount"] = finalDf.NegCount.fillna(0) + finalDf.StrongNegCount.fillna(0)
finalDf["Month"] = pd.DatetimeIndex(finalDf["Date"]).month
finalDf["Year"] =pd.DatetimeIndex(finalDf["Date"]).year



AggTopicWiseCount = finalDf.groupby(['Topics','Courier','Month'])['TotalCount'].sum().reset_index()         
AggTopicWiseCount.to_csv("./output/AggTopicWiseCount.csv",index =False)
#Step 6
ActualTopicScoreForAllMonths = AggTopicWiseCount.pivot_table(index=['Courier','Topics'], columns='Month', values='TotalCount').reset_index()       
ActualTopicScoreForAllMonths.to_csv("./output/ActualTopicScoreForAllMonths.csv",index =False)


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))

ActualTopicScoreForAllMonths = ActualTopicScoreForAllMonths.fillna(0)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))

import numpy as np
for i in range(len(ActualTopicScoreForAllMonths)):
    ActualTopicScoreForAllMonths.iloc[i:i+1,2:] = np.transpose(min_max_scaler.fit_transform(np.transpose(ActualTopicScoreForAllMonths.iloc[i:i+1,2:].values)))

NormalizeTopicScoreForAllMonths = ActualTopicScoreForAllMonths
NormalizeTopicScoreForAllMonths.to_csv("./output/NormalizeTopicScoreForAllMonths.csv",index =False)
# 


#axs = sns.barplot(y="Topics", x="NormalizeValues", data=finalDf,label = "Score Board of Topics",color="salmon" )      
#axs.set_xlabel("Total Score") 
#axs.set_title('Score Board ')
#fig = axs.get_figure()
#fig.savefig('./output/ScoreBoard.png')    


