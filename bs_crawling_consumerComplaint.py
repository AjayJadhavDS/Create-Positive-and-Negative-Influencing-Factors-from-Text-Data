# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:36:23 2017

@author: ajadhav
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:41:39 2017

@author: ajadhav
"""

import requests 
from bs4 import BeautifulSoup
import pandas as pd
import os
df =[]
df2 =[]
firstPart = "https://www.consumercomplaints.in"

for x in range(1,50):
    testUrl ="https://www.consumercomplaints.in/dtdc-courier-cargo-b100102/page/{0}".format(x)
    soup = BeautifulSoup(requests.get(testUrl).text, "html.parser")
    print(x)
    for title in soup.select("a[id^=c]"):
         items = title.get('href')
         if items:
             broth = BeautifulSoup(requests.get(firstPart+items).text, "html.parser")
             item =broth.select("table.w100 td.compl-text")[0]
             item2 =broth.select("table.w100 td.small")[0]
             item2 = item2.text.split(' Submit',-1)[0]+''
             item2 = item2.rpartition('on')[-1]
             df.append(item.text)
             df2.append(item2)
            
comment = pd.DataFrame(df)
dates = pd.DataFrame(df2)

finalDataframe = pd.concat([comment, dates], axis=1)
finalDataframe.columns = ["Review","Date"]
pd.DataFrame.to_csv(finalDataframe,"DTDC-consumercomplaints.csv")


                  
         
