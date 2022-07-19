# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 20:18:37 2022

@author: leigh
"""


import pandas as pd

import glob
import csv
import os

csv_file_path = 'D:/stocks data/NASDAQMomentumPreds'

csv_file_path_2 = 'D:/stocks data/NASDAQpctChange'

temp_df = pd.DataFrame()
preds_df = pd.DataFrame()
i = 0
for file in glob.glob(csv_file_path + '/*.csv'):
    #if (i < 10):
        #i = i + 1
        with open(file, 'r') as f:
            #print(f.name)
            if f.name != "D:/stocks data/NASDAQMomentumPreds\HCCHU": 
                if f.name.count("HCCHU") == 0:
                    temp_df = pd.read_csv(f.name)
                    temp_df.columns = ["index", f.name.replace('D:/stocks data/NASDAQMomentumPreds\\', "").replace(".csv", ""), "target"]
                    temp_df.set_index("index",inplace = True) 
                    temp_df.drop("target", axis=1, inplace=True)
                    temp_df[f.name.replace('D:/stocks data/NASDAQMomentumPreds\\', "").replace(".csv", "")] =  temp_df[f.name.replace('D:/stocks data/NASDAQMomentumPreds\\', "").replace(".csv", "")].apply(lambda x: x.replace('[','').replace(']','')) 
                    temp_df[f.name.replace('D:/stocks data/NASDAQMomentumPreds\\', "").replace(".csv", "")] =  temp_df[f.name.replace('D:/stocks data/NASDAQMomentumPreds\\', "").replace(".csv", "")].astype(float) 
                    
                    #print(temp_df)
                    if len(preds_df)==0:  # if the dataframe is empty
                       preds_df = temp_df  # then it's just the current df
                    else: 
                        preds_df = pd.merge(preds_df, temp_df, on='index', how='outer')
                        #preds_df = pd.concat([preds_df,temp_df], ignore_index=True, inplace=axis=1)
                        #preds_df = preds_df.join(temp_df)

print("finished building preds df")
print(preds_df)

temp_df = pd.DataFrame()
date_df = pd.DataFrame()
i = 0
for file in glob.glob(csv_file_path_2 + '/*.csv'):
    #if (i < 10):
        #i = i + 1
        with open(file, 'r') as f:
            if f.name.count("HCCHU") == 0:
                temp_df = pd.read_csv(f.name)
                temp_df.columns = ["date " + f.name.replace('D:/stocks data/NASDAQpctChange\\', "").replace(".csv", ""), "pct_change " + f.name.replace('D:/stocks data/NASDAQpctChange\\', "").replace(".csv", "")]
                #print(temp_df)
                if len(date_df)==0:  # if the dataframe is empty
                    date_df = temp_df  # then it's just the current df
                else: 
                    #date_df = pd.merge(date_df, temp_df, on='index', how='outer')
                    #pd.concat([date_df,temp_df], ignore_index=True, axis=1)
                    date_df = date_df.join(temp_df, how='outer')
        
print("finished building date df")
print(date_df)

date_df.drop(date_df.index[0:29], inplace=True)
date_df.reset_index(inplace=True)
date_df.drop("index", axis=1, inplace=True)

print(date_df)

"""for date in date_df["date AACG"].values:
    for (columnName, columnData) in date_df.iteritems():
        if "date" in columnName:
           for index, row in date_df.iterrows():
                print(row["date AAME"])
                print(index)"""
                
print("dropped first 30")

#longest = 0
#longestName = ""
#for columnName, columnData in date_df.iteritems():
#    if "date " in columnName:
        #print("date found")
#        if len(columnData.values) > longest:
            #print("new longest")
#            longest = len(columnData.values)
#            longestName = columnName
            
#print("Longest Name: " + longestName)
            
totalPctChange = 0
pctChange = 0
checkpoint = False
datepoint = ""
with open('D:/stocks data/NASDAQMomentumProfit.csv', 'r') as file:
    fileReader = csv.reader(file)
    for date in date_df["date ZYNE"].values: #iterate through dates of the sample with that contains all the dates other could contain
        if checkpoint == True:
            largestPred = 0 
            print(date)
            for (columnName, columnData) in date_df.iteritems(): #iterate through the columns of the date df
                if "date" in columnName:
                    for index, row in date_df.iterrows(): #iterate through the rows of the date df
                        #print(index)
                        #print(date)
                        #print(row[columnName])
                        if date == row[columnName]: #find each date equivalent to sample date
                            #print("index: " + str(index))
                            pred = preds_df.at[index, columnName.replace("date ", "")] #find each prediction at each date point
                            #print("pred: " + str(pred))
                            if (pred > largestPred).any(): #if the prediction is the largest
                                largestPred = pred #save it
                                pctChange = date_df.at[index, columnName.replace("date ", "pct_change ")] #save the pct change at the largest pred point
                                #print("pct_change: " + str(pctChange) + " pct change check")
                                #if (pctChange != preds_df.at[index, "target " + columnName.replace("date ", "")]):
                                #    print("pctChange does not match")
            totalPctChange = totalPctChange + pctChange
            print("total percent change: " + str(totalPctChange))
            with open('D:/stocks data/NASDAQMomentumProfit.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow([date, str(totalPctChange)])
        if checkpoint == False:
            for row in fileReader:
                print(row)
                if (row[0].count("2") > 0):
                    print("found a 2")
                    print(date)
                    if (date == row[0]):
                        print("date matched")
                        checkpoint = True
                        totalPctChange = float(row[1])
            
print("total percent change: " + str(totalPctChange))
            
            
            
            
            
            
            
            
            
            
            
            
            
            