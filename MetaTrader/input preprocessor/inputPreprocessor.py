import os
import re
import pandas as pd
import numpy as np
import datetime as dt
from intervaltree import IntervalTree


pattern_string = "Gap of (\d+)s found between (\d+) and (\d+)."



def read_gaps_from_txt_files(path):
    ret_data = pd.DataFrame()
    data_list = []
    with open(path) as f:
        content = f.readlines()
        for line in content:
            groups = re.search(pattern_string, line)
            if groups is not None:
                data_list.append(list(groups.groups()))
    ret_data = pd.DataFrame(data_list, columns=["interval", "from", "to"])
    return ret_data



def readCandleAndGapDataFromCsvAndTxtFiles(dirpath):
    dir_content = os.listdir(dirpath)
    files = (k.split(".")[0] for k in dir_content)
    candle_data = pd.DataFrame()
    gap_data = pd.DataFrame()
    for filePair in sorted(list(set(files))):
        print(filePair)
        csv_data = pd.read_csv(dirpath+filePair+".csv", header=None, dtype=str)
        candle_data = pd.concat([candle_data, csv_data], ignore_index=True)
        text_data = read_gaps_from_txt_files(dirpath+filePair+".txt")
        gap_data = pd.concat([gap_data, text_data], ignore_index=True)
    candle_data.columns=["date", "time", "open", "high", "low", "close", "volume"]    
    candle_data['datetime'] = candle_data['date'].str.replace('.', '') + candle_data['time'].str.replace(':', '') + "00"
    candle_data['is_gap'] = None
    return (candle_data, gap_data)



def cleanUnnecessaryColumns(candle_df):
    candle_df.drop('date', axis=1, inplace=True)
    candle_df.drop('time', axis=1, inplace=True)



def parseDateTimeColumnForDataframe(dataframe, fromCol, toCol):
    if not fromCol in dataframe.columns:
        raise Exception ( "Column " + fromCol + " is not in dataframe.")
    l = None
    if toCol == 'year':
        l = lambda row: row[fromCol][0:4]
    elif toCol == 'month':
        l = lambda row: row[fromCol][4:6]
    elif toCol == 'day':
        l = lambda row: row[fromCol][6:8]
    elif toCol == 'week':
        l = lambda row: dt.datetime(int(row[fromCol][0:4]), int(row[fromCol][4:6]), int(row[fromCol][6:8])).isocalendar()[1]
    elif toCol == 'dayOfWeek':
        l = lambda row: dt.datetime(int(row[fromCol][0:4]), int(row[fromCol][4:6]), int(row[fromCol][6:8])).isocalendar()[2]
    elif toCol == 'dayOfYear':
        l = lambda row: dt.datetime(int(row[fromCol][0:4]), int(row[fromCol][4:6]), int(row[fromCol][6:8])).strftime("%j")
    elif toCol == 'hour':
        l = lambda row: row[fromCol][8:10]
    elif toCol == 'minute':
        l = lambda row: row[fromCol][10:12]
    dataframe[toCol] = dataframe.apply(l, axis=1)



def flagCandleDataWithGaps(candle_data, gap_data):
    gaps = pd.DataFrame()
    candles = pd.DataFrame()
    gaps['from_ts'] = gap_data['from'].astype(float)
    gaps['to_ts'] = gap_data['to'].astype(float)
    gaps['to_ts'] = gaps['to_ts']+0.1
    candles['datetime_ts'] = candle_data['datetime'].astype(float)
    tree = IntervalTree.from_tuples(zip(gaps['from_ts'], gaps['to_ts']))
    col = (tree.overlaps(x) for x in candles['datetime_ts'])
    df = pd.DataFrame(col)
    candle_data['is_gap'] = df[0]



def filterOutliers(candles):
    c = candles
    x_o1 = c[~(np.abs(c.open.astype(float)-c.open.astype(float).mean())<=(3*c.open.astype(float).std()))]
    x_c1 = c[~(np.abs(c.close.astype(float)-c.close.astype(float).mean())<=(3*c.close.astype(float).std()))]
    x_h1 = c[~(np.abs(c.high.astype(float)-c.high.astype(float).mean())<=(3*c.high.astype(float).std()))]
    x_l1 = c[~(np.abs(c.low.astype(float)-c.low.astype(float).mean())<=(3*c.low.astype(float).std()))]
    x_o2 = c[c.open.astype(float) <= 0.0]
    x_c2 = c[c.close.astype(float) <= 0.0]
    x_h2 = c[c.high.astype(float) <= 0.0]
    x_l2 = c[c.low.astype(float) <= 0.0]
    filtered = x_o1.append(x_o2)
    filtered = filtered.append(x_c1).append(x_c2)
    filtered = filtered.append(x_h1).append(x_h2)
    filtered = filtered.append(x_l1).append(x_l2)
    filtered = filtered.drop_duplicates()
    valid = c[~c.isin(filtered)].dropna()
    return valid.reset_index(drop=True), filtered, valid



def filterCorruptedCandles(candles):
    validCandles = candles[candles['is_gap'] == 'False']
    validCandles.drop('is_gap', axis=1, inplace=True)
    return validCandles.reset_index(drop=True)



def createContiguousPartitionsByTimeWithMinPartitionSize(candles, minPartitionSize = -1, processIndicatorStep = 1000):
    max_len = len(candles)
    l = list()
    dataFrameBuffer = None
    for index, row in candles.iterrows():
        if dataFrameBuffer is None:
            dataFrameBuffer = pd.Series.to_frame(row).transpose()
        else:
            col = dataFrameBuffer['datetime'].astype(long)
            df_max = max(col)
            row_max = long(row['datetime'])
            diff = abs(row_max-df_max)
            if (diff <= 100):    # 100 equals to 60s in candle['datetime']
                dataFrameBuffer = dataFrameBuffer.append(row, ignore_index=True)
            else:
                if dataFrameBuffer.shape[0] >= minPartitionSize:
                    l.append(dataFrameBuffer)
                dataFrameBuffer = None
        if index % 1000 == 0:
            print(str(index) + "/" + str(max_len))
    if dataFrameBuffer.shape[0] > minPartitionSize:
        l.append(dataFrameBuffer)
    return l



def writeDataFrame(partitionDataFrame, outDirPath, fileNamePrefix, writeCandleCountToFileName):
    fileName = fileNamePrefix
    column = partitionDataFrame['datetime'].astype(long)
    minVal = min(column)
    maxVal = max(column)
    fileName = fileName + "_" + str(minVal)
    if writeCandleCountToFileName:
        fileName = fileName + "_" + str(partitionDataFrame.shape[0])
    if minVal != maxVal:
        fileName = fileName + "_" + str(maxVal)
    partitionDataFrame.to_csv(os.path.join(outDirPath, fileName+".csv"), index=False, sep=";")



def dataframeListCsvWriter(outDirPath, fileNamePrefix, listOfDataFrames, minPartitionSize = -1,
                           writeCandleCntToFileName = True):
    if not os.path.exists(outDirPath):
        os.makedirs(outDirPath)
    else:
        if not os.path.isdir(outDirPath):
            raise Exception(outDirPath + " already exists, but it's not a directory.")
    for df in listOfDataFrames:
        if df.shape[0] >= minPartitionSize:
            writeDataFrame(df, outDirPath, fileNamePrefix, writeCandleCntToFileName)



def partitionizeAndWriteCsvOnTheFly(outDirPath, fileNamePrefix, candles, minPartitionSize = -1,
                                    writeCandleCntToFileName = True, processIndicatorStep = 1000):
    if not os.path.exists(outDirPath):
        os.makedirs(outDirPath)
    else:
        if not os.path.isdir(outDirPath):
            raise Exception(outDirPath + " already exists, but it's not a directory.")
    max_len = len(candles)
    dataFrameBuffer = None
    for index, row in candles.iterrows():
        if dataFrameBuffer is None:
            dataFrameBuffer = pd.Series.to_frame(row).transpose()
        else:
            col = dataFrameBuffer['datetime'].astype(long)
            df_max = max(col)
            row_max = long(row['datetime'])
            diff = abs(row_max-df_max)
            if (diff <= 100):    # 100 equals to 60s in candle['datetime']
                dataFrameBuffer = dataFrameBuffer.append(row, ignore_index=True)
            else:
                if dataFrameBuffer.shape[0] >= minPartitionSize:
                    writeDataFrame(dataFrameBuffer, outDirPath, fileNamePrefix, writeCandleCntToFileName)
                dataFrameBuffer = None
        if index % processIndicatorStep == 0:
            print(str(index) + "/" + str(max_len))
    if dataFrameBuffer.shape[0] > minPartitionSize:
        writeDataFrame(dataFrameBuffer, outDirPath, fileNamePrefix, writeCandleCntToFileName)



