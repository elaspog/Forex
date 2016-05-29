#!/usr/bin/env python


""" Candle input preprocessor tools

These functions provide preprocessing functionalities to the MetaTrader candle history dataset from:
http://www.histdata.com/download-free-forex-data/?/metatrader/1-minute-bar-quotes
"""

__author__ = "Laszlo Pogany"
__copyright__ = "Copyright 2016, The Small Forex Project (POC)"
__credits__ = ["Laszlo Pogany"]
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Laszlo Pogany"
__status__ = "Development"


import os
import re
import pandas as pd
import numpy as np
import datetime as dt
from intervaltree import IntervalTree



def read_gaps_from_txt_files(path, pattern_string = "Gap of (\d+)s found between (\d+) and (\d+)."):
    """ Gap history file reader.

    Reads the gap history from file, parses the values and creates a dataframe to store the raw values.

    Args:
        path: file path of the txt file describing gaps between candles
        pattern_string: regex pattermns of rows holding the 3 interesting values.
    Returns:
        dataframe of the gaps where rows contain the following valiues:
            (interval_size, from_timestamp, to_timestamp)
    """
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
    """ Candle and Gap history reader.

    Reads and parses all CSV and TXT files of a given directory containig history data of candles and gaps.
    All data in the separate files of the directory is merged into a single dataframe of candles and single dataframe of gaps.
    The columns of the dataframe are labelled according to the input CSV format.
    A joint datetime column and an empty gap column is created.

    Args:
        dirpath: path to the directory containing the CSV and TXT files.
    Returns:
        tuple of two dataframes, containing the raw candle history and gap information.
    Depends on:
        read_gaps_from_txt_files(path, pattern_string)
    """
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
    """ Drops unnecessary columns (time and date) of dataframe containing the raw candle.

    Args:
        candle_df: dataframe of the candle history.
    Returns:
        the dataframe of the candle history.
    """
    candle_df.drop('date', axis=1, inplace=True)
    candle_df.drop('time', axis=1, inplace=True)



def nthOccurenceOfADayOfTheWeekInMonth(day):
    """ Calculates the Nth occurence of a day of the week for a given month.

    Args:
        day: number of the day in month.
    Returns:
        Nth occurence of a day of the week in month.
    """
    actual_day = day
    counter = 0
    while actual_day > 0:
        actual_day = actual_day - 7
        counter = counter + 1
    return str(counter)



def parseDateTimeColumnForDataframe(dataframe, fromCol, toCol):
    """ Parses the relevant data and time informations from the special, joined 'datetime' field of the dataframe.

    Modifies the input dataframe according to the joint 'datetime' column, where 'datetime' is in format: YYYYMMDDhhmmss.
    The function adds additional columns to the dataframe. The new column contains numbers in string type.

    Args:
        dataframe: dataframe, what has a column with a defined 'datetime' format.
        fromCol: the name of the column what contains 'datetime' in format: YYYYMMDDhhmmss.
        toCol: the name of the new column being added to the dataframe.
    Returns:
        The modified version of the input dataframe.
    """
    if not fromCol in dataframe.columns:
        raise Exception ( "Column " + fromCol + " is not in dataframe.")
    l = None
    if toCol == 'year':
        l = lambda row: row[fromCol][0:4]
    elif toCol == 'month':
        l = lambda row: row[fromCol][4:6]
    elif toCol == 'day':
        l = lambda row: row[fromCol][6:8]
    elif toCol == 'hour':
        l = lambda row: row[fromCol][8:10]
    elif toCol == 'minute':
        l = lambda row: row[fromCol][10:12]
    elif toCol == 'week':
        l = lambda row: dt.datetime(int(row[fromCol][0:4]), int(row[fromCol][4:6]), int(row[fromCol][6:8])).isocalendar()[1]
    elif toCol == 'dayOfWeek':
        l = lambda row: dt.datetime(int(row[fromCol][0:4]), int(row[fromCol][4:6]), int(row[fromCol][6:8])).isocalendar()[2]
    elif toCol == 'dayOfYear':
        l = lambda row: dt.datetime(int(row[fromCol][0:4]), int(row[fromCol][4:6]), int(row[fromCol][6:8])).strftime("%j")
    elif toCol == "nthDayOfWeekInMonth":
        l = lambda row: nthOccurenceOfADayOfTheWeekInMonth(int(row['day']))
    dataframe[toCol] = dataframe.apply(l, axis=1)



def flagCandleDataWithGaps(candle_data, gap_data):
    """ Uses the gap information to flag the corrupted candles in the dataframe.

    Function adds an 'is_gap' column to the dataframe.
    True values indicate that the candle is corrupted, False values indicate, that there was no gap percieved that could
    corrupt the candle.

    Args:
        candle_data: dataframe of the candles, with 'datatime' column with format: YYYYMMDDhhmmss.
        gap_data: dataframe of the gaps, with 3 columns: interval_size, from_timestamp, to_timestamp
            where interval_size is given in sec, from_timestamp, to_timestamp has the YYYYMMDDhhmmss format.
    Returns:
        the candle dataframe with an indicator column.
    """
    gaps = pd.DataFrame()
    candles = pd.DataFrame()
    gaps['from_ts'] = gap_data['from'].astype(float)
    gaps['to_ts'] = gap_data['to'].astype(float)
    gaps['to_ts'] = gaps['to_ts']+0.1   # hack
    candles['datetime_ts'] = candle_data['datetime'].astype(float)
    tree = IntervalTree.from_tuples(zip(gaps['from_ts'], gaps['to_ts']))
    col = (tree.overlaps(x) for x in candles['datetime_ts'])
    df = pd.DataFrame(col)
    candle_data['is_gap'] = df[0]



def filterOutliers(candles):
    """ Filters the outliers from the dataset.

    Determines the outliers by using the standard deviation and checks the non-zero condition on prices.

    Args:
        candles: the original candle dataset with columns: 'open', 'close', 'high', 'low'.
    Returns:
        valid: dataframe of correct candles with a new index.
        filtered: dataframe of filtered candles with the original index.
        valid: dataframe of correct candles with the original index.
    """
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
    """ Filters the corrupted candles from the dataset and drops the indicator column of candle corruption.

    Args:
        candles: the candle dataset with filled corruption indicator column: 'is_gap'.
    Returns:
        the filtered and reindexed dataset.
    """
    validCandles = candles[candles['is_gap'] == 'False']
    validCandles.drop('is_gap', axis=1, inplace=True)
    return validCandles.reset_index(drop=True)



def createContiguousPartitionsByTimeWithMinPartitionSize(candles, minPartitionSize = -1, processIndicatorStep = 1000):
    """ Makes partitions from the candle dataset according to the minimal partition size.

    Args:
        candles: dataframe of unpartitioned dataset.
        minPartitionSize: the minimal size of partitions/contiguous candles (lesser partitions are dropped).
        processIndicatorStep: value for indication process (default value = 1000).
    Returns:
        List of datafrmaes with minimal partition size.
    """
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
    """ Writes partitionized dataframes into a file.

    Gets a dataframe containing candles with minimal size of rows and writes it into a file.
    The file can have a prefix in it's name, but datetime and optionaly the partitions size is appended to it too.

    Args:
        partitionDataFrame: dataframe containing candles.
        outDirPath: output directory path.
        fileNamePrefix: prefix of the file name.
        writeCandleCountToFIleName: optional candle count in file's name.
    """
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
    """ Creates a directory for outputs, iterated over the dataframes and writes them to disk.

    Args:
        outDirPath: path of the output directory.
        fileNamePrefix: file name prefix used by writeDataFrame(...).
        listOfDataFrames: dataframes to be written to the disk.
        minPartitionSize: for checking partitions, only partitions with more candles than min value will be written.
        writeCandleCntToFileName: optional name paramter used by writeDataFrame(...).
    Depends on:
        writeDataFrame(partitionDataFrame, outDirPath, fileNamePrefix, writeCandleCountToFileName).
    """
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
    """ Makes partitions from candle dataset and writes them to the disk.

    Does not require a list of dataframes stored in the memory.
    It iterates over the dataframe, searches for contiguous partitions larger than the minimal size,
    creates dataframes on the fly and writes them to disk.

    Args:
        outDirPath: path of the output directory.
        fileNamePrefix: file name prefix used by writeDataFrame(...).
        candles: dataframe of unpartitioned dataset.
        minPartitionSize: for checking partitions, only partitions with more candles than min value will be written.
        writeCandleCntToFileName: optional name paramter used by writeDataFrame(...).
        processIndicatorStep:  value for indication process (default value = 1000).
    """
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



