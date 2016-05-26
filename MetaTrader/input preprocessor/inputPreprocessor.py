import os
import re
import pandas as pd
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


