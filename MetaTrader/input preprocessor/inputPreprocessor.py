import os
import re
import pandas as pd
import datetime as dt


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


'''    
for index, row in gap_data.iterrows():
    cond1 = candle_data['datetime'] >= row['from']
    cond2 = candle_data['datetime'] <= row['to']
    candle_data.ix[cond1 & cond2, 'is_gap'] = True
    print index
'''

def flagCandleDataWithGaps(candle_data, gap_data):
    c_i = 0
    g_i = 0
    c_len = len(candle_data)
    g_len = len(gap_data)
    while (c_i < c_len) & (g_i < g_len) :
        if candle_data.ix[c_i,'datetime'] < gap_data.ix[g_i,'from']:
            c_i = c_i + 1
            continue
        if candle_data.ix[c_i,'datetime'] > gap_data.ix[g_i,'to']:
            g_i = g_i + 1
            continue
        if (candle_data.ix[c_i,'datetime'] >= gap_data.ix[g_i,'from']) & (candle_data.ix[c_i,'datetime'] <= gap_data.ix[g_i,'to']):
            candle_data.ix[c_i,'is_gap'] = True
            c_i = c_i + 1
        else:
            print('ERROR', candle_data.ix[c_i,'datetime'], gap_data.ix[g_i,'from'], gap_data.ix[g_i,'to'])
        print str(c_i) + "/" + str(c_len) + "\t" + str(g_i) + "/" + str(g_len)
    return candle_data


def readCandleAndGapDataFromCsvAndTxtFiles(dirpath):
    dir_content = os.listdir(dirpath)
    files = (k.split(".")[0] for k in dir_content)
    candle_data = pd.DataFrame()
    gap_data = pd.DataFrame()
    for filePair in sorted(list(set(files))):
        print(filePair)
        
        csv_data = pd.read_csv(dirpath+filePair+".csv", header=None)
        candle_data = pd.concat([candle_data, csv_data], ignore_index=True)
        
        text_data = read_gaps_from_txt_files(dirpath+filePair+".txt")
        gap_data = pd.concat([gap_data, text_data], ignore_index=True)
    
    candle_data.columns=["date", "time", "open", "high", "low", "close", "volume"]    
    candle_data['datetime'] = candle_data['date'].str.replace('.', '') + candle_data['time'].str.replace(':', '') + "00"
    candle_data['is_gap'] = False
    
    return (candle_data, gap_data)
        

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


