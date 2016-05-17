import os
import re
import pandas as pd


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
        
        csv_data = pd.read_csv(dirpath+filePair+".csv", header=None)
        candle_data = pd.concat([candle_data, csv_data], ignore_index=True)
        
        text_data = read_gaps_from_txt_files(dirpath+filePair+".txt")
        gap_data = pd.concat([gap_data, text_data], ignore_index=True)
    
    candle_data.columns=["date", "time", "open", "high", "low", "close", "volume"]    
    candle_data['datetime'] = candle_data['date'].str.replace('.', '') + candle_data['time'].str.replace(':', '') + "00"
    return (candle_data, gap_data)
        
