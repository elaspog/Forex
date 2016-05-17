# Input Preprocessor

Processes the csv and txt data imported from: http://www.histdata.com/download-free-forex-data/?/metatrader/1-minute-bar-quotes

Where CSV data format is:

	2000.05.30,17:27,0.930200,0.930200,0.930200,0.930200,0
	2000.05.30,17:35,0.930400,0.930500,0.930400,0.930500,0
	...
	2000.12.29,11:24,0.937500,0.937500,0.937400,0.937400,0
	2000.12.29,11:25,0.937300,0.937400,0.937300,0.937300,0


And text data format is:

	HistData.com (c) 2012
	File: DAT_MT_EURUSD_M1_2000.csv Status Report

	Gap of 442s found between 20000530172736 and 20000530173504.
	Gap of 180s found between 20000530173505 and 20000530173811.
	...
	Gap of 103s found between 20001229110722 and 20001229110911.
	Gap of 115s found between 20001229111754 and 20001229111955.

	Average tick interval: 22435 miliseconds.
	Maximum tick interval found: 9866000 miliseconds.
