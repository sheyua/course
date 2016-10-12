import pandas as pd
import datetime
import pickle
import time
import os

def restart_syms_min(syms=['NVDA'], loc='./data/min', num_day=15, liq_thr=0.95, sleep_time=1, num_minute = 391):
	# loop over symbols
	for sym in syms:
		# sleep before making the next urlrequest
		time.sleep(sleep_time)
		# otherwise you might be blocked by Google finance
		sym_dat = {}
		url = 'https://www.google.com/finance/getprices?i=60&p='+str(num_day)+'d&f=d,c,v&q='+sym
		filename = loc+'/'+sym+'.dat'
		# url may not be valid
		try:
			df = pd.read_csv(url, skiprows=7, header=None)
		except:
			print(sym,'has url error and is thus dropped!')
			continue
		# filter out illiquid symbols
		if(len(df.index) < liq_thr*num_day*num_minute):
			print(sym,'is not liquid enough and thus dropped!')
			continue
		# loop over days
		ddx = 0
		while(ddx < len(df.index) and df.loc[ddx,0][0] == 'a'):
			# looking for the next date index
			nddx = ddx+1
			while(nddx < len(df.index) and df.loc[nddx,0][0] != 'a'):
				nddx += 1
			# range nailed down ddx:nddx
			this_df = df[ddx:nddx].copy(deep=True).reset_index(drop=True)
			this_date = datetime.datetime.fromtimestamp(int(this_df.loc[0,0][1:])).date()
			this_df.loc[0,0] = '0' # always start at 0 as the exact first traded timestamp
			this_df[0] = this_df[0].astype(int)
			this_df.rename(columns = {0:'Minute', 1:'Close', 2:'Volume'}, inplace=True)
			sym_dat[this_date] = this_df
			ddx = nddx
		with open(filename,'wb') as outfile:
			pickle.dump(sym_dat, outfile, pickle.HIGHEST_PROTOCOL)
			print(sym, 'has been saved!')

def resume_syms_min(syms=['NVDA'], loc='./data/min', num_day=1, liq_thr=0.0, sleep_time=1, num_minute = 391):
	# loop over symbols
	for sym in syms:
		# read pickle file back
		filename = loc+'/'+sym+'.dat'
		with open(filename,'rb') as infile:
			try:
				sym_dat = pickle.load(infile)
			except:
				print(sym, 'fails to resume!')
				continue
		# sleep before making the next urlrequest
		time.sleep(sleep_time)
		# otherwise you might be blocked by Google finance
		url = 'https://www.google.com/finance/getprices?i=60&p='+str(num_day)+'d&f=d,c,v&q='+sym
		# url may not be valid
		try:
			df = pd.read_csv(url, skiprows=7, header=None)
		except:
			print(sym,'has url error and is thus not updated!')
			continue
		# filter out illiquid symbols
		if(len(df.index) < liq_thr*num_day*num_minute):
			os.remove(filename)
			print(sym,'is not liquid enough and thus deleted!')
			continue
		# loop over days
		ddx = 0
		while(ddx < len(df.index) and df.loc[ddx,0][0] == 'a'):
			# looking for the next date index
			nddx = ddx+1
			while(nddx < len(df.index) and df.loc[nddx,0][0] != 'a'):
				nddx += 1
			# range nailed down ddx:nddx
			this_df = df[ddx:nddx].copy(deep=True).reset_index(drop=True)
			this_date = datetime.datetime.fromtimestamp(int(this_df.loc[0,0][1:])).date()
			# check if the file is in profile
			if this_date in sym_dat.keys():
				ddx = nddx
				continue
			this_df.loc[0,0] = '0' # always start at 0 as the exact first traded timestamp
			this_df[0] = this_df[0].astype(int)
			this_df.rename(columns = {0:'Minute', 1:'Close', 2:'Volume'}, inplace=True)
			sym_dat[this_date] = this_df
			ddx = nddx
		with open(filename,'wb') as outfile:
			pickle.dump(sym_dat, outfile, pickle.HIGHEST_PROTOCOL)
			print(sym, 'has been saved!')

def restart_trends_day(syms=['COMPUT'], loc='./data/day', startdate=datetime.date(2004,1,7), enddate=datetime.date(2013,1,7), sleep_time=1):
	# loop over symbols
	for sym in syms:
		# sleep before making the next urlrequest
		time.sleep(sleep_time)
		# otherwise you might be blocked by Google finance
		url = 'https://www.google.com/finance/historical?q=GOOGLEINDEX_US:'+sym
		url += '&output=csv&startdate='+str(startdate.year*10000+startdate.month*100+startdate.day)
		url += '&enddate='+str(enddate.year*10000+enddate.month*100+enddate.day)
		filename = loc+'/'+sym+'.dat'
		# url may not be valid
		try:
			df = pd.read_csv(url)
		except:
			print(sym,'has url error and is thus dropped!')
			continue
		# having problem calling df['Date'] here
		Date = []
		Close = df['Close']
		for idx in range(len(df.index)):
			Date.append(datetime.datetime.strptime(df.ix[idx,0],'%d-%b-%y'))
		sym_dat = pd.DataFrame({'Date':Date, 'Close':Close})
		with open(filename,'wb') as outfile:
			pickle.dump(sym_dat.sort_values(by='Date').reset_index(drop=True), outfile, pickle.HIGHEST_PROTOCOL)
			print(sym, 'has been saved!')

def resume_trends_day(syms=['COMPUT'], loc='./data/day', startdate=datetime.date(2012,1,7), enddate=datetime.datetime.today().date(), sleep_time=1):
	# loop over symbols
	for sym in syms:
		# read pickle file back
		filename = loc+'/'+sym+'.dat'
		with open(filename,'rb') as infile:
			try:
				sym_dat_old = pickle.load(infile)
			except:
				print(sym, 'fails to resume!')
				continue
		# sleep before making the next urlrequest
		time.sleep(sleep_time)
		# otherwise you might be blocked by Google finance
		url = 'https://www.google.com/finance/historical?q=GOOGLEINDEX_US:'+sym
		url += '&output=csv&startdate='+str(startdate.year*10000+startdate.month*100+startdate.day)
		url += '&enddate='+str(enddate.year*10000+enddate.month*100+enddate.day)
		# url may not be valid
		try:
			df = pd.read_csv(url)
		except:
			print(sym,'has url error and is thus not updated!')
			continue
		# having problem calling df['Date'] here
		Date = []
		Close = df['Close']
		for idx in range(len(df.index)):
			Date.append(datetime.datetime.strptime(df.ix[idx,0],'%d-%b-%y'))
		sym_dat_add = pd.DataFrame({'Date':Date, 'Close':Close}).sort_values(by='Date')
		# save the concatenated dataframe
		sym_dat = pd.concat([sym_dat_old, sym_dat_add]).drop_duplicates(subset='Date', keep='last')
		with open(filename,'wb') as outfile:
			pickle.dump(sym_dat.sort_values(by='Date').reset_index(drop=True), outfile, pickle.HIGHEST_PROTOCOL)
			print(sym, 'has been saved!')

def restart_syms_day(syms=['NVDA'], loc='./data/day', startdate=datetime.date(2004,1,7), enddate=datetime.datetime.today().date(), sleep_time=1):
	# loop over symbols
	for sym in syms:
		# sleep before making the next urlrequest
		time.sleep(sleep_time)
		# otherwise you might be blocked by Google finance
		url = 'https://www.google.com/finance/historical?q='+sym
		url += '&output=csv&startdate='+str(startdate.year*10000+startdate.month*100+startdate.day)
		url += '&enddate='+str(enddate.year*10000+enddate.month*100+enddate.day)
		filename = loc+'/'+sym+'.dat'
		# url may not be valid
		try:
			df = pd.read_csv(url)
		except:
			print(sym,'has url error and is thus dropped!')
			continue
		# having problem calling df['Date'] here
		Date = []
		Open = df['Open']
		Close = df['Close']
		High = df['High']
		Low = df['Low']
		Volume = df['Volume']
		for idx in range(len(df.index)):
			Date.append(datetime.datetime.strptime(df.ix[idx,0],'%d-%b-%y'))
		sym_dat = pd.DataFrame({'Date':Date,
								'Open':Open,
								'Close':Close,
								'High':High,
								'Low':Low,
								'Volume':Volume})
		with open(filename,'wb') as outfile:
			pickle.dump(sym_dat.sort_values(by='Date').reset_index(drop=True), outfile, pickle.HIGHEST_PROTOCOL)
			print(sym, 'has been saved!')

def resume_syms_day(syms=['NVDA'], loc='./data/day', startdate=datetime.date(2016,9,7), enddate=datetime.datetime.today().date(), sleep_time=1):
	# loop over symbols
	for sym in syms:
		# read pickle file back
		filename = loc+'/'+sym+'.dat'
		with open(filename,'rb') as infile:
			try:
				sym_dat_old = pickle.load(infile)
			except:
				print(sym, 'fails to resume!')
				continue
		# sleep before making the next urlrequest
		time.sleep(sleep_time)
		# otherwise you might be blocked by Google finance
		url = 'https://www.google.com/finance/historical?q=GOOGLEINDEX_US:'+sym
		url += '&output=csv&startdate='+str(startdate.year*10000+startdate.month*100+startdate.day)
		url += '&enddate='+str(enddate.year*10000+enddate.month*100+enddate.day)
		# url may not be valid
		try:
			df = pd.read_csv(url)
		except:
			print(sym,'has url error and is thus not updated!')
			continue
		# having problem calling df['Date'] here
		Date = []
		Open = df['Open']
		Close = df['Close']
		High = df['High']
		Low = df['Low']
		Volume = df['Volume']
		for idx in range(len(df.index)):
			Date.append(datetime.datetime.strptime(df.ix[idx,0],'%d-%b-%y'))
		sym_dat_add = pd.DataFrame({	'Date':Date,
										'Open':Open,
										'Close':Close,
										'High':High,
										'Low':Low,
										'Volume':Volume})
		# save the concatenated dataframe
		sym_dat = pd.concat([sym_dat_old, sym_dat_add]).drop_duplicates(subset='Date', keep='last')
		with open(filename,'wb') as outfile:
			pickle.dump(sym_dat.sort_values(by='Date').reset_index(drop=True), outfile, pickle.HIGHEST_PROTOCOL)
			print(sym, 'has been saved!')
