import pandas as pd
import numpy as np
import datetime
import pickle
import time
import os

# nasdaq
def nasdaq_lst(loc='./data/sym'):
	filename = loc+'/nasdaq.lst'
	try:
		sym_lst = pd.read_pickle(filename)
	except:
		print('Retrieving Nasdaq Symbol List...')
		df = pd.read_csv('ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt', sep='|')
		sym_lst = df[df['Test Issue'] == 'N']
		sym_lst.reset_index(drop=True, inplace=True)
		sym_lst.to_pickle(filename)
	return sym_lst

# nyse
def nyse_lst(loc='./data/sym'):
	filename = loc+'/nyse.lst'
	try:
		sym_lst = pd.read_pickle(filename)
	except:
		print('Retrieving NYSE Symbol List...')
		sym_lst = pd.read_csv('http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download')
		sym_lst.to_pickle(filename)
	return sym_lst

# all
def all_ex_lst(loc='./data/sym'):
	nasdaq = [ sym for sym in nasdaq_lst(loc)['Symbol'] ]
	nyse = [ sym for sym in nyse_lst(loc)['Symbol'] ]
	return nasdaq+nyse

# get liquid symbol list
def liq_lst(loc='./data/min'):
	f_lst = os.listdir(loc)
	sym_lst = []
	for f in f_lst:
		if(f[-4:] == '.dat'):
			sym_lst.append(f[:-4])
	return sym_lst

# get google domestic trends
def trend_lst():
	return [
			'ADVERT',	# Jan-01-2014
			'AIRTVL',	# Jan-01-2014
			'AUTOBY',	# Jan-01-2014
			'AUTOFI',	# Jan-02-2014
			'AUTO',		# Jan-01-2014
			'BIZIND',	# Jan-01-2014
			'BNKRPT',	# Jan-07-2014
			'COMPUT',	# Jan-01-2014
			'CONSTR',	# Jan-01-2014
			'CRCARD',	# Jan-01-2014
			'DURBLE',	# Jan-01-2014
			'EDUCAT',	# Jan-01-2014
			'INVEST',	# Jan-01-2014
			'FINPLN',	# Jan-01-2014
			'FURNTR',	# Jan-01-2014
			'INSUR',	# Jan-01-2014
			'JOBS',		# Jan-01-2014
			'LUXURY',	# Jan-02-2014
			'MOBILE',	# Jan-01-2014
			'MTGE',		# Jan-01-2014
			'RLEST',	# Jan-01-2014
			'RENTAL',	# Jan-01-2014
			'SHOP',		# Jan-01-2014
			'SMALLBIZ',	# Jan-01-2014
			'TRAVEL',	# Jan-01-2014
			'UNEMPL',	# Jan-01-2014
			]

# restart minute level stock data
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

# resume minute level stock data
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

# restart day level trend data
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

# resume day level trend data
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
			os.remove(filename)
			print(sym,'has url error and is thus deleted!')
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

# restart day level stock data
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

# resume day level stock data
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
		url = 'https://www.google.com/finance/historical?q='+sym
		url += '&output=csv&startdate='+str(startdate.year*10000+startdate.month*100+startdate.day)
		url += '&enddate='+str(enddate.year*10000+enddate.month*100+enddate.day)
		# url may not be valid
		try:
			df = pd.read_csv(url)
		except:
			os.remove(filename)
			print(sym,'has url error and is thus deleted!')
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

# aggreate from day to week
def day2week(df_in):
	# make a deep copy
	df = df_in.copy(deep = True)
	# assert open and close exist
	if 'Open' not in df.columns:
		df['Open'] = np.append(df['Close'][0], df['Close'][:-1])
		is_sym = False
	else:
		is_sym = True
	# get isoweek
	df['Week'] = df['Date'].map(lambda x: x.isocalendar()[0:2])
	# first day of the week
	df['1st_day'] = np.append(True, df['Week'][1:].values != df['Week'][:-1].values)
	# d_ln_p
	df['d_ln_p'] = np.log(df['Close']) - df['1st_day'].astype(float) * np.log(df['Open']) -\
	 np.append(0., (1.-df['1st_day'][1:].astype(float)) * np.log(df['Close'][:-1].values))
	# for computing volatility
	if is_sym:
		# insure high and low
		high = np.max(df[['Open','Close','High','Low']], axis=1)
		low = np.min(df[['Open','Close','High','Low']], axis=1)
		u = np.log(high / df['Open']).values
		d = np.log(low / df['Open']).values
		c = np.log(df['Close'] / df['Open']).values
		df['sigma_sq'] = 0.511*(u-d)**2 - 0.019*(c*(u+d)-2.*u*d) - 0.383*c**2 
	# aggregate
	grouped = df.groupby('Week')
	ret = grouped['d_ln_p'].mean()
	ans = pd.DataFrame( {'Week':ret.index, 'Ret':ret.values} )
	if is_sym:
		ans['Sigma'] = np.sqrt(grouped['sigma_sq'].mean().values)
		ans['LnVol'] = np.log(grouped['Volume'].mean().values)
	# return
	return ans
