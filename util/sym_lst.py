import pandas as pd
from os import listdir

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
def liq_ex_lst(loc='./data/min'):
	f_lst = listdir(loc)
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
