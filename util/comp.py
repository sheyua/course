import numpy as np
import pandas as pd

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
	df['1st_day'] = np.append(True, df['Week'][1:] != df['Week'][:-1])
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
