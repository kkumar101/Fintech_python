#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 01:56:06 2023

@author: karankumar
"""

import pandas as pd, yfinance as yf, numpy as np, pandas_datareader.data as web
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
import requests
import zipfile
import io
import os

start = '2010-01-01'
end = '2023-04-26'

#Q1
sp = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp.columns= sp.columns.str.replace(' ','_').str.lower()
list = sp[sp["gics_sector"]== 'Information Technology']['symbol'].to_list()
returns = yf.download(list, start=start, end=end)['Adj Close'].pct_change()[1:]

#Winsorization
returns = returns.clip(lower=returns.quantile(q=.05),upper=returns.quantile(q=.95),axis=1)
returns = returns.dropna(thresh=int(returns.shape[0] * .95), axis=1)

#Removing trading days not having data of atleast 95% of the stocks
returns = returns.dropna(thresh=int(returns.shape[1] * .95))

# Filling up missing observations by average return of that particular trading day
avg_ret = returns.mean(1)
returns = returns.apply(lambda x: x.fillna(avg_ret))

#Q2 PCA Analysis
pca = PCA().fit(returns)
components = pd.DataFrame(pca.components_, columns=returns.columns) 
explained_variances = pca.explained_variance_ratio_
cumulative_variances = np.cumsum(explained_variances)

# Create a bar chart showing the explained variance ratios of the first 10 principal components 
plt.bar(x=range(1,11), height=explained_variances[:10], color='green')
plt.title("Variance Ratio explained by PCA")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio") 
plt.xticks(range(1, 11))
plt.xlim((0, 10)) 
plt.ylim((0, 1)) 
plt.show()

#Q3
pc1 = pd.Series(index=returns.columns, data=pca.components_[0])
pc1.to_frame()
spy = yf.download('SPY', start=start,end=end)['Adj Close'].pct_change()[1:]
merged = pd.merge(pc1.to_frame(), sp[['gics_sector','symbol']],
                  left_index=True, right_on='symbol',how='inner').set_index('symbol')

# Change column name
merged.rename(columns={0:'pc1'}, inplace=True)
pc_small=merged.nsmallest(10,'pc1')
pc_large=merged.nlargest(10,'pc1')

#Compare the performances
ret_h = returns[pc_large.index].mean(1)
cret_h = ret_h.cumsum().apply(np.exp)
ret_l = returns[pc_small.index].mean(1) #representing the smallest weight stocks 
cret_l = ret_l.cumsum().apply(np.exp)

ret_sec = returns[merged.index].mean(1) #representing sectors 
cret_sec = ret_sec.cumsum().apply(np.exp)
cret_mkt = spy.cumsum().apply(np.exp) #representing the market

# Plotting
cret_h.plot(title='PCA Portfolios vs. Sector vs. Market') 
cret_l.plot()
cret_sec.plot()
cret_mkt.plot()
plt.legend(['PCA Largest', 'PCA Smallest','Sector','market'])


#Q4 
# Download daily stock returns of the IT sector
sp = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp.columns = sp.columns.str.replace(' ','_').str.lower()
it = sp[sp["gics_sector"] == 'Information Technology']
it_data = yf.download(it['symbol'].to_list(), start=start, end=end)['Adj Close']
it_ret = it_data.resample('M').last().pct_change()[1:]
it_ret = it_ret.apply(lambda x: x.fillna(x.mean()), axis=1)
file_ret = pd.DataFrame(it_ret,columns = pc_small.index)

#Get S&P 500 Data
sp500_index = yf.download('^GSPC',start='2010-01-01',end='2023-04-26')['Adj Close']
sp500_ret = sp500_index.resample('M').last().pct_change()[1:]
all_stock_ret_data = pd.merge(file_ret,sp500_ret,how='inner', left_index=True, right_index=True)
all_stock_ret_data = all_stock_ret_data.rename(columns= {'Adj Close':'sp500'})

# Download 3-month Tbill
rf = web.DataReader('TB3MS','fred',start = '2010-01-01', end= '2023-04-26')
rf = (1 + (rf / 100)) ** (1 / 12) - 1
rf = rf.resample('M').last()
all_stock_ret_data.index = all_stock_ret_data.index.date
rf.index = rf.index.date
all_data = pd.merge(all_stock_ret_data,rf,how = 'inner',left_index = True,right_index = True)

#Q5
excess_ret = all_data.drop(columns = ['TB3MS'])
excess_ret.index = pd.to_datetime(excess_ret.index)
annual_sharpe = excess_ret.groupby(excess_ret.index.year).agg('mean')/excess_ret.groupby(excess_ret.index.year).agg('std')                                                                                                                             
sharpe_ratio = annual_sharpe.agg('mean').dropna()
top5 = sharpe_ratio.drop('sp500').nlargest(5).index.tolist()   
top5_it_stock = sharpe_ratio.drop('sp500').nlargest(5).reset_index().rename(columns={'index':'security',0:'Sharpe'})
top5_it_stock

#Q6
def estimate(x,y):
    n = np.size(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    ss_xy = np.sum(x*y)-n*mean_x*mean_y
    ss_xx = np.sum(x*y)-n*mean_x*mean_x
    
    beta = ss_xy/ss_xx
    return(beta)

beta_nvda = estimate(excess_ret['sp500'],excess_ret['NVDA'])
beta_mpwr = estimate(excess_ret['sp500'],excess_ret['MPWR'])
beta_lrcx = estimate(excess_ret['sp500'],excess_ret['LRCX'])
beta_swks = estimate(excess_ret['sp500'],excess_ret['SWKS'])
beta_nxpi = estimate(excess_ret['sp500'],excess_ret['NXPI'])

#Q7
directory = '/Users/karankumar/Desktop/Fintech/HW4/'

def download_files(y, m):
    SEC_URL = 'https://www.sec.gov/'
    FSN_PATH = 'files/dera/data/financial-statement-and-notes-data-sets/' 
    filing = f'{y}_{"%02d" % m}_notes.zip'
    url = SEC_URL + FSN_PATH + filing
    sub_dir = f'{y}M{"%02d" % m}'
    final_dir = directory + sub_dir
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    output = final_dir + '/' + filing
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for file in z.namelist():
            if file.endswith('num.tsv') or file.endswith('sub.tsv'): 
                with open(final_dir + '/' + file, 'wb') as f:
                    f.write(z.read(file))
    print(f'Successfully downloaded files for {y}M{"%02d" % m}')

for i in range(1, 3): 
    download_files(2023, i)
    
num_files = glob.glob(f'{directory}/**/*num.tsv', recursive=True) 
sub_files = glob.glob(f'{directory}/**/*sub.tsv', recursive=True)
num_df = pd.concat((pd.read_csv(f, delimiter='\t', encoding='utf-8') for f in num_files)) 
chech_num = num_df.head(50)
sub_df = pd.concat((pd.read_csv(f, delimiter='\t', encoding='utf-8') for f in sub_files)) 
sub_final_10k = sub_df[sub_df['form'] == '10-K'][['name','adsh','cik','filed']].\
    drop_duplicates(['cik','filed'])

sp = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp.columns= sp.columns.str.replace(' ','_').str.lower()
sp_match=pd.merge(sp,sub_final_10k,how='inner',left_on='cik',right_on='cik')
sp_match_num = pd.merge(sp_match,num_df,how='inner',on='adsh')[['symbol','tag','security', 
                                                                'gics_sector','cik','adsh','filed','ddate','value']]
sp_match_num = sp_match_num.drop_duplicates(['cik','adsh','tag','filed','ddate'])

latest_sp_match_num1 = sp_match_num.sort_values(['cik', 'adsh', 'tag', 'filed', 'ddate'], 
 ascending=[True, True, True, True, False]).\
 groupby(['cik', 'adsh', 'tag', 'filed']).first().reset_index()


include1 = ['NXPI','SWKS','LRCX','MPWR','NVDA']
sp_selected_cik = sp[sp["gics_sector"].isin(include1)]['cik'].to_list() 
sub_df_it = pd.concat((pd.read_csv(f, delimiter='\t',
    encoding='utf-8') for f in sub_files))[lambda x: 
                                           x['cik'].isin(sp_selected_cik)].\
                                           query('form=="10-K"')[['name','adsh','cik','filed']].\
                                           drop_duplicates(['cik','filed'])
# Add sector information from sp file
sp_match_it = pd.merge(sp, sub_df_it,how='inner', on='cik')

#Download Fianncial Data
api_key = '501fd60d6e53f8a2c83fb886755dfc15'
for company in include1:
    bs_url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{company}?limit=1&apikey={api_key}'
    is_url = f'https://financialmodelingprep.com/api/v3/income-statement/{company}?limit=1&apikey={api_key}'
    
    bs_data = requests.get(bs_url).json()
    is_data = requests.get(is_url).json()

    current_assets = bs_data[0]['totalCurrentAssets']
    current_liabilities = bs_data[0]['totalCurrentLiabilities']
    total_assets = bs_data[0]['totalAssets']
    total_liabilities = bs_data[0]['totalLiabilities']
    net_income = is_data[0]['netIncome']
    revenue = is_data[0]['revenue']

    current_ratio = current_assets / current_liabilities
    debt_ratio = total_liabilities / total_assets
    profit_margin = net_income / revenue
    return_on_assets = net_income / total_assets
    print('------')
    print(f'Stock: {company}')
    print(f'Current Ratio: {current_ratio:.2f}')
    print(f'Debt Ratio: {debt_ratio:.2f}')
    print(f'Profit Margin: {profit_margin:.2f}')
    print(f'Return on Assets: {return_on_assets:.2f}')
    print(' ')

#Q8
stocks = ['NVDA','MPWR','LRCX','SWKS','NXPI']
ret_stock = returns[pc_small.index]
ret_stock_new = pd.DataFrame(ret_stock, columns=stocks)
ret_select = ret_stock_new.mean(1) # representing the selected 5 stocks
cret_selected = ret_select.cumsum().apply(np.exp)
# Plotting
cret_selected.plot(title='PCA Selected 5 Stocks vs. Sector vs. Market')
cret_sec.plot()
cret_mkt.plot()
plt.legend(['PCA Selected','Sector','market'])

