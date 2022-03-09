#!/usr/bin/env python
# coding: utf-8

# In[112]:


import yfinance as yf
import streamlit as st
import pandas as pd
import datetime 

st.write("""
# My project basic UI
Some information about Google stock
""")

tickerSymbol = 'GOOGL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.news
websites = []
title = []
for i in range(len(tickerData.news)):
    websites.append(tickerData.news[i]['link'])
    title.append(tickerData.news[i]['title'])
    
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=30)
t_day = str(today.year)+'-'+str(today.month)+'-'+str(today.day)
w_day = str(yesterday.year)+'-'+str(yesterday.month)+'-'+str(yesterday.day)

#get the historical prices for this ticker
tickerdf = tickerData.history(period='1d', start=w_day, end=t_day)
# Open	High	Low	Close	Volume	Dividends	Stock Splits

st.write("""
## Closing Price
""")
st.line_chart(tickerdf.Close)
st.write("""
## Volume Price
""")
st.line_chart(tickerdf.Volume)
    
df = pd.DataFrame()
df['Title'] = title
df['Article Link'] = websites


st.header("""
Lates News Articles
""")
for i in range(len(websites)):
    st.text(title[i])
    text='check out this [link]({link})'.format(link=websites[i])
    st.write(text)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




