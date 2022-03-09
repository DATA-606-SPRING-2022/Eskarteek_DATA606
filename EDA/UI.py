#!/usr/bin/env python
# coding: utf-8

# In[53]:


import yfinance as yf
import streamlit as st
import pandas as pd
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


    
df = pd.DataFrame()
df['Title'] = title
df['Article Link'] = websites



st.write(df)

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




