# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:59:07 2021

@author: saideep
"""
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pandas_datareader._utils import RemoteDataError
import pandas as pd
from datetime import datetime, timedelta
import pandas_datareader.data as web
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from datetime import datetime
import plotly.express as px
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pandas_datareader._utils import RemoteDataError
import pandas as pd
from datetime import datetime, timedelta
import pandas_datareader.data as web
from datetime import datetime as dt
from datetime import datetime, timedelta
from utils import Header, make_dash_table
from datetime import datetime , timedelta
import pandas as pd
import pathlib
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from utils import Header, make_dash_table
import pandas as pd
import pathlib
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import yfinance as yf
import plotly.express as px
from fbprophet import Prophet
from datetime import datetime as dt
from datetime import datetime, timedelta
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import datetime
import plotly.express as px
from dash import Dash, dcc, html, callback, Input, Output
from dash import dash_table
import nltk
import yfinance as yf
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
# get relative data folder



PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("./data").resolve()


df_fund_facts = pd.read_csv(DATA_PATH.joinpath("df_fund_facts.csv"))
df_price_perf = pd.read_csv(DATA_PATH.joinpath("df_price_perf.csv"))
df = pd.read_csv('/Users/karteekedumudi/Desktop/Capstone_Key_Label.csv')
# start the App
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, 
    external_stylesheets=external_stylesheets,
    # to verify your own website with Google Search Console
    meta_tags=[{'name': 'google-site-verification', 'content': 'Wu9uTwrweStHxNoL-YC1uBmrsXYFNjRCqmSQ8nNnNMs'}])
app.title = 'Portfolio Risk Manager'


server = app.server



prediction_col1 =  dbc.Col([ 
                html.Br(),
                dbc.Row([html.H3(children='Product Summary')]),
                dbc.Row([
                    html.P(
                                        "\
                                    This Application is designed to asses the portfolio Risk Management for a stock \
                                    trader or an enterprise, using various machine learning techniques. Data used to train \
                                    includes news articles and previous stock data to predict the stock movement, \
                                    closing price and trend. This application has almost 100 stocks listed from\
                                     NASDAQ which can be analysed. It is backed up with a huge kaggle database, with \
                                     almost 200,000 records and millions of stock considered a core for this tool.",
                                        style={"color": "black"},
                                        className="row",
                                    ),
                ]), 
                dbc.Row([dbc.Col(
                    html.Div(
                                [
                                    html.H6(
                                        ["Kaggle data sample"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table(df_fund_facts)),
                                ])),
                    
                    dbc.Col(
                    html.Div(
                                [
                                    html.H6(
                                        "Base Model Performance",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-1",
                                        figure={
                                            "data": [
                                                go.Bar(
                                                    x=[
                                                        "Decision Tree Regressor",
                                                        "Random Forest Regressor",
                                       
                                                   ],
                                                    y=[
                                                        "8.873",
                                                        "6.0515",
                                                    ],
                                                    marker={
                                                        "color": "#97151c",
                                                        "line": {
                                                            "color": "rgb(255, 255, 255)",
                                                            "width": 2,
                                                        },
                                                    },
                                                    name="Regression Model Performance (MSE)",
                                                ),
                                                go.Bar(
                                                    x=[
                                                        "K Nearest Neighbours",
                                                        "Decision Tree",
                                                        "Random Forest",
                                                    ],
                                                    y=[
                                                        "76.15",
                                                        "65.03",
                                                        "59.71",
                                                    ],
                                                    marker={
                                                        "color": "#dddddd",
                                                        "line": {
                                                            "color": "rgb(255, 255, 255)",
                                                            "width": 2,
                                                        },
                                                    },
                                                    name="Classification Model Performance (Accuracy Score)",
                                                ),
                                            ],
                                            "layout": go.Layout(
                                                autosize=False,
                                                bargap=0.35,
                                                font={"family": "Raleway", "size": 10},
                                                height=200,
                                                hovermode="closest",
                                                legend={
                                                    "x": -0.0228945952895,
                                                    "y": -0.189563896463,
                                                    "orientation": "h",
                                                    "yanchor": "top",
                                                },
                                                margin={
                                                    "r": 0,
                                                    "t": 20,
                                                    "b": 10,
                                                    "l": 10,
                                                },
                                                showlegend=True,
                                                title="",
                                                width=330,
                                                xaxis={
                                                    "autorange": True,
                                                    "range": [-0.5, 4.5],
                                                    "showline": True,
                                                    "title": "",
                                                    "type": "category",
                                                },
                                                yaxis={
                                                    "autorange": True,
                                                    "range": [0, 22.9789473684],
                                                    "showgrid": True,
                                                    "showline": True,
                                                    "title": "",
                                                    "type": "linear",
                                                    "zeroline": False,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ])),
                
                ]),
                dbc.Row([dbc.Col(
                    html.Div(
                                [
                                    html.H6(
                                        "KNN Classification Model Confusion Matrix", className="subtitle padded"
                                    ),
                                    html.Img(
                                        src=app.get_asset_url("Classification_confusion_matrix_intro_page.png"),
                                        className="risk-reward",
                                    ),
                                ])),
                    dbc.Col(
                    html.Div(
                                [
                                    html.H6(
                                        "Co-relation Matrix", className="subtitle padded"
                                    ),
                                    html.Img(
                                                src=app.get_asset_url("Intro_page_Heatmap.png"),
                                        className="risk-reward",
                                    ),
                                ])),
                ]), 
                dbc.Row([dbc.Col(
                    html.Div(
                                [
                                    html.H6(
                                        "KDE plot for Closing Prices in the Data",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-2",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=[0,
                                                     50,
                                                     100,
                                                     150,
                                                     200,
                                                     250,
                                                     300,
                                                     350,
                                                     400,
                                                     450,
                                                     500,
                                                     550,
                                                     600,
                                                     650,
                                                     700,
                                                     750,
                                                     800,
                                                     850,
                                                     900,
                                                     950,
                                                     1000,
                                                     1050,
                                                     1100,
                                                     1150,
                                                     1200,
                                                     1250,
                                                     1300,
                                                     1350,
                                                     1400,
                                                     1450,
                                                     1500,
                                                     1550,
                                                     1600,
                                                     1650,
                                                     1700,
                                                     1750,
                                                     1800,
                                                     1850,
                                                     1900,
                                                     1950,
                                                     2000,
                                                     2050,
                                                     2100,
                                                     2150,
                                                     2200,
                                                     2250,
                                                     2300,
                                                     2350,
                                                     2400,
                                                     2450],
y = [0.0, 0.002, 0.004, 0.006, 0.0055, 0.004, 0.002, 0.001,0.0015,0.0001,0.0001, 0.0001, 0.0002, 0.0003, 0.0002, 0.0001, 0.0006,0.00058,0.00055, 0.00045, 0.00034999999999999996, 0.00024999999999999996, 0.00014999999999999996,0.00004999999999999996,0.00005, 0.00007, 0.00009,0.0001, 0.00008, 0.00006000000000000001,0.00034999999999999996, 0.00024999999999999996, 0.00014999999999999996,0.00004999999999999996,0.00005, 0.00007, 0.00009,0.00001, 0.00008, 0.00006000000000000001
                                                    ],
                                                    line={"color": "#97151c"},
                                                    mode="lines",
                                                    name="Calibre Index Fund Inv",
                                                )
                                            ],
                                            
                                            "layout": go.Layout(
                                                autosize=True,
                                                title="",
                                                font={"family": "Raleway", "size": 10},
                                                height=200,
                                                width=340,
                                                hovermode="closest",
                                                legend={
                                                    "x": -0.0277108433735,
                                                    "y": -0.142606516291,
                                                    "orientation": "h",
                                                },
                                                margin={
                                                    "r": 20,
                                                    "t": 20,
                                                    "b": 20,
                                                    "l": 50,
                                                },
                                                showlegend=True,
                                                xaxis={
                                                    "autorange": True,
                                                    "linecolor": "rgb(0, 0, 0)",
                                                    "linewidth": 1,
                                                    "range": [0, 2500],
                                                    "showgrid": False,
                                                    "showline": True,
                                                    "title": "Close Price",
                                                    "type": "linear",
                                                },
                                                yaxis={
                                                    "autorange": False,
                                                    "gridcolor": "rgba(127, 127, 127, 0.2)",
                                                    "mirror": False,
                                                    "nticks": 4,
                                                    "range": [0, 0.008],
                                                    "showgrid": True,
                                                    "showline": True,
                                                    "ticklen": 10,
                                                    "ticks": "outside",
                                                    "title": "Density Plot",
                                                    "type": "linear",
                                                    "zeroline": False,
                                                    "zerolinewidth": 4,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ])),
                    dbc.Col(
                    html.Div(
                                [
                                    html.H6(
                                        "KDE Plot for Sentiment Scores", className="subtitle padded"
                                    ),
                                    html.Img(
                                        src=app.get_asset_url("KDE_sentiment.png"),
                                        className="risk-reward",
                                    ),
                                ])),
                ])
                
            ], style = {'padding': '0px 0px 0px 150px'})

# # prediction 2

prediction_col2 =  dbc.Col([ 
                html.Br(),
            
                dbc.Row([html.H3(children='Preditive analytics')]),
                dbc.Row([
                    dbc.Col(html.Label(children='Stock Symbol:'), width={"order": "first"}, 
                                style = {'padding': '15px 0px 0px 0px',"font-weight": "bold"}),
                    dcc.Dropdown(
                        id='stock_symbol',
                        options= [{"label": df['ticker'][i], "value": df['ticker'][i]} for i in range(len(df))],
                        multi=False,
                        clearable=True,
                        disabled=False,
                        style = {"width": "50%", 'padding': '5px 0px 5px 10px', 'display': 'inline-block'}
                    )
                ]), 
                dbc.Row([
                    dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")
                    
                ]),
                dbc.Row([
                                    html.H6(id='first')
                                ]),
                dbc.Row([
                                  html.H6(id='second')  
                                ]),
                dbc.Row([
                                  html.H6(id='third')  
                                ]),
                dbc.Row([
                                    html.H6("Performance", className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-4", style={'marginBottom': '3em'}
                                        #options = Time_Series_figure(stock_symbol)
                                    )
                                ]),
                    html.Br(),
                    html.Br(),
                dbc.Row([
                    
                                   
                                        dash_table.DataTable(id='news_table',style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'maxWidth': 0})
                                   
                                ]),
                dbc.Row([
                                    html.H6("Performance", className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-5", style={'marginBottom': '3em'}
                                        #options = Time_Series_figure(stock_symbol)
                                    )
                                ])
                
                    
                
            ], style = {'padding': '0px 0px 0px 150px'})




   

##############################################################
# prepare the layout
app.layout = html.Div([
    html.H1(children='PORTFOLIO RISK MANAGER',style={'color':'#6e4318'}),#654321
    #html.Div(children='''vaccinations are Covid,FLU Influenza,Hepatities,Varzos '''),
    html.Br(),
    
    dcc.Tabs(style = {'width': '100%',"font-weight": "bold"}, children=[
        # this is the first tab
        
        # this is the 3rd tab
        dcc.Tab(label='OVERVIEW', children = [prediction_col1
            #dbc.Row([prediction_col1, prediction_col2])
            
        ]), # the end of the 3rd tab
        dcc.Tab(label='PREDICTIONS', children = [prediction_col2
            
        ]),
        # this is the 4th tab
#         dcc.Tab(label='vaccine Manufactorer Prediction', children = [
#             dbc.Row([sales_col_map]),
#             dbc.Row([sales_col_table])
#         ]) # the end of the 4th tab




    ], colors={
        "border": "white",
        "primary": "gold",
        "background": "cornsilk"}) # end of all tabs

# ], style = {
# 'background-image': 'url("/assets/image1.jpg")',
# 'background-repeat': 'no-repeat',
# 'background-size': 'cover'
# }) # the end of app.layout
])

####################################################################################################################
# creating the call back for manufactotar drop down
@app.callback(
    Output('graph-4', 'figure'),
    Output('news_table', 'data'),
    Output('graph-5', 'figure'),
    Output('first', 'children'),
    Output('second', 'children'),
    Output('third', 'children'),
    Input('submit-val', 'n_clicks'),
    State('stock_symbol', 'value'))
def Time_Series_figure(n_clicks,stock_symbol):
    if stock_symbol!=None:
        end = datetime.date.today()
        end = end.strftime('%Y-%m-%d')
        start = datetime.date.today() - datetime.timedelta(days = 365)
        start = start.strftime('%Y-%m-%d')
        data = yf.download(stock_symbol, 
          start=start, 
          end=end, 
          progress=False)
        data['Date'] = data.index
        # Select only the important features i.e. the date and price
        data = data[["Date","Close"]] # select Date and Price
        # Rename the features: These names are NEEDED for the model fitting
        data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
        data.head(5)
        m = Prophet(yearly_seasonality=True,daily_seasonality=True) # the Prophet class (model)
        m.fit(data) # fit the model using all data
        future = m.make_future_dataframe(periods=30) #we need to specify the number of days in future
        prediction = m.predict(future)
        data = yf.download(stock_symbol, 
          start=start, 
          end=end, 
          progress=False)
        data['Date'] = data.index
        pred = prediction
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred['ds'], y=pred['yhat'], name='Predicted'))
        fig.add_trace(go.Scatter(x=pred['ds'], y=data['Close'], name = 'Actual'))
        now = datetime.date.today()
        now = now.strftime('%m-%d-%Y')
        yesterday = datetime.date.today() - datetime.timedelta(days = 2)
        yesterday = yesterday.strftime('%m-%d-%Y')
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 50
        # save the company name in a variable
        company_name = stock_symbol
        #As long as the company name is valid, not empty...
        if company_name != '':
            print(f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')
            #Extract News with Google News
            googlenews = GoogleNews(start=yesterday,end=now)
            googlenews.search(company_name)
            result = googlenews.result()
            #store the results
            df = pd.DataFrame(result)
        try:
            list_ =[] #creating an empty list 
            for i in df.index:
                dict_ = {} #creating an empty dictionary to append an article in every single iteration
                article = Article(df['link'][i],config=config) #providing the link
                try:
                    article.download() #downloading the article 
                    article.parse() #parsing the article
                    article.nlp() #performing natural language processing (nlp)
                except:
                    pass 
                #storing results in our empty dictionary
                dict_['Date']=df['date'][i] 
                dict_['Media']=df['media'][i]
                dict_['Title']=article.title
                dict_['Article']=article.text
                dict_['Summary']=article.summary
                dict_['Key_words']=article.keywords
                list_.append(dict_)
            check_empty = not any(list_)
            # print(check_empty)
            if check_empty == False:
                news_df=pd.DataFrame(list_)#creating dataframe
                news_df

        except Exception as e:
            #exception handling
            print("exception occurred:" + str(e))
            print('Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.' )
        news_df

        word_list =[]
        for text in news_df['Article']:
            sentences = sent_tokenize(str(text))
            tokens = word_tokenize(text)
            words = [word for word in tokens if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            str1 = ''
            for i in words:
                str1 = str1+i+" "
            word_list.append(str1)
        news_df['words'] = word_list
        news_df
        vader = SentimentIntensityAnalyzer()
        scores = news_df['words'].apply(vader.polarity_scores).tolist()
        news_df['Score'] = scores
        neg = []
        pos = []
        neu = []
        comp = []
        for i in news_df['Score']:
            neg.append(i['neg'])
            pos.append(i['pos'])
            neu.append(i['neu'])
            comp.append(i['compound'])
        news_df['Positive Score'] = pos
        news_df['Negative Score'] = neg
        news_df['Neutral Score'] = neu
        news_df['Compound'] = comp
        pn = []
        for i in news_df['Compound']:
            if i > 0:
                pn.append(1)
            elif i == 0:
                pn.append(0)
            else:
                pn.append(-1)
        news_df['PNN Score'] = pn 
        pie_list = [news_df['Positive Score'].mean(), news_df['Negative Score'].mean(), news_df['Neutral Score'].mean()]
        Name_list = ['Positive','Negative','Neutral']
        fig2 = px.pie(news_df, values=pie_list, names=Name_list, title='Stock Name '+'Financial News Sentiment for '+stock_symbol)
        fetched = False
        d = 1
        while fetched == False:
            today = dt.today()
            yesterday = today - timedelta(days = d)
            today = today.strftime('%Y-%m-%d')
            yesterday = yesterday.strftime('%Y-%m-%d')
            data = yf.download(stock_symbol, 
                                start=yesterday, 
                                end=today, 
                                progress=False,
            )
            if len(data) == 1:
                fetched = True
            else:
                d+=1
        predict_df = news_df.drop(['Date','Media','Article','Summary','Key_words','words','Score','Title'], axis = 1)
        predict_df['Open'] = data['Close'][0]
        predict_df['Volume'] = data['Volume'][0]
        key_df = pd.read_csv('Capstone_Key_Label.csv').drop('Unnamed: 0', axis = 1)
        predict_df.insert(loc=0, column='ticker', value=key_df[key_df['ticker'] == stock_symbol]['keys'].iloc[0])
        pickled_model_Regression = pickle.load(open('Capstone_RFR_model.pkl', 'rb'))
        pickled_model_classification = pickle.load(open('Capstone_Classification_model.pkl', 'rb'))

        def predicted_closing_price():
            if news_df['PNN Score'].sum()/len(news_df['PNN Score']) > 0.5:
                pred_close = pickled_model_Regression.predict(predict_df).max()
            elif news_df['PNN Score'].sum()/len(news_df['PNN Score']) > 0.3 and news_df['PNN Score'].sum()/len(news_df['PNN Score']) < 0.5:
                pred_close = pickled_model_Regression.predict(predict_df).mean()
            else:
                pred_close = pickled_model_Regression.predict(predict_df).min()
            return pred_close

        predicted = pickled_model_classification.predict(predict_df)

        pred_close = predicted_closing_price()

        if predicted.sum()/len(predicted) > 0.8:
            first = "Predicted to Strongly increase today..."
        elif predicted.sum()/len(predicted) > 0.5:
            first = "This stock is doing okay..."
        else:
            first = "Predicted to decrease..."

        second = f"Predicted Close price is: {pred_close}"

        if pred_close > predict_df['Open'].iloc[0]:
            per_change = (predicted_closing_price() - predict_df['Open'].iloc[0])*100/predict_df['Open'].iloc[0]
            third = f"Stock is going to improve today by {per_change}%"
        else:
            per_change = (predict_df['Open'].iloc[0] - pred_close)*100/predict_df['Open'].iloc[0]
            third = f"Stock price might fall today by {per_change}%"
        return fig,df[['date','title','link']].to_dict('records'),fig2,first,second,third
    fig = go.Figure()

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          yaxis = dict(showgrid=False, zeroline=False, tickfont = dict(color = 'rgba(0,0,0,0)')),
                          xaxis = dict(showgrid=False, zeroline=False, tickfont = dict(color = 'rgba(0,0,0,0)')))
    return fig,None,fig,None,None,None

# run the app 
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
    