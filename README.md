# DATA606 Capstone Project

# STOCK TREND PREDICTION USING NEWS

# SENTIMENT ANALYSIS

### Sai Karteek Edumudi,
### Prof. Dr Chaojie Wang 
### Department of Data Science, University of Maryland Baltimore County, MD 
### Email: df56654@umbc.edu

### ABSTRACT

The efficient Market Hypothesis is the popular theory about stock prediction. With its failure, much research has been carried out in the area of prediction of stocks. This project is about taking nonquantifiable data such as financial news articles about a company and predicting its future stock trend with news sentiment classification, prediction, and trend analysis. Assuming the news articles have an impact on the stock market, this is an attempt to study the relationship between news and stock trend. To show this, we created three different models one which depicts the polarity of news articles being positive or negative, one classifier to predict movement, and a regression model to predict the expected closing price. Observations show that RF and DT, KNN perform well in all types of testing. Facebook Prophet model which predicts the trend of the stock. Experiments are conducted to evaluate various aspects of the proposed model and encouraging results are obtained in all the experiments. The accuracy of the overall model is more than 80% as it uses multiple models for providing a final estimate. 
 


### KEYWORDS

Text Mining, Sentiment analysis, Decision Tree, Random Forest, Facebook Prophet, Stock trends

## 1. INTRODUCTION

In the finance field, the stock market and its trends are extremely volatile in nature. It attracts researchers to 
find techniques in capturing the volatility and predicting its next moves. Investors and market analysts study the market 
behavior and plan their buy or sell strategies accordingly. As the stock market produces a large amount of data every day, 
it is very difficult for an individual to consider all the current and past information for predicting future trend of a stock.
 Mainly there are two methods for forecasting market trends. One is Technical analysis and other is Fundamental analysis. 
Technical analysis considers past price and volume to predict the future trend whereas fundamental analysis 
On the other hand, Fundamental analysis of a business involves analyzing its financial data to get some insights. 
The efficacy of both technical and fundamental analysis is disputed by the efficient-market hypothesis which 
states that stock market prices are essentially unpredictable.


This research follows the Fundamental analysis technique to discover the future trends of stock by
considering news articles about a company as prime information and tries to classify news as
good (positive) and bad (negative). If the news sentiment is positive, there are more chances that
the stock price will go up and if the news sentiment is negative, then the stock price may go down.
We assumed that news articles and stock prices are related to each other and, that the news may have 
the capacity to fluctuate stock trends. So, we thoroughly studied this relationship and concluded that stock 
trends can be predicted using news articles and previous price history. As news articles capture sentiment 
about the current market, we automate this sentiment detection and based on the words in the news articles,
 we can get an overall news polarity. If the news is positive, then we can compare it with the closing price 
to verify that the news impact is good in the stock market, so more chances of the stock price going high. 
And if the news is negative, then it may impact the stock price to go down in trend. We have taken the past 
ten years' data for more than 100 stocks listed on NASDAQ, their stock prices, and news articles.

## 2. LITERATURE SURVEY

Stock price trend prediction is an active research area, as more accurate predictions are directly
related to more returns in stocks. Therefore, in recent years, significant efforts have been put
into developing models that can predict for future trend of a specific stock or overall market.
Most of the existing techniques make use of the technical indicators. Some of the researchers
showed that there is a strong relationship between news article about a company and its stock
prices fluctuations. Following is discussion on previous research on sentiment analysis of text
data and different classification techniques.
Nagar and Hahsler in their research [1] presented an automated text mining based approach to
aggregate news stories from various sources and create a News Corpus. The Corpus is filtered
down to relevant sentences and analyzed using Natural Language Processing (NLP) techniques.
A sentiment metric, called NewsSentiment, utilizing the count of positive and negative polarity
words is proposed as a measure of the sentiment of the overall news corpus. They have used
various open source packages and tools to develop the news collection and aggregation engine
as well as the sentiment evaluation engine. They also state that the time variation of
NewsSentiment shows a very strong correlation with the actual stock price movement.


Yu et al [2] present a text mining based framework to determine the sentiment of news articles
and illustrate its impact on energy demand. News sentiment is quantified and then presented as a
time series and compared with fluctuations in energy demand and prices.
J. Bean [3] uses keyword tagging on Twitter feeds about airlines satisfaction to score them for
polarity and sentiment. This can provide a quick idea of the sentiment prevailing about airlines
and their customer satisfaction ratings. We have used the sentiment detection algorithm based
on this research.
This research paper [4] studies how the results of financial forecasting can be improved when
news articles with different levels of relevance to the target stock are used simultaneously. They
used multiple kernels learning technique for partitioning the information which is extracted
from different five categories of news articles based on sectors, sub-sectors, industries etc.
News articles are divided into the five categories of relevance to a targeted stock, its sub
industry, industry, group industry and sector while separate kernels are employed to analyze
each one. The experimental results show that the simultaneous usage of five news categories
improves the prediction performance in comparison with methods based on a lower number of
news categories. The findings have shown that the highest prediction accuracy and return per
trade were achieved for MKL when all five categories of news were utilized with two separate
kernels of the polynomial and Gaussian types used for each news category.

## 3. METHODOLOGY

#### 3 .1. System Design

Following system design is proposed in this project to classify news articles for generating stock
trend signal.


```
News Collection
```
```
Text Preprocessing
```
```
Polarity Detection
Algorithm
```
```
(News, Polarity
score)
```
```
Document
Representation
```
```
Classification and Regression Analysis
(Build the model)
```
```
Time Series Forecast
```
```
Model testing and fine tuning
```
```
Plot time series and
predict close price
```
```
Plot Scoring of news
sentiment
```
```
Building WebApp and deploying
```
System Design
This design can logically be seen as three phases. Result of phase 1 is news
articles with its polarity score. This result is given as an input to the phase 2. In phase 2, text is
converted in tf-idf vector space so that it can be given to the classifier and regressor. Then these
models are programmed for the same data to compare results. At the end of phase 2, we
evaluate the results given by all classifiers and also test for checking classifier performance for
new news articles. In phase 3, we check for relationship between news articles and stock price
data. We plot both the data using plotly plots and record the results. Later we use all this information to build
a interactive Web Appllication using Dash. In the following sections,
each block of the design is explained.

3 .1.1. News Collection
We collected 100 stocks data for past ten years. This data includes major key events news 
articles of the company and also daily stock prices of company using yfinance API
for the same time period. Daily stock prices contain six values as Open, High, Low,
Close, Adjusted Close, and Volume. For integrity throughout the project, we considered
Close price as everyday stock price. We have collected this data from major news
aggregators such as news.google.com, newspaper3k, finance.yahoo.com.


3 .1.2. Pre Processing
Text data is unstructured data. So, we cannot provide raw test data to classifier as an input.
Firstly, we need to tokenize the document into words to operate on word level. Text data
contains more noisy words which are not contributing towards model building. So, we need to
drop those words. In addition, text data may contain numbers, more white spaces, tabs,
punctuation characters, stop words etc. We also need to clean data by removing all those words.
For this purpose, we created own stop-word list which specifically contains stopwords related to
finance world and also general English stop words. This stop words list contains general words 
including Generic, names, Date and numbers, Also, to ignore words that appear in only one or two documents, we are considering minimum
document frequency which considers words that appear in minimum three documents.
Stemming is also important to reduce redundancy in words. Using stemming process, all the
words are replaced by its original version of word. For example, the words ‘developed’,
‘development’, ‘developing’ are reduced to its stem word ‘develop’. Some of the pre-processing
is done before applying polarity detection algorithm. And some of them are applied after
applying polarity detection algorithm.

3 .1.3. Sentiment Detection Algorithm
For automatic sentiment detection of news articles, we are following Dictionary based approach
which uses Bag of Word technique for text mining. This method is based on the research of J.
Bean in his implementation of Twitter sentiment analysis for airline companies [6]. To detect the
polarity we used  VADER Sentiment analyzer, VADER ( Valence Aware Dictionary for Sentiment Reasoning) 
is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) 
and intensity (strength) of emotion and embedding text using Gensim Word2Vec these are a modern approach 
for representing text in natural language processing. TF-IDF Term frequency (TF) is how often a word 
appears in a document, divided by how many words there are and, inverse document frequency (IDF) 
is how unique or rare a word is., we need two types of words collection; i.e. positive words and negative
words. Then we can match the article’s words against both these words list and count numbers
of words appears in both the dictionaries and calculate the score of that document.
We created the polarity words dictionary using general words with positive and negative
polarity. Also addition to this, we used Finance specific words with its polarity using
McDonald’s research [16]. 


Here, we are considering one assumption as if the score of the document is 1, then we label it as
positive as we are considering two class problem for this implementation. As a result, we get
news collection with its sentiment score and polarity as positive or negative.

3 .1.4. Document Representation
In order to reduce the complexity of text documents and make them easier to work with, the
documents has to be transformed from the full text version to a document vector which
describes the contents of the document. To represent text documents, we are using TF-IDF
scheme. The higher tf-idf value a term gets, the more important it is. A high value is reached
when the term frequency in the given document is high and when there are few other documents
in the collection containing the given term/feature. This term weighting method tends therefore
to filter out common terms by giving them a very low value.

3 .1.5. Classifier Learning
As most of the research shows that Decision Tree, Random Forest and K Nearest Neighbours classification
algorithms performs good in text classification. So, we are considering all three algorithms to
classify the text and check each algorithm’s accuracy. We can compare all the results such as
accuracy, precision, recall and other model evaluation methods. All three classification
algorithms are implemented and tested with new data.


3 .1.6. Resgression Learning
As most of the research shows that Decision Tree, Random Forest and K Nearest Neighbours classification
algorithms performs good in text classification. So, we are considering all three algorithms to
classify the text and check each algorithm’s accuracy. We can compare all the results such as
accuracy, precision, recall and other model evaluation methods. All three classification
algorithms are implemented and tested with new data.

3 .1.7. System Evaluation
We divided the data into train and test set. Also, we created unknown data set for classifier and Regressor
to check accuracy against new data. We evaluated all the models performance by
checking each one’s accuracy, precision, recall, ROC curve area. The results of which are provided in the 
presentation.

3 .1.8. Testing with new Data and Building Web Application

News articles from Jan 2021 to April 2022 are used as unknown test set. When comparing
results of all models, KNN classifier and Random Forest Regressor performs well for unknown data.
Other algorithms also worked good but these performed the best. We used Dash to build a interactive
Web App for users. 

## 4. Model Evaluation

We tested the models using different testing options so that we can compare each method
against different scenarios. Following are the test options on which we tested our models.
- 5 - fold cross validation
- 10 - fold cross validation
- 15 - fold cross validation
- 70% Data split
- 80% Data split

## 5. CONCLUSION

Finding future trend for a stock is a crucial task because stock trends depend on number of
factors. We assumed that news articles and stock price are related to each other and, news may have capacity to 
fluctuate stock trend. So, we thoroughly studied this relationship and concluded that stock trend can be predicted 
using news articles and previous price history. As news articles capture sentiment about the current market, 
we automate this sentiment detection and based on the words in the news articles, we can get an overall news polarity.
If the news is positive, then we can compare with the closing price to verify that the news impact is good in 
the stock market, so more chances of stock price go high. And if the news is negative, then it may impact the 
stock price to go down in trend. 

## FUTURE WORK

We would like to extend this research by adding more company’s data and check the prediction
accuracy. For those companies where availability of financial news is a challenge, we would be
using twitter data for similar analysis. We can also incorporate similar strategies for algorithmic
trading.

## ACKNOWLEDGEMENTS

Authors would like to thank our guides, teachers, family and friends who supported in the
completion of this research project. Appreciating everyone who helped us knowingly or
unknowingly for this project.
```

## REFERENCES

[1] Anurag Nagar, Michael Hahsler, Using Text and Data Mining Techniques to extract Stock
Market Sentiment from Live News Streams, IPCSIT vol. XX (2012) IACSIT Press, Singapore
[2] W.B. Yu, B.R. Lea, and B. Guruswamy, A Theoretic Framework Integrating Text Mining and
Energy Demand Forecasting, International Journal of Electronic Business Management. 2011,
5 (3): 211- 224
[3] J. Bean, R by example: Mining Twitter for consumer attitudes towards airlines, In Boston
Predictive Analytics Meetup Presentation, 2011
[4] Yauheniya Shynkevich, T.M. McGinnity, Sonya Coleman, Ammar Belatreche, Predicting Stock
Price Movements Based on Different Categories of News Articles, 2015 IEEE Symposium
Series on Computational Intelligence
[5] P. Hofmarcher, S. Theussl, and K. Hornik, Do Media Sentiments Reflect Economic Indices?
Chinese Business Review. 2011, 10 (7): 487- 492
[6] R. Goonatilake and S. Herath, The volatility of the stock market and news, International
Research Journal of Finance and Economics, 2007, 11 : 53-65.
[7] Spandan Ghose Chowdhury, Soham Routh , Satyajit Chakrabarti, News Analytics and Sentiment
Analysis to Predict Stock Price Trends, (IJCSIT) International Journal of Computer Science and
Information Technologies, Vol. 5 (3) , 2014, 3595 - 3604
[8] Robert P. Schumaker, Yulei Zhang, Chun-Neng Huang, Sentiment Analysis of Financial News
Articles
[9] Győző Gidófalvi, Using News Articles to Predict Stock Price Movements, University of
California, San Diego La Jolla, CA 92037, 2001
[10] L. Breiman, Random forests. Machine Learning, 45(1):5-32, 2001
[11] Data Mining Lab 7: Introduction to Support Vector Machines (SVMS)
[12] Joachims T., Text Categorization with Support Vector Machines: Learning with Many Relevant
Features, European Conference on Machine Learning (ECML), Application of Machine
Learning and Data mining in Finance, Chemnitz, Germany, 1998)
[13] Kyoung-jae Kim, Financial time series forecasting using support vector machines,
Neurocomputing 55 (2013) 307 – 319
[14] Pegah Falinouss, Stock Trend Prediction using News articles, The Lulea University of
Technology, 2007
[15] https://en.wikipedia.org/wiki/Support_vector_machine
[16] [http://www3.nd.edu/~mcdonald/Word_Lists.html](http://www3.nd.edu/~mcdonald/Word_Lists.html)
[17] https://jeffreybreen.wordpress.com/2011/07/04/twitter-text-mining-r-slides/
