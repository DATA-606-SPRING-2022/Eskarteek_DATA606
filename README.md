# DATA606 Capstone Project

##                           Risk Management for any Organization using data analytics and Sentiment analysis 

What is your issue of interest (provide sufficient background information)?


* This project can be used to explore, analyze, and mitigate risk from various perspectives for an Organization. It provides risk analysis on a real-time basis to detect potential hazards and to act faster, thus mitigating the risk. Using inputs like twitter tweets about the Organization,the CEO or the products they offer. I will design an end to end system which updates on daily basis showing top risks on a given day, its impact on stock market and website traffic. The system would segrigate these risks by considering all the factors into different priority levels and provide results in a user dashboard. Also, these risks would be stored on a daily basis for furthur analysis to find frequently occuring problems in the system.These issues can be for example User interface related issues, issues related to a perticular product or an individual and so..on. 



<img width="585" alt="Screen Shot 2022-02-07 at 4 11 56 PM" src="https://user-images.githubusercontent.com/98825247/152872680-3414a92c-6f7c-4115-a60d-b732e8489e24.png">




Why is this issue important to you and/or to others?

* This kind of a technique is important for any Organization which has a social media presence (which i assume most companies have) to understand problems faced by their end users. 

What questions do you have in mind and would like to answer?

* I would like to look at the impact of my idea when used in real-time. Its important for me to understand about what level can the risks be mitigated using a system like this. I would also like to look at the impact of social media platform like twitter on an organization and how important is it to analyse tweets in risk management and mitigation.


Where do you get the data to analyze and help answer your questions (creditability of source, quality of data, size of data, attributes of data. etc.)?

* The data used for this project is fetched on daily basis from Twitter Api (Twitter tweets), SimilarWeb Api(website traffic information) ,Yahoo finance Api (financial information). These are Api's which are legally allowed to scrape data from websites. 


What will be your unit of analysis (for example, patient, organization, or country)? Roughly how many units (observations) do you expect to analyze?

* The unit of analysis here is an Organization. With my current work, my units of observation are sentiment of twitter tweets, impact on stock market and website traffic.


What variables/measures do you plan to use in your analysis (variables should be tied to the questions in #3)?

* I would like to look at sentiment scores, website traffic data like new customers, bounce rates,.. and yahoo financial information like impact on stock prices, polarity of financial news, and more. 

What kinds of techniques/models do you plan to use (for example, clustering, NLP, ARIMA, etc.)?

* NLP (Sentiment analysis)

How do you plan to develop/apply ML and how you evaluate/compare the performance of the models?

* I would like to use VADAR sentiment analyser to analyse sentiment in the data. User Keyword extraction techniques like TF-IDF on negetive tweets to find out the most important issues to be solved on a given day. 

What outcomes do you intend to achieve (better understanding of problems, tools to help solve problems, predictive analytics with practicle applications, etc)?

* I would like to understand the impact of this project and predictive analytics in real-time application.


## Instructions

Please use the README.md profile for the proposal. Write it in your personal account's repository. After completion and before submission, refresh the contents in the class organization using "Fetch Upstream". Submit the link to the repository in the class organization.
