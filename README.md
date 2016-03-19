# IntroToDataScience
In this course I have done four projects:


## 1. Twitter Sentiment Analysis

###Our topic – “iPad Pro”
  
  Motivation:
  
  Apple recently introduced its new iPad line: “iPad Pro” in its latest September special event.
  
  Twitter - popular social media site: 
  
   over 300 million active monthly users
   
   over 500 million tweets per day
   
  Purpose: To examine the future of iPad Pro & consider use of the data from a marketing perspective
  

## 2. NBA Ground Analysis

Obtained a list of players information through using py-Goldsberry.

Accessed the API that the NBA uses for stats.nba.com.

Collected data about individual NBA players via Shots API.

PlayerProfile API & PlayerDashPtShotLog API: Help explore the data with detailed players information (Running additional experiments).


## 3. Yelp Recommender

Like many prosperous internet companies, Yelp amasses data at an amazing rate.  As part of a data hackathon challenge, Yelp makes part of this data set available to students who are interested in analyzing it and attempting to answer a business intelligence question as part of a competition. This data is provided in .JSON format, and includes information on hundreds of thousands of users, users’ reviews, and businesses. 

In order to begin analysis on the data to answer our question, we chose the collaborative filtering method of generating recommendations out of three popular recommender systems. Collaborative filtering has the advantage of learning market segments, though it has problems with cold starts (cannot be used with insufficient data). Since Yelp already has a data set, cold starts would not be a problem. We also considered content-based systems, which have the advantage of not requiring community and can compare between items, but require content descriptions and also suffer with cold starts. The final type of system we considered was knowledge-based, which creates deterministic recommendations and assured quality, but does not react to short term trends. After carefully weighing the pros and cons of the three types of recommender systems, we deemed the collaborative filtering system the best fit for the Yelp data set.
	
We used Spark Core API and SQL to handle the data. Apache Spark is a fast, generalized engine for large-scale data processing. We chose it for its ease of use, speed, and unified engine capabilities. It is considered an up and coming competitor to Hadoop. which for some processes like logistic regression, it can run up to a hundred times faster than.
	
Collaborative filtering is a way of generating recommendations with large data sets that are typically user centered. It analyzes user rankings and reviews for similarities in their past rankings and reviews and uses those similarities to predict rankings and reviews that a user hasn’t yet done. Since the reviews and the rankings represent separate data sets, we created two separate collaborative filtering recommendation models: one focusing on the review based data and one on the ranking based data.
	
For the first part, we focused on user ranking data (see figure 2 for an example matrix). We used the memory-based variant of this technique, which focuses on the user rating data to compute similarity. In this method, the Pearson correlation, see below, equation gives us the similarity of two users.
	
Implementing the collaborative filtering method based on review similarity was done using a similar process, but with a focus on lexical similarity between users. The variables represented distinct words in the reviews rather than rankings from individual businesses (see figure 3 for an example table of this data) and ran the same process on that.
	
A final part of this process was neighbor selection. For a given active user, we needed to select a number of correlated users to serve as a source of predictions: the number of nearest neighbors. There are two common approaches, both of which we experimented with. First, the standard approach, to choose a set number n users’ ranks based on similarity weights; second, to include all users whose similarity weight is above a given threshold. 
	
We used two different accuracy measures to examine the error rate between our two different collaborative filtering recommendation models. The mean absolute error computes the deviation between predicted ratings and actual ratings. The root mean square error is similar to the mean absolute error, but it places more emphasis on larger deviation. 
	

## 4. Twitter and Stock Index Granger Causality

Social media: the easiest & fastest way to transmit and receive information.

Twitter: Aggregate of tweets could be seen as an indicator of collective mood.

Data scientists have been made several attempts to examine Twitter’s predictive potential of consumer purchasing by observing users’ mood.

“Twitter mood predicts the stock market” by Bollen, Mao, and Zeng in 2011.

Our Analysis: Analyze Twitter data through sentiment analysis & compare it with instead with the SP500 and Nasdaq index (4 different tests)

