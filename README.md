# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone: Sentiment Analysis for Shopee
Grace Chia

### Problem Statement

Shopee is a fast-growing e-commerce platform in Southeast Asia. In order to build an enjoyable online shopping experience, Shopee's product managers would first need to identify customer sentiment and understand users' pain points.

As such, the data science team has been tasked to classify the positive and negative reviews on Shopee Singapore's Google Play using Natural Language Processing (NLP). The model that achieves the highest accuracy and recall on the validation set will be selected for production. The team will be using topic modeling to identify the key pain points among dissatisfied customers. This will enable product managers to address their immediate needs.

---

### Executive Summary

The data science team web scraped 4,461 unique user reviews on Shopee's Google Play. Classical machine learning and deep learning NLP approaches were explored when building a sentiment classification model for Shopee reviews. The Voting Classifier was eventually selected as our production model as it achieved the highest accuracy and recall on the validation set. An ensemble of a TF-IDF Logistic Regression and a TF-IDF SVC, the Voting Classifier attained an accuracy of 0.904 and a recall of 0.87 on the validation data. When scored on the test set, the production model achieved 0.892 on accuracy and 0.87 on recall, outperforming the baseline accuracy of 0.61. We can thus conclude that the model generalises well on unseen data. This model serves as a good start for Shopee's product managers to classify sentiments and understand customers' pain points through Google Play reviews. The model was deployed to [Heroku](https://shopee-sentiment-analysis.herokuapp.com/).

The data science team has also identified 3 pain points among dissatisfied customers through topic modeling. These pain points include a poor payment experience, fraudulent sellers and items, and slow app performance. Addressing these pain points would not only promote an enjoyable shopping experience for customers, but would also differentiate Shopee from its competition.

Customers are most frustrated when they run into payment issues during checkout. The payment was either denied or the payment page simply took too long to load. Product managers will need to work closely with software engineers to enhance the stability and reliability of its payment gateway service.

Several customers have dealt with fraudulent sellers on Shopee. Identifying and blocking fraudulent sellers at the early stages is the key to fraud prevention. This involves tracking a range of behavioural pattern among sellers and combing through their digital identity for any signs of suspicious behaviour. Shopee could also improve its buyers' protection scheme to regain consumers’ trust.

Users have reported that the app hangs or crashes at the login and payment page. As such, developers will need to optimise its traffic scheduling platform's latency to accommodate high traffic during big sale events.

---

### Background

According to research done by [Google, Temasek and Bain](https://www.thinkwithgoogle.com/intl/en-apac/tools-resources/research-studies/e-conomy-sea-2019-swipe-up-and-to-the-right-southeast-asias-100-billion-internet-economy/), Singapore's e-commerce market was worthed US$2 billion in 2019, and is expected to grow 3.5 times to reach US$7 billion by 2025. High mobile penetration and fast internet connectivity, will be the key drivers of e-commerce growth. Furthermore, as consumers were encouraged to stay home as a result of COVID-19, many have started to grow accustomed to shopping online. This may just be the start of a new norm in consumer behaviour, where more start to shift from offline to online shopping.

While e-commerce presents huge opportunities, competition has been on the rise, with the key players being Lazada, Shopee and Qoo10. In efforts to build customer engagement and grow their share of wallet, competitors have improved the checkout experience, introduced gamification and overcame the challenges of last-mile delivery. Given stiff competition, it is crucial that Shopee continues to delight customers and address pain points in the online shopping experience. The sentiment analysis tool will enable Shopee's product managers to classify user sentiment accurately and understand the pain points.

---

### Methodology

**Data collection**: Users can rate [Shopee's app](https://play.google.com/store/apps/details?id=com.shopee.sg&hl=en_SG) on Google Play with a star rating and review. The ratings are on a 5-point scale, with 1 being the lowest score and 5 being the highest score one could possibly give. Since the goal of our project is to predict if a review has a positive or negative sentiment based on textual data, we decided to scrape real user reviews on Google Play. The scraping was done using [Google-Play-Scraper](https://github.com/JoMingyu/google-play-scraper) to collect users' reviews and ratings on Shopee's app. A total of 4,461 unique reviews were collected.

**Data cleaning**:
- Remove duplicated reviews
- Remove reviews that do not have any meaningful words
- Remove reviews that are non-English or gibberish

**Pre-processing**:
- Remove HTML tags
- Use regular expression to remove special characters and numbers
- Lowercase words
- Use NLTK to remove stopwords
- Remove frequently occurring words that appear in both positive and negative sentiments, like 'app', 'shopee', 'item', 'seller', 'bad'. Removing these words led to a 1 and 2 percentage point improvement in our model's accuracy and recall rate, respectively.
- Use NLTK to stem words to their root form. Note that the model returned better accuracy when we used stemming, rather than lemmatizing.

---

### Key Insights from EDA

**Time frame of the reviews written**  
The number of reviews for Shopee on Google Play has increased across all ratings (1-5 stars), between Jan and April 2020. This may likely be a result of the recent rise in e-commerce purchases. As the COVID-19 outbreak resulted in the forced closure of many brick-and-mortar stores during Singapore's circuit breaker period, many consumers started to turn to online shopping instead.

**Number of thumbs up received**  
Negative reviews with 1 or 2-star ratings receive more thumbs up on average, than positive reviews. This may suggest that several others face the same issues as those who have written these negative reviews.

**Number of meaningful words**  
The average number of meaningful words in a negative review (15 words) is higher than that in a positive review (7 words). There is also a noticeable higher variance in the number of meaningful words among negative reviews than positive reviews, suggesting that dissatisfied customers are more likely to write longer reviews.

**Barplots: Top uni-grams and bi-grams**  
'Use', 'time' and 'order' are the top 3 most frequently occurring uni-grams in negative reviews. 'Customer service' is the top bi-gram seen among negative reviews. We can thus infer that users are somewhat dissatisfied with Shopee's customer service.   

'Good', 'shop' and 'easi' are the top 3 most frequently seen uni-grams in positive reviews. The bi-grams give us some context to the word 'easi', where it probably refers to an 'easy to use online shopping platform'.  

**VADER sentiment analysis**  
Given a compound score threshold of 0.175, VADER is able to correctly classify 80% of sentiments. As Shopee's product managers would also like to prioritise the identification of negative reviews so that they can fix immediate problems if necessary, achieving a decent recall rate is important. VADER is able to correctly classify 74% of actual negative reviews.

---

### Data Dictionary

An overview of the features in our dataset.

| Feature     	| Type 	| Description                                                       	|
|:-------------	|:------	|:-------------------------------------------------------------------	|
| content     	| obj  	| Raw text containing user reviews                                  	|
| content_stem 	| obj  	| Pre-processed text for modeling                                   	|
| score       	| int  	| No. of star ratings the user gave (1-5)                           	|
| target      	| int  	| Target variable <br>Postive sentiment: 0<br>Negative sentiment: 1 	|

---

### Model Evaluation

The table below provides an overview of the models' performance, sorted by accuracy and recall on the validation set in descending order.

|                                                             	| Accuracy on Training Set 	| Accuracy on Validation Set 	| Recall on Validation Set 	|
|:-------------------------------------------------------------	|:--------------------------:	|:----------------------------:	|:--------------------------:	|
| Voting Classifier (TF-IDF Logistic Regression & TF-IDF SVC) 	| 0.902                    	| 0.904                      	| 0.87                     	|
| TF-IDF & SVC                                                	| 0.901                    	| 0.899                      	| 0.87                     	|
| Count Vectorizer & Naïve Bayes                              	| 0.898                    	| 0.898                      	| 0.83                     	|
| Bidirectional LSTM                                          	| 0.916                    	| 0.896                      	| 0.82                     	|
| TF-IDF & Logistic Regression                                	| 0.906                    	| 0.892                      	| 0.81                     	|
| TF-IDF & Naïve Bayes                                        	| 0.904                    	| 0.882                      	| 0.79                     	|
| Count Vectorizer & SVC                                      	| 0.900                    	| 0.872                      	| 0.75                     	|
| Count Vectorizer & Logistic Regression                      	| 0.882                    	| 0.861                      	| 0.71                     	|

|                                                                               	| Accuracy on Test Set 	| Recall on Test Set 	|
|-------------------------------------------------------------------------------	|----------------------	|--------------------	|
| Production Model: Voting Classifier (TF-IDF Logistic Regression & TF-IDF SVC) 	| 0.892                	| 0.87               	|

The Voting Classifier was selected as our production model as it achieved the highest accuracy and recall on the validation set. An ensemble of a logistic regression and SVC, the Voting Classifier attained an accuracy of 0.904 on the validation data and a recall of 0.87. When scored on the test set, the production model attained an accuracy of 0.892 and recall of 0.87. As the model is able to correctly predict 87% of the actual negative reviews on Google Play, Shopee's product managers will now be able to identify negative reviews and understand the pain points among dissatisfied customers.

The winning class on the Voting Classifier was determined through soft voting. As the SVC had a better recall rate than the Logistic Regression, we gave a higher voting weight to the SVC than the Logistic Regression. Given that the base estimators (Logistic Regression and SVC) of our Voting Classifier are rather diverse algorithms, the ensemble gave a better accuracy score than the best of the base estimators. The ensemble aggregates the predictions of the base estimators and helps to cover for the weaknesses of the individual models, which in turn reduces the production model's bias.

---

### Recommendations
We have identified 3 pain points among dissatisfied customers through topic modeling. These pain points include a poor payment experience, fraudulent sellers and items, and slow app performance. The recommendations are as follows:

**Poor payment experience**

Customers are most frustrated when they run into payment issues during checkout. The payment was either denied or the payment page simply took too long to load. Although customers are given the option of paying via bank transfers, the process is not entirely seamless. Some were not able to upload the receipt for verification due to a platform error. A handful of customers have also mentioned in their Google Play reviews that they were not able to apply their voucher or discount codes at the point of checkout. We can imagine the disappointment that customers face when they were not able to enjoy the promotional prices that were promised.

Consumers today demand a fast, convenient and secure payment experience. As such, the software engineers at Shopee will need to look into enhancing the stability and reliability of its payment gateway service. Given that card payments are processed through third-party providers, engineers will need to figure out if these payment issues are surfacing from the third-party providers' end, or as a result of Shopee's internal systems. Reducing friction in the payment process will definitely help to reduce cart abandonment.

**Fraudulent sellers and items**  

Several customers have reported fraudulent seller behaviour. These scams include not having received the items ordered, or receiving counterfeit, defective products. When these issues were raised to Shopee's customer service, the problem was either not resolved in a timely fashion, or customers did not receive their refunds. This has thus disappointed and angered many affected customers and several feel that Shopee favours the sellers over the buyers.   

Building a robust model that identifies and blocks fraudulent sellers at the early stages will help to prevent seller fraud and preserve buyers' trust. This involves tracking a range of behavioural pattern among sellers and combing through their digital identity for any signs of suspicious behaviour. While the team continues to build better fraud detection algorithms, it will not be possible to completely stamp out frauds. Shopee could improve its buyers' protection scheme to boost buyers' confidence. Given Shopee's escrow service, in the event when the order is not delivered or the items do not match the seller's description, Shopee should reimburse the full cost to the customer.

**Slow app performance**

Users have reported that the app hangs or crashes at the login and payment page. This is an issue of latency, meaning that there is a delay between an action and a response. Several other reviews have also mentioned that the app's functionalities had slowed down considerably after updating the app. Some were reportedly not able to load Shopee's games after updating the app, and were thus upset that they could not collect Shopee coins. These coins can be used to offset the cost of the next purchase.

Developers will need to optimise Shopee's traffic scheduling platform's throughput and latency. Ensuring smooth running of the platform is crucial especially during big sale events where the server runs the risk of being overloaded due to high traffic. Developers will also need to stress test before releasing new app updates to ensure that the app's performance is not compromised.

---

### Conclusion

The production model is a Voting Classifier consisting of an ensemble of a Logistic Regression and SVC. Given its relatively high accuracy (0.892) and recall (0.87) on the test set, we can conclude that the model generalises well on unseen data. This model serves as a good start for Shopee's product managers to classify sentiments and understand customers' pain points through Google Play reviews.

The data science team has also identified 3 pain points among dissatisfied customers through topic modeling. Product managers will need to work closely with software engineers and developers to build a seamless payments experience, crack down on fraudulent sellers and improve app performance in terms of latency.

Despite the model's relatively high performance, there is still room for improvement. Misclassifications tend to occur when users leave mixed reviews on Shopee. For instance, when the user praises Shopee for being 'good' in one aspect, but indicates that there are other areas of improvement, the model classifies this is a positive review. Sentiments are inherently subjective. However, given that the user is dissatisfied with certain aspects of the shopping experience, it is worth classifying such reviews as negative sentiments. This will allow product managers to be informed of these pain points. In order to reduce misclassifications, we can train the model on reviews from the Apple App Store too. Given that we have also identified customers' pain points, the next step would be to use these topics as features in our sentiment classification model. This may help to improve the model's accuracy and recall.
