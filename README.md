# AIRBNB LISTING PRICE AND RATING PREDICTION 

## Introduction and Motivation 

Airbnb is a marketplace for short term rentals, allowing you to list part or all your living space for others to rent. The company itself has grown rapidly from its founding in 2008 to a 30-billion-dollar valuation in 2016 and is currently worth more than any hotel chain in the world. One challenge that Airbnb hosts face is determining the optimal nightly rent price. In many areas, renters are presented with a good selection of listings and can filter on criteria like price, number of bedrooms, room type, and more. Since Airbnb is a marketplace, the amount a host can charge on a nightly basis is closely linked to the dynamics of the marketplace.  
If the host charges above the market price, then renters will select other affordable alternatives. If the nightly rent price is set too low, the hosts will miss out on potential revenue. So, we use machine learning models to predict the optimal prices that the hosts can set for their properties. This is done by comparing the property with other listings on a variety of parameters like location, property size and other demographics. Also, based on the listing specifications and prices, the hosts would like to know what review rating they can expect or weigh how their listing compares to other similar ones and if they have a price premium with respect to other similar listings or are losing out on rating because of their pricing.

## Dataset Description  

The main dataset for the listing price and the rating prediction of the host property is obtained from Kaggle. The features describing the properties of a listing in the dataset include number of bathrooms & bedrooms, number of reviews, review score, neighborhood, GPS coordinates, description of the listing etc. Table 4 shows a list of columns that belong to the two groups, which can be found in the appendix.  
The dataset has a mix of Continuous, categorical, Nominal, ordinal and Boolean features. log_price, number of reviews, review score rating are continuous variables; Property_type, Room_type, Amenities, Description, Bed_type, Name of the rental, Neighborhood are all categorical variables; Accommodates, Bathrooms, Bedrooms fall under the group of Nominal variables; ID is an Ordinal variable; while the Cleaning fee, Host_has_profile_pic, Host_identity_verified, Instant bookable are all Boolean category variables, while the remaining are either the Time (first review, host since) or geographical variables (latitude and longitude). 
Original dataset has 74,111 instances and 29 features including target variables log_price (which is the per night price of the listing in logarithmic scale) and review_scores_rating (which is the review rating score given by the users). 

## Data Pre-processing:

During the initial Exploration, we found out there were some issues with original dataset such as missing values, non-English characters in description field and mismatching datatypes. 

*	Some of the fields in dataset had missing values as shown in Table 5 in Appendix. We handled missing values in below 2 ways:
>*	Removed rows which had missing values.
>*	Host_response_rate which is one of the important features in determining review_scores_rating, had 18299 missing values. Rather than removing missing values, we imputed those values using median of the field.
*	Non-English characters were removed from the Description field during text preprocessing.
*	Features like first_review and last_review were changed to correct datatypes (i.e. Date).
*	Feature transformation: 
>*	Dummy encoding: Dummy variables were created for categorical variables such as room_type, city, property_type, bed_type and cancellation_policy.
>*	Range scaling: scale range is adjusted for the predictors used for all regression models built for listing price prediction.
*	We calculated correlation among input variables as well as correlation between target and input variables to find which predictors are influencing listing price more.
*	Removal of outliers: removed prices less than $10 and more than $600, as these prices were not having enough support.

## Feature Engineering 
We added more dimensions to original dataset to improve the real-world relevancy in predicting the listing price and review ratings.

### Amenities
All the listings had a feature called amenities which provided the list of all facilities available in the listing. We collated the unique amenities across all the listings and grouped them into categories according to business logic. After the amenity groups were created, one hot encoding was performed and each group for transformed into a dummy variable. So, for ex, even if one facility from the group is present in the listing, the corresponding amenity group would be activated and encoded with a 1.

### Text feature extraction
All Airbnb listings are provided with the detailed description of the accommodations in a text field “Description”. This field is very useful for hosts to highlight important and attractive features of their properties. We found out how this field impacts the price as well as user review ratings by creating 3 new features out of the description field:

1.	Sentiment Intensity score – VADER sentiment intensity analyzer was used to identify the sentiment of each description. The range of values are from -1 to 1. VADER gives a cumulative score by adding the sentiment score of each word in the description. Strong positive words get a +ve score and similar –ve score is given to negative words. Negation is also incorporated while giving the score. Capitalized words and punctuations affect the overall score too.
2.	Description Length - The length of the description was obtained by performing a word count on this field. This feature was engineered to find out if longer descriptions yielded in a higher rating and were people willing to pay the extra money just because the host was very friendly and enthusiastic.
3.	Topic modelling – All the description records were combined to form the corpus and Latent Dirichlet Allocation was employed on the corpus to find out the top 5 topics. After we got the top words in the 5 topics, 4 were meaningful. These 4 topics were now labelled after looking at the topic word distribution carefully. The topics were -
>1.	Description about the listing – like high floored, furnished, etc.
>2.	Transport – nearest transport facilities from the listing
>3.	Attractions – Nearby places of interest from the listing
>4.	Amenities – facilities provided by the host
 The probability of each of the topics in every description was calculated and added as features in the dataset.
### Distance addition
We added some external data to bring in more transport and attraction features which will help in predicting the price better. The geo co-ordinates of all the railway stations in the 6 cities were obtained and the distance to the closest station from a listing was added as a feature. Also, the distance from the listing to the top 3 attractions in the city were added features. These metrics were added as we found while performing topic modelling on the description that transport and attraction words were among the ones with a high probability.

## Modelling – Listing Price

### Methodology
Since the listing price is continuous in nature, we used linear as well as tree based regressors to build prediction models. 
  Below are the models we evaluated to predict the listing price:

*	Linear Regression 
>*	Base Model (with Raw Data)
>*	Base Model + Amenities
>*	Base Model + Amenities + Description Related Fields
>*	Base Model + Amenities + Description Related Fields + Distances
*	Random Forest Regressor
*	SVM Regressor
>*	Linear Kernel
>*	rbf Kernel
>*	Polynomial Kernel
*	Light GBM Regressor

### Model Comparison & Results Interpretation
For price prediction, we started with a linear regression model with independent features available in the original dataset (which is our base model). Then we added the new features which we got from feature engineering to our base model and we were able to improve the overall accuracy for linear regression. Then we also evaluated the performance of tree based regressors as well as SVM regressors on this data. We saw that the tree based regressors were performing very well while the SVM based regressors took a lot of time to train and their performance was not as good as the random forest regressor. 
As tree based regressors were performing better for our data, we tried a boosting based approach to maximize the accuracy. We used Light GBM which is a very popular Gradient Boosting method. It uses a leaf-based approach as compared to the branch-based approach seen in other boosting algorithms, which makes it faster and lighter on the memory requirements. Also, our dataset was having more than 10,000 records which was easily satisfying the basic requirement of LGBM.

![Pricing_Modelling_Results](https://github.com/ManuGMathew/AML-Project---Airbnb/blob/master/images/price_results.png)

Some important insights that we could gather from this analysis are:

*	When Accommodates increases by 1 person the Log price increases by 1.303 times 
*	When Distance to station decreases by 1 km the Log price increases by 1.5236 times
*	Room Type being Entire Home/Apartment increases Log price by 0.9192 times

### Model robustness
Robustness characterizes how effective a model is when it is being tested on a new independent but a similar dataset. We wanted to test the scope of expansion in business for a host into a new city without any listing price tool available for the new city. The evaluation metric used is the difference between the training and test error. The following steps illustrate how model robustness was tested:

1.	Train the model for New York
2.	Test the model for a new city – Chicago
3.	Check the difference in error between the above model and a model trained and tested on Chicago

Also, since we also wanted to incorporate some business context while predicting for a new city, a scaling factor was introduced which considered the real estate mean prices of both the cities.

## Modelling – Rating 
### Methodology
We used the user rating received by a listing on AirBnB as the target and built prediction models to identify any disparities in rating between similar listed attractions priced differently. The rating scores on the AirBnB listing datasets are largely left skewed and regression methods might not be best suited for this task. 

Since ratings are a well-known parameter in the hotel business world in a categorical ordinal scale, we have transformed this prediction problem from a regression to classification. This is done by transforming the range of continuous target variable rating values into a set of intervals that was used as discrete classes. We tried performing discretization of the target variables using three methods, namely binning the values into equally probable intervals (EP) at quantiles, Equal width intervals (EW) and using K-Means to minimize the sum of distances of each element to the corresponding point’s bin’s centroid (KM). 
Looking at the distribution of the bins (as shown in Appendix), we see that the equally probable bins are best suited to be modelled as a classification problem, as they provide immunity against class imbalance, so we selected this method and split the data into 3 bins (Rating range: 0-93,93-98,98-100).

### Model Comparison & Results Interpretation
From the model comparison below we can see that, Random Forest is giving better accuracy compared to other models. Random Forest is better than Logistic Regression over all metrics – Precision, Recall and F1 measure.
![Rating_Modelling_Results](https://github.com/ManuGMathew/AML-Project---Airbnb/blob/master/images/rating_results.png)

### Feature interpretation and insights
Important features selected using the Random Forest for review score rating are shown in below figure. As expected, price, proximity of the listing to station and other attractions, facilities such as Bathroom, bedroom and number of accommodates, host related parameters such as host response rate and days since hosted and amenities like 24-hour check-in play an important role in determining review score rating of the listing. We did hyperparameter tuning using GridsearchCV to select optimum parameters.

## Conclusion
In this study, we used different machine learning algorithms to predict Airbnb's listing price and rating.

*	Linear Regression with engineered amenities, description and distance measures yielded the best LR model with a RMSE of $63.4. This showed the importance of adding external data to the existing set and engineering the features to predict the target even better.
*	Light GBM Regression yields the lowest RMSE among all regression models, which is $56.44. Overall, all models have RMSE in the range of $56 – $69. Also, the most important features in our best model (LGBM – R) were the engineered variables – Distances, topics and sentiments.
*	Among all the classifiers, we chose the best one in discriminative and tree-based classifier. Logistic regression and Random forest classifier yielded the best results for predicting the rating.
*	Random forest is better than Logistic Regression over all class-wise metrics of Precision, Recall and F1. The relative error, distance of spillage in classified points is lesser with Random Forest.

## References

1.	Torgo, L. and Gama, J. (2018). [online] Regression as Classification. Available at: 
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.5374&rep=rep1&type=pdf [Accessed 20 Oct. 2018]. 
2.	AirBnB listings in major US cities. (2018). Kaggle.com. Retrieved 22 October 2018, from https://www.kaggle.com/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml 
3.	Inside Airbnb. Adding data to the debate. (2018). Inside Airbnb. Retrieved 22 October 2018, from http://insideairbnb.com/get-the-data.html 
4.	11 Toughest Cities to Book an Airbnb (and Tips on When to Visit). (2015). Beyond Pricing Blog. Retrieved 22 October 2018, from https://blog.beyondpricing.com/11-toughest-citiesto-book-an-airbnb/ 





