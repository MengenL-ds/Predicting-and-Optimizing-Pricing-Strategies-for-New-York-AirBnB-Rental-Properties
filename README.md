# Predicting and Optimizing Airbnb Pricing Strategies for AirBnB Listings in Bristol, UK

For this project, I chose to work with the 2019 Bristol, United Kingdom Airbnb listings dataset to predict `reviews_per_month`, which serves as a proxy for listing popularity. This analysis provides an opportunity to understand what factors make a listing popular and how guest engagement can be influenced. By identifying key drivers of guest reviews, Airbnb and hosts can use these insights to optimize their listings and potentially forecast the popularity of new properties before they are posted.

The dataset includes a variety of features such as **listing ID, name, host ID and name, neighbourhood, neighbourhood group, latitude, longitude, room type, price, minimum nights, number of reviews, last review date, calculated host listings count, reviews per month**, and **availability_365**. Each feature offers unique insights into different aspects of a listing. For example, **neighbourhood_group** reflects location desirability, **room_type** affects guest preferences, and **price** plays a crucial role in booking decisions.

After reviewing the dataset, features such as neighbourhood, neighbourhood_group, number_of_reviews, and availability_365 appear potentially relevant to listing popularity. For example, listings in high-demand areas may receive more reviews, while price and room_type could influence booking frequency. Availability_365 may also affect how often a listing is reviewed. However, the true predictive value of these features will be evaluated through exploratory data analysis and model interpretation techniques such as SHAP values and feature importance scores. To build a robust predictive model, I will perform EDA, engineer relevant features, and compare several machine learning approaches.

This project aims to uncover actionable insights for hosts and Airbnb by leveraging data-driven approaches to predict listing price

## Running the App
Navigate to app.py in deploy directory
```
python deploy/app.py
```
The server will start at:
http://localhost:3000

NOTE: Make sure you enter the correct type value for each feature, an example can be found at deploy/payload.json

