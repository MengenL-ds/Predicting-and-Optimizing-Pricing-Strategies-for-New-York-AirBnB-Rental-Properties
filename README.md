# Predicting and Optimizing Airbnb Pricing Strategies for AirBnB Listings in Bristol, UK

For this project, I chose to work with the 2019 New York City Airbnb listings dataset to predict `reviews_per_month`, which serves as a proxy for listing popularity. This analysis provides an opportunity to understand what factors make a listing popular and how guest engagement can be influenced. By identifying key drivers of guest reviews, Airbnb and hosts can use these insights to optimize their listings and potentially forecast the popularity of new properties before they are posted.

The dataset includes a variety of features such as **listing ID, name, host ID and name, neighbourhood, neighbourhood group, latitude, longitude, room type, price, minimum nights, number of reviews, last review date, calculated host listings count, reviews per month**, and **availability_365**. Each feature offers unique insights into different aspects of a listing. For example, **neighbourhood_group** reflects location desirability, **room_type** affects guest preferences, and **price** plays a crucial role in booking decisions.

After reviewing the dataset, features such as neighbourhood, neighbourhood_group, number_of_reviews, and availability_365 appear potentially relevant to listing popularity. For example, listings in high-demand areas may receive more reviews, while price and room_type could influence booking frequency. Availability_365 may also affect how often a listing is reviewed. However, the true predictive value of these features will be evaluated through exploratory data analysis and model interpretation techniques such as SHAP values and feature importance scores. To build a robust predictive model, I will perform EDA, engineer relevant features, and compare several machine learning approaches.

_Below is a summary of what each feature represents:_

- `id`: Unique listing ID
- `listing_url`:
- `scrape_id`:
- `last_scraped`:
- `source`:
- `name`: Name of the listing
- `description`:
- `neighborhood_overview`:
- `picture_url`:
- `host_id`: Host's unique ID
- `host_url`:
- `host_name`: Name of the host
- `host_since`:
- `host_location`:
- `host_about`:
- `host_response_time`:
- `host_response_rate`
- `host_acceptance_rate`:
- `host_is_superhost`:
- `host_thumbnail_url`:
- `host_picture_url`:
- `host_neighbourhood`:
- `host_listing_count`:
- `host_total_listing_count`:
- `host_verifications`:
- `host_has_profile_pic`:
- `host_identity_verified`:
- `neighbourhood_cleansed`:
- `neighbourhood_group_cleansed`: 
- `accomodates`:
- `bathrooms`:
- `bathrooms_text`:
- `bedrooms`
- `beds`
- `amenities`:
- `price`:
- `minimum_nights`
- `minimum_minimum_nights`
-
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `room_type`: Type of space offered (e.g., entire home, private room)
- `price`: Price per night in dollars
- `minimum_nights`: Minimum nights required for booking
- `number_of_reviews`: Total number of reviews received
- `last_review`: Date of the last review (YYYY-MM-DD)
- `reviews_per_month`: Average number of reviews per month
- `calculated_host_listings_count`: Number of listings managed by the host
- `availability_365`: Number of days available for booking in 2019

This project aims to uncover actionable insights for hosts and Airbnb by leveraging data-driven approaches to predict listing

