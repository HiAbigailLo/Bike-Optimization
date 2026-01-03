# Bike-Optimization

This project aims to determine the demand for bikes at Bay Area Transit stations at different times of day.

Log of changes are as follows:
#### 01/01/2026
- Added Bay Wheels trip data from June 2025
- Added bike_clean.Rmd 
  - Code to clean initial bay wheels data
- Added clean.csv
  - product of baywheelsdata.csv processed with bike_clean.Rmd
- Added hourly.csv
  - product of clean_df being processed by bike_clean.Rmd last chunk
    - sorts demand by station and hour

#### 01/02/2026
- Added bike.py to perform a random forest

#### 01/03/2026
- Added models.py
  - created a baseline random forest model 

- Added models2.py
  - trains Random Forest, k-NN (k=10), k-NN (k=5), Decision Tree, Gradient Boosting, and Linear Regression models
  - Lists all of the models' Correlation, MAE, RMSE, R², Train_Time_sec, Predict_Time_sec to compare

- Added feature.py
  - performs feature selection

- Added parameter.py
  - performs parameter tuning

  