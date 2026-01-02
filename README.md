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
