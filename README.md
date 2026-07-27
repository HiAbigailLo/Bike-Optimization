# Bay Area Bike Share Demand Prediction
Phase 1: 
- Built and tuned a Random Forest to predict hourly bike demand using trip-level data from Bay Wheels stations
- Compared 5 ML algorithms, simplified the feature set, and tuned hyperparameters to raise R² by 7.6% (0.48 → 0.52) and cut MAE by 4.5%

Phase 2: 
- Adding weather data and re-evaluating phase 1 with newer features
- Building a custom tree-ensemble model inspired by Random Forest, targeting a specific weakness of the vanilla approach for this data: `station_numeric` is a high-cardinality categorical feature encoded via arbitrary integer order

## Problem
Bike-share systems (BSS) have grown as an integral part of urban transportation, offering a low-cost, low-emission alternative for short trips and connections without a car. Demand imbalance, where bikes pile up at popular stations while origin stations empty out during peak hours, is a challenge for BSS worldwide. Previous literature identified Random Forests as one of the most commonly used and best-performing algorithms for predicting station-level demand (Albuquerque et al., 2021). This project is a case study on whether this established approach can reliably predict hourly demand in the Bay Area, using Lyft’s Bay Wheels System Data.

### Project Motivation
Prior work used a Random Forest model to rank features using dock counts sampled minute-by-minute across 70 stations from 2013-2015; a separate Negative Binomial regression model found time-of-day, temperature, and humidity to be the strongest predictors (Ashqar et al., 2019). This project focuses on a different prediction variable, demand, with a more recent snapshot of the system with 584 stations, with Random Forest as the predictive model. Trip-level data was cleaned in R,  decomposed into temporal features (hour, day of week, day of month, weekend/workingday/rush-hour indicators), and aggregated into a station-hour demand label, then modeled in Python.

## Results

| Model | Correlation | MAE | RMSE | R² |
|---|---|---|---|---|
| Baseline (Random Forest, 7 features) | 0.7009 | 1.2578 | 1.9047 | 0.4805 |
| Tuned (Random Forest, 4 features) | **0.7191** | **1.2009** | **1.8366** | **0.5170** |

- Model comparison: Random Forest outperformed k-NN, Decision Tree, and Linear Regression: correlation 0.70, 0.58, 0.56, and 0.19, respectively. 
- Dropping the `weekend`, `workingday`, and `is_rush_hour` binary indicators improved correlation slightly. However, they were redundant with the raw `hour`/`day_of_week` features.
- Station identity (`station_numeric`) alone accounts for 62% of the tuned model's feature importance

## Running the Code
Clean the raw trip data first (R):
```
clean_bike.Rmd → clean.csv
bike_analyses.Rmd → hourly.csv
```

Then install and run the Python pipeline in order:
```bash
pip install pandas numpy scikit-learn matplotlib
python bike_ml.py      # baseline model comparison
python feature.py      # feature selection
python parameter.py    # hyperparameter tuning + final model
```

## Dataset | Bay Wheels System Data
Trip-level data provided by Lyft's Bay Wheels regional bike-share program (San Francisco, Oakland, Berkeley, San Jose). https://www.lyft.com/bikes/bay-wheels/system-data 

## References

1. Albuquerque, V., Sales Dias, M., & Bacao, F. (2021). Machine learning approaches to bike-sharing systems: A systematic literature review. ISPRS International Journal of Geo-Information, 10, Article 62. https://doi.org/10.3390/ijgi10020062
2. Ashqar, H. I., Elhenawy, M., & Rakha, H. A. (2019). Modeling bike counts in a bike-sharing system considering the effect of weather conditions. Case Studies on Transport Policy, 7, 261–268. https://doi.org/10.1016/j.cstp.2019.02.011
3. Fricker, C., & Gast, N. (2016). Incentives and redistribution in homogeneous bike-sharing systems with stations of finite capacity. EURO Journal on Transportation and Logistics, 5(3), 261–291. https://doi.org/10.1007/s13676-014-0053-5
4. Yang, Z., Hu, J., Shu, Y., Cheng, P., Chen, J., & Moscibroda, T. (2016). Mobility modeling and prediction in bike-sharing systems. In Proceedings of the 14th Annual International Conference on Mobile Systems, Applications, and Services (pp. 165–178). Association for Computing Machinery. https://doi.org/10.1145/2906388.2906408

