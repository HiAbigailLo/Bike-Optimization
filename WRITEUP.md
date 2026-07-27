# Bay Area Bike Share Demand Prediction: Phase 1 Report

## Hypothesis
There is a machine learning model trained on features (hour of day, day of week, station identifier, and day of month) that can successfully predict demand on a station level in the Bay Area, replicating prior work on bike-sharing systems across the world.

## Final Model

| Model | Correlation | MAE | RMSE | R² |
|---|---|---|---|---|
| Baseline (7 features, all params) | 0.7009 | 1.2578 | 1.9047 | 0.4805 |
| Feature Selection (4 features, original params) | 0.7021 | 1.2556 | 1.9012 | 0.4824 |
| Tuned Parameters (4 features, optimized params) | 0.7191 | 1.2009 | 1.8366 | 0.5170 |

- `n_estimators`: 300
- `max_depth`: 40
- `min_samples_split`: 10
- `min_samples_leaf`: 4
- Best cross-validation MAE: 1.2147

MAE was chosen as the validation metric over RMSE because it weighs each error equally, rather than letting rare, high-demand stations dominate the score, which better reflects predictions for situations where a two-bike prediction error is as important as one at a busy station.

## Discussion

The most predictive feature was `station_numeric`, with a 0.624 feature importance. In comparison, the next-highest feature was `hour` with a feature importance of 0.190, followed by `day_of_week` (0.100) and `day_of_month` (0.086). The station effect is stronger than the combined effect of temporal features, indicating that the station a trip starts in encodes underlying geographical factors. This could be interpreted as meaning that the location matters more for predicting hourly demand.

The final model backs up the hypothesis; a Random Forest was able to be trained on four features and successfully predicts station-hour demand (R² = 0.517, correlation = 0.72), matching the pattern of prior work applying similar models to bike-sharing systems across the world (Ashqar et al., 2019; Yang et al., 2016).

When evaluating features, dropping `weekend`, `workingday`, and `is_rush_hour` improved performance slightly since these three binary indicators are redundant features similar to `hour` and `day_of_week`. This replicates findings from Yang et al. (2016), where similar features (day-of-week, holiday, and workday) had far less feature importance (0.03, 0.003, 0.006) than hour-of-day (0.14) or recent-history counts (0.63) in their Random Forest model on a larger scale (2,806-station system, 100 million trips).

## References
1. Albuquerque, V., Sales Dias, M., & Bacao, F. (2021). Machine learning approaches to bike-sharing systems: A systematic literature review. ISPRS International Journal of Geo-Information, 10, Article 62. https://doi.org/10.3390/ijgi10020062
2. Ashqar, H. I., Elhenawy, M., & Rakha, H. A. (2019). Modeling bike counts in a bike-sharing system considering the effect of weather conditions. Case Studies on Transport Policy, 7, 261–268. https://doi.org/10.1016/j.cstp.2019.02.011
3. Fricker, C., & Gast, N. (2016). Incentives and redistribution in homogeneous bike-sharing systems with stations of finite capacity. EURO Journal on Transportation and Logistics, 5(3), 261–291. https://doi.org/10.1007/s13676-014-0053-5
4. Yang, Z., Hu, J., Shu, Y., Cheng, P., Chen, J., & Moscibroda, T. (2016). Mobility modeling and prediction in bike-sharing systems. In Proceedings of the 14th Annual International Conference on Mobile Systems, Applications, and Services (pp. 165–178). Association for Computing Machinery. https://doi.org/10.1145/2906388.2906408

