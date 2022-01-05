# Airbnb Time Series Forecasting Summary

The goal of this project is to predict the number of future Airbnb registrations at a given point in time.

## Data

The data is a single csv file such that each observation is a unique Airbnb housing registration for a certain point in time indicated by its "listing id" and the specific date that the listing id was booked. For this project, I limited the scope of the Airbnb listings to the city of Los Angeles from the years 2009 - 2019

## Tools

Data table manipulation was performed with the standard Pandas and Numpy libraries. Graphing the time series data was done in Matplotlib. The Statsmodels library was utilized for ARIMA modeling.

## Model

The original time series data is non-stationary, with an obvious upward trend and seasonal characteristics apparent near the last quarter of every year. Hence, a series of transformations, rolling averages, and decompositions were necessary to make the data stationary. ACF and PACF plots were used to help determine the number of auto-regressive lag terms (p) and moving average terms (q). Differencing degree of 1 showed improvement to stationarity. The model was fitted based on the AIC loss.

## Conclusion

In the end, a log transformation of the original data with the parameters ARIMA(1, 1, 1) showed an adequate. Different choices can be made for data transformations,  ARIMA parameters, and an alternative choice of loss (BIC) in attempts to improve the model. This can be done with a grid search given the time, and will be left for future work.

### Useful Resources

- https://people.duke.edu/~rnau/411arim.htm
- https://www.datascience.com/blog/introduction-to-forecasting-with-arima-in-r-learn-data-science-tutorials
- https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method
- https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/
- https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/