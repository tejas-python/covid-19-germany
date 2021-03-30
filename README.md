# COVID - 19 DATA ANALYSIS GERMANY
The following repository contains the Note book dealing with Covid 19 data analysis for the country Germany.

## Objectives

- The main goal of this note book is the analysise and predict the number of new cases for the country germnay in future days.  
-   Obtain data insights using pandas.
-  Cleaning the data with appropriate techniques.
- performing epxloratory data analysis (EDA) on the data to get better insights.
-  Modeling the data with various model with appropriate feature selection techniques.

# ABOUT THE DATA
 ### The data is obtaind for the website ourworldindata.org 
  </ul> 
    <li>data source: <a href="https://ourworldindata.org/coronavirus-source-data" target="_blank">https://ourworldindata.org/coronavirus-source-data</a></li>
    <li>data type available : .xslx .csv .json (daily updated)</li>
    <li>more information on the data: <a href="https://github.com/owid/covid-19-data/tree/master/public/data/" target="_blank">https://github.com/owid/covid-19-data/tree/master/public/data/</a></li>
    
</ul>

# Importing the libraries required such  as 
- ##  Pandas for loding the data ans performing basic operations .
- ##  Matplotlib ans seaborn for vizalization and EDA. 

``` python
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy  as np 
import datetime as dt
```
#  Variation of features with repect to time (i.e from jan20 to march21)

## Total number of cases

![total cases](./images/totalcases.png)

## Total number of deaths

![total deaths](./images/totaldeath.png)

## New cases variation with each day

![new cases ](./images/download.png)

## New deaths variation withrespect to each day

![new death ](./images/newdeaths.png)


## Variation of new cases with repect to each month 
![new cases ](./images/newcasesmonths.png)

## Variation of new deaths with repect to each month 

![new deaths ](./images/new_deathsmonth.png)


## Variation of new cases with date with reproduction rate as hue 
![new cases ](./images/huepositverate.png)

## Variation of new cases with date with reproduction rate as hue 
![new cases ](./images/huereproductionrate.png)

## Variation of new cases with date with strigency index as hue 

![new cases ](./images/huetrigencyndex.png)

## Line plot comparing the total cases new cases total deaths and new deaths 


![line plot](./images/compare3.png)
### With total cases
![line plot](./images/compare4.png)

# Modeling the data for the prediction of new cases 
#
# Prophet
## Performing time series forcasting using fb peophet to predict the new cases.
## Prophet
Prophet is open source software released by Facebook’s Core Data Science team. It is available for download on CRAN and PyPI.

We use Prophet, a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. 

## Why Prophet?

* **Accurate and fast:**  Prophet is used in many applications across Facebook for producing reliable forecasts for planning and goal setting. Facebook finds it to perform better than any other approach in the majority of cases. It fit models in [Stan](https://mc-stan.org/) so that you get forecasts in just a few seconds.

* **Fully automatic:** Get a reasonable forecast on messy data with no manual effort. Prophet is robust to outliers, missing data, and dramatic changes in your time series.

* **Tunable forecasts:** The Prophet procedure includes many possibilities for users to tweak and adjust forecasts. You can use human-interpretable parameters to improve your forecast by adding your domain knowledge



## References 
- https://facebook.github.io/prophet/
- https://facebook.github.io/prophet/docs/
- https://github.com/facebook/prophet
- https://facebook.github.io/prophet/docs/quick_start.html

```python

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
```
## Prediction for the next 70 days (i.e after march)
![new cases ](./images/phrophet.png)
![new cases ](./images/prophet1.png)

## Model evaluation by cross validation and rmse plot 
![new cases ](./images/prophet2.png)

## we can see that the accuracy of the model with the help of the rmse plot. we can see the point lie far from the line which mention the model dose not show good acuracy. 

#



# MULTIPLE LINEAR REGRESSION MODEL 
### What Is Multiple Linear Regression (MLR)?
Multiple linear regression (MLR), also known simply as multiple regression, is a statistical technique that uses several explanatory variables to predict the outcome of a response variable. The goal of multiple linear regression (MLR) is to model the linear relationship between the explanatory (independent) variables and response (dependent) variable.

In essence, multiple regression is the extension of ordinary least-squares (OLS) regression because it involves more than one explanatory variable.

Formula and Calculation of Multiple Linear Regression

y_i = beta_0 + beta _1* x_i1 + beta _2 * x_i2 + ... + beta _p*x_ip + epsilon

where, for  i = n  observations:

y_i=tdependent variable 

x_i=explanatory variables 

beta_0=y-intercept

beta_p=slope coefficients for each explanatory variable

epsilon=the model's error term (also known as the residuals)

### What Multiple Linear Regression Can Tell You
Simple linear regression is a function that allows an analyst or statistician to make predictions about one variable based on the information that is known about another variable. Linear regression can only be used when one has two continuous variables—an independent variable and a dependent variable. The independent variable is the parameter that is used to calculate the dependent variable or outcome. A multiple regression model extends to several explanatory variables.

## Libraries used 
```python 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # To standardise the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4, random_state=1) #splitting the data
from sklearn import linear_model
regr = linear_model.LinearRegression() # intialising the model
from sklearn.metrics import r2_score # for evaluation  
```
## Traing the data as hole by spliting them to testing and traning sets and performing the multiple linear regression .
 ###  We obtain the following accuracy 

* Test data score 0.92
* Train data score 0.93
* Mean absolute error: 0.05
* Residual sum of squares (MSE): 0.00
* R2-score: 0.91


## Splitting the data into test and train data , splitting the data from jan to december of 2020 for traning and jan to march 2021 for testing.
```python
df_grouped1 = df_germany[   ( df_germany['month'] != 'January2021')& ( df_germany['month'] != 'February2021' )& ( df_germany['month'] != 'March2021')]
df_grouped2 =df_germany[  ( df_germany['month'] == 'January2021') |( df_germany['month'] == 'February2021' )| ( df_germany['month'] == 'March2021')]
```
###  We obtain the following accuracy for the above model 



 * Test data score 0.91
* Regression score for second wave 0.88
* Mean absolute error: 0.04
* Residual sum of squares (MSE): 0.00
* R2-score: 0.92

![image.png](./images/pred.png)