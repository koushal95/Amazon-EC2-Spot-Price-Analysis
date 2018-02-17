# Amazon-EC2-Spot-Price-Analysis
Amazon web services provides different pricing models pay-per-use, fixed, and auction-based (spot price). It is seen that the spot price is a minimum of 5 times cheaper than the other pricing models but there is no guarentee that you will be given the instance. It depends on the price you bid. Thus, analysis of historical data for spot price inorder to efficiently (minimal cost) schedule the jobs is important.

This repository presents accessing and analyzing the historical spot pricing is easy using modern data science toolsets and analysis can lead to insight that can save you significant money.

### Upcoming
Prediction on the Spot prices using Machine Learning!

### Update
Added the code for prediction of spot prices using random forests. 
Cleaned the code and made a modular approach. To predict one has to just call the wrapper function with suitable arguments.
The first argument is list of instance types you are interested.
The second argument is the product description and the third is the region.
The fourth and final argument is optional argument, number of days for which the price is to be predicted. By default it provides for the next seven days(1 week).

### Repo walk-through
This repository contains a jupyter notebook that contains code and plots for,

* Data fethcing using boto3

* Exploratory data analysis

* Data insights

And the best part! The python file "spot_price_predict.py" contains the code for,

* Data import

* Feature extraction and engineering

* Data preparation

* Predictive Model built using random forests.

If there is anything you want to talk about please reach out. If you find any issues feel free to update them on the issues of this repository.
