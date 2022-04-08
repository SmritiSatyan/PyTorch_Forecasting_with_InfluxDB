# PyTorch_Forecasting_with_InfluxDB

# Introduction

[InfluxDB](https://www.influxdata.com/) is a database that specializes in the collection, storage, and processing of time series data.
In this Python project, you will see:
1. How to establish connection between InfluxDB and InfluxDB client,
2. How to push dataframe to InfluxDB,
3. How to fetch dataframe from InfluxDB to Python client, and
4. How to use this data to perform forecasting using PyTorch.


First, we install ``influxdb`` and ``influxdb-client`` that will help establish connection between InfluxDB Cloud and Python client. We use [PyTorch](https://pytorch-forecasting.readthedocs.io/en/stable/) forecasting to make predctions on this data. For this, we install ``torch`` and ``pytorch-forecasting``. We use the Stallion time series data present in PyTorch. We predict the beverage sales (in dollars) for 6 months based on 21,000 rows of historical data.  

Stallion data contains the below datasets:
* pricesalespromotion.csv: Holds the price, sales & promotion in dollars.
* historicalvolume.csv: contains sales data.
* weather.csv: the average maximum temperature at Agency monthly/ 
* Industrysodasales.csv: Holds industry level soda sales 
* eventcalendar.csv: Holds event details (sports, carnivals, etc.)
* industry_volume.csv: industry actual beer volume
* demographics.csv: demographic details
