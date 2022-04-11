# Import required libraries
import os
import warnings

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings

warnings.filterwarnings("ignore")  # avoid printing absolute paths

import copy
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_forecasting import (Baseline, TemporalFusionTransformer,
                                 TimeSeriesDataSet)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import \
    optimize_hyperparameters
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.data.examples import get_stallion_data

# Replace 'token', 'org', and 'bucket' with your account specific values
token = "generated token"
org = "email"
bucket = "bucket"
# The url depends on the location selected during sign up
url = "https://us-east-1-1.aws.cloud2.influxdata.com"    
    
# Fetch stallion data and store in a dataframe
stallion_data = get_stallion_data()
token = "generated token"
org = "email"
url = "https://us-east-1-1.aws.cloud2.influxdata.com"
    
# using line protocol to write data to InfluxDB cloud
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)
    
# line protocol should have the syntax
'''<measurement>[,<tag_key>=<tag_value>[,<tag_key>=<tag_value>]] <field_key>=<field_value>[,<field_key>=<field_value>] [<timestamp>]''''
    
# add the columns required for the prediction
data = "beverage_sales,volume=59.315,date=2013-01-01,timeseries=166 actual_price=52.212"
write_api.write(bucket, org, data)
    

# Querying ingested data
results = [] # initialize an empty list to append the records

with InfluxDBClient(url="https://us-east-1-1.aws.cloud2.influxdata.com", token=token, org=org) as client:
    query = """option v = {timeRangeStart: -1h, timeRangeStop: now()}
                        from(bucket: "my-bucket-1")
                        |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
                        |> group(columns: ["beverage_sales"])
                        |> filter(fn: (r) => r["_field"] == "actual_price")"""
    tables = client.query_api().query(query, org=org)
    for table in tables:
        for record in table.records:
            results.append((record.get_field(), 
                                record.get_value(), 
                                record.get_measurement(), 
                                record.get_time())) # Other columns from the data can be fetched using df_name.get("column-name")
                        
# convert the list to a dataframe                        
stallion_df = pd.DataFrame(results, columns=['_field', '_value', '_measurement', other-columns])

# Add time index
stallion_df["time_idx"] = stallion_df["date"].dt.year * 12 + stallion_df["date"].dt.month
stallion_df["time_idx"] -= stallion_df["time_idx"].min()

# Add additional features that contribute to forecasting
stallion_df["month"] = stallion_df.date.dt.month.astype(str).astype("category")  # categories need to be strings
stallion_df["log_volume"] = np.log(stallion_df.volume + 1e-8)
stallion_df["avg_volume_by_sku"] = stallion_df.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
stallion_df["avg_volume_by_agency"] = stallion_df.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

# Encode special days as a single variable
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
stallion_df[special_days] = stallion_df[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
stallion_df.sample(10, random_state=521)

# stallion_df.describe()

max_prediction_length = 6 # 6 months of record is chosen as validation data
max_encoder_length = 24
training_cutoff = stallion_df["time_idx"].max() - max_prediction_length

# Convert the dataframe to a PyTorch forecasting TimeSeriesDataSet format
# Define the GroupNormalizer to normalize a given entry by groups.
# For every group, a scaler is fit and applied. It can be used as target normalizer as well as to normalize other variable.
training = TimeSeriesDataSet(
    stallion_df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length // 2,  # Encoder length should be long since it is in the validation set
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={"special_days": special_days},  # a group of categorical variables is treated as a single variable
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ], 
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


# Create validation set to predict the last max_prediction_length points in time for every series
validation = TimeSeriesDataSet.from_dataset(training, stallion_df, predict=True, stop_randomization=True)

# Create dataloaders for the model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


# Calculate mean absolute error. 
# Evaluate Baseline model that predicts the next 6 months sales by repeating last available values from historic data
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()


# Configure the network and trainer, to identify optimal learning rate hyperparameter 
pl.seed_everything(42)
# Trainer runs training, validation and test dataloaders
trainer = pl.Trainer(
    gpus=0,
    # clipping gradients is a hyperparameter that helps prevent divergence of the gradient in recurrent neural networks
    gradient_clip_val=0.1,
)

# Initialize TemporalFusionTransformer to forecast time series
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    # number of attention heads. Set to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement is seen in the validation loss after 'x' epochs
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# Determine the optimal learning rate
res = trainer.tuner.lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

# The optimal learning rate is slightly lower. We pick the value as 0.03 to avoid confusing PyTorch Lightning
# due to the noise at lower loearning rates
print(f"Suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()


# Configure the network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=30,
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=30,  # comment out for training and running validation every 30 batches
    # fast_dev_run=True,  # comment out to check that the network dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

# using the `TemporalFusionTransformer` to fit the data
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder, otherwise 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# Fit data to the network
# For large datasets, training may take hours
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Evaluate performance by loading best model using validation loss
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calculate mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()

# Once training is complete, use predict()
# raw predictions is a dictionary, from which meta data, such as quantiles can be extracted
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
for idx in range(4):  # plot 4 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);
   
# Select last 24 months from stallion_df (max_encoder_length has been initialized to 24)
# The presence of covariates requires defining known covariates beforehand
# Predict on new data
encoder_data = stallion_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

# Select last known stallion_df point and create decoder stallion_df from it. This can be done by repeating it and 
# incrementing the month
last_data = stallion_df[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat(
    [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
    ignore_index=True,
)

# Add time index consistent with "stallion_df"
decoder_data["time_idx"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

# Adjust additional time features
decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category")  # categories have be strings

# Combine encoder and decoder stallion_df, predict on generated data using predict()
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

# Plot 4 examples
for idx in range(4):  # plot 4 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False);
