# Import required libraries
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from pytorch_forecasting.data.examples import get_stallion_data

warnings.filterwarnings("ignore")  # avoid printing absolute paths

import pytorch_lightning as pl
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

stallion_df = get_stallion_data()

token = "YOUR_TOKEN_HERE"
org = "YOUR_EMAIL_HERE"

# An example would be <url = "https://us-east-1-1.aws.cloud2.influxdata.com">
url = "YOUR_URL_HERE"
bucket = "YOUR_BUCKET_HERE"
# pushing contents of dataframe to InfluxDB cloud
stallion_data = get_stallion_data()

filter_val = [
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
    "2017-01-01T00:00:00.000000000",
]

filtered_df = stallion_data[stallion_data["date"].isin(filter_val)]

# The below steps are performed since free tier can only store data upto 30 days
filtered_df["date"] = pd.to_datetime(filtered_df.date) + pd.offsets.DateOffset(years=5)
filtered_df["date"] = pd.to_datetime(filtered_df.date) + pd.offsets.DateOffset(months=3)
filtered_df["date"] = pd.to_datetime(filtered_df.date) + pd.offsets.DateOffset(days=3)

# Convert the 'time' column to an index
stallion_data = stallion_data.set_index("time")

"""
Enable logging for DataFrame serializer
"""
loggerSerializer = logging.getLogger(
    "influxdb_client.client.write.dataframe_serializer"
)
loggerSerializer.setLevel(level=logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
loggerSerializer.addHandler(handler)

"""
Ingest DataFrame
"""
print()
print("=== Ingesting DataFrame via batching API ===")
print()
startTime = datetime.now()


with InfluxDBClient(url=url, token=token, org=org) as client:

    """
    Use batching API
    """
    with client.write_api() as write_api:
        write_api.write(
            bucket="myBucket",
            record=stallion_data,
            data_frame_tag_columns=["agency", "sku"],
            data_frame_measurement_name="stallion_data",
        )
        print()
        print("Wait to finishing ingesting DataFrame...")
        print()

print()
print(f"Import finished in: {datetime.now() - startTime}")
print()

results = []
with InfluxDBClient(
    url="https://us-east-1-1.aws.cloud2.influxdata.com", token=token, org=org
) as client:
    query = """option v = {timeRangeStart: -30d, timeRangeStop: now()}

    from(bucket: "myBucket")
    |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
    |> filter(fn: (r) => r["_measurement"] == "stallion_data")"""
    tables = client.query_api().query(query, org=org)
    for table in tables:
        for record in table.records:
            # results.append(record)
            results.append(
                [
                    record.get_field(),
                    record.get_value(),
                    record.get_measurement(),
                    record.get_time(),
                    record.values.get("agency"),
                    record.values.get("sku"),
                ]
            )

# convert the list to a dataframe
influx_df = pd.DataFrame(
    results, columns=["_field", "_value", "_measurement", "time", "agency", "sku"]
)
temp_data = influx_df[["_field", "_value"]]

# remove irrelevant columns
influx_df = influx_df.drop(["_measurement", "_field", "_value"], axis=1)

# The columns are converted to rows when the data is pushed to InfluxDB cloud
regrouped_df = temp_data.pivot_table(
    values="_value", index=temp_data.index, columns="_field", aggfunc="first"
)
regrouped_df = regrouped_df.fillna(0)
stallion_df = pd.concat([influx_df, regrouped_df], axis=1)
stallion_df = stallion_df[:8040]
stallion_df = stallion_df.rename({"time": "date"}, axis=1)  # new method

# generate new 'time' values for forecasting purposes only
m = 134
out = (
    pd.MultiIndex.from_product(
        [
            [f"date-{i}" for i in range(1, m + 1)],
            pd.date_range("2018-01-01", "2022-12-01", freq="MS")
            + pd.DateOffset(days=2),
        ]
    )
    .to_frame(name=["val", "date"])
    .reset_index(drop=True)
)

stallion_df["date"] = out["date"].values


# add time index
stallion_df["time_idx"] = (
    stallion_df["date"].dt.year * 12 + stallion_df["date"].dt.month
)
stallion_df["time_idx"] -= stallion_df["time_idx"].min()

# add additional features
stallion_df["month"] = stallion_df.date.dt.month.astype(str).astype(
    "category"
)  # categories have be strings
stallion_df["log_volume"] = np.log(stallion_df.volume + 1e-8)
stallion_df["avg_volume_by_sku"] = stallion_df.groupby(
    ["time_idx", "sku"], observed=True
).volume.transform("mean")
stallion_df["avg_volume_by_agency"] = stallion_df.groupby(
    ["time_idx", "agency"], observed=True
).volume.transform("mean")

# we want to encode special days as one variable and thus need to first reverse one-hot encoding
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
stallion_df[special_days] = (
    stallion_df[special_days]
    .apply(lambda x: x.map({0: "-", 1: x.name}))
    .astype("category")
)
stallion_df.sample(10, random_state=521)

# stallion_df.describe()

max_prediction_length = 6  # 6 months of record is chosen as validation data
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
    min_encoder_length=max_encoder_length
    // 2,  # Encoder length should be long since it is in the validation set
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={
        "special_days": special_days
    },  # a group of categorical variables is treated as a single variable
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
    allow_missing_timesteps=True,
)


# Create validation set to predict the last max_prediction_length points in time for every series
validation = TimeSeriesDataSet.from_dataset(
    training, stallion_df, predict=True, stop_randomization=True
)

# Create dataloaders for the model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)


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
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=20,
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
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)

# Select last 24 months from stallion_df (max_encoder_length has been initialized to 24)
# The presence of covariates requires defining known covariates beforehand
# Predict on new data
encoder_data = stallion_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

# Select last known stallion_df point and create decoder stallion_df from it. This can be done by repeating it and
# incrementing the month
last_data = stallion_df[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat(
    [
        last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i))
        for i in range(1, max_prediction_length + 1)
    ],
    ignore_index=True,
)


# Add time index consistent with "stallion_df"
decoder_data["time_idx"] = (
    decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
)
decoder_data["time_idx"] += (
    encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()
)

# Adjust additional time features
decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype(
    "category"
)  # categories have be strings

# Combine encoder and decoder stallion_df, predict on generated data using predict()
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
new_raw_predictions, new_x = best_tft.predict(
    new_prediction_data, mode="raw", return_x=True
)

# Plot 4 examples
for idx in range(4):  # plot 4 examples
    best_tft.plot_prediction(
        new_x, new_raw_predictions, idx=idx, show_future_observed=False
    )
