import logging
from datetime import datetime

import pandas as pd
from influxdb_client import InfluxDBClient
from pytorch_forecasting.data.examples import get_stallion_data

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

# data preprocessing
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
