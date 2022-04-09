import pandas as pd
    
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings
    
from pytorch_forecasting.data.examples import get_stallion_data
import numpy as np
stallion_df = get_stallion_data()
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
results = [] # initialize an empty list to append records

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
                                record.get_time()))
                        
# convert the list to a dataframe                        
influx_df = pd.DataFrame(results, columns=['_field', '_value', '_measurement', column-names])

Note: You can use ``record.values.get("column-name")`` if you wish to fetch other columns from InfluxDB.
