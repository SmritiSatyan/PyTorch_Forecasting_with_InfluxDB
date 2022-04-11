import pandas as pd
    
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings
    
from pytorch_forecasting.data.examples import get_stallion_data
import numpy as np
stallion_df = get_stallion_data()
token = "generated token"
org = "email"
url = "https://us-east-1-1.aws.cloud2.influxdata.com"
    
# pushing contents of dataframe to InfluxDB cloud
stallion_data = get_stallion_data()
stallion_data = stallion_data.rename(columns={"date": "time"})
stallion_data = stallion_data.set_index('time')
        
"""
Enable logging for DataFrame serializer
"""
loggerSerializer = logging.getLogger('influxdb_client.client.write.dataframe_serializer')
loggerSerializer.setLevel(level=logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
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
        write_api.write(bucket='myBucket', record=stallion_data,
                        data_frame_tag_columns=['agency', 'sku'],
                        data_frame_measurement_name="stallion_data")
        print()
        print("Wait to finishing ingesting DataFrame...")
        print()

print()
print(f'Import finished in: {datetime.now() - startTime}')
print()

results = []

with InfluxDBClient(url="https://us-east-1-1.aws.cloud2.influxdata.com", token=token, org=org) as client:
    query = """option v = {timeRangeStart: -10h, timeRangeStop: now()}
                        from(bucket: "myBucket")
                        |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
                        |> group(columns: ["stallion_data"])"""
    tables = client.query_api().query(query, org=org)
    for table in tables:
        for record in table.records:
            results.append((record.get_field(), 
                                record.get_value(), 
                                record.get_measurement(), 
                                record.get_time()))
                        
# convert the list to a dataframe                        
influx_df = pd.DataFrame(results, columns=['_field', '_value', '_measurement', column-names])
