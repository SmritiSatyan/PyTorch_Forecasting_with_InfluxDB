from influxdb_client import InfluxDBClient

# Fill in the below attributes after creating an account on InfluxDB Cloud
token = "YOUR_TOKEN_HERE"
org = "YOUR_EMAIL_HERE"
bucket = "YOUR_BUCKET_HERE"
# The url depends on the region selected during sign up, an example is <url= "https://europe-west1-1.gcp.cloud2.influxdata.com">
url = "YOUR_URL_HERE"
client = InfluxDBClient(url=url, token=token, org=org, debug=True)
"""
Flux query execution/other source code
"""
results=[]   
query = """option v = {timeRangeStart: -30d, timeRangeStop: now()}
                        from(bucket: "myBucket")
                        |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
                        |> filter(fn: (r) => r["_measurement"] == "stallion_data")"""
tables = client.query_api().query(query, org=org)
for table in tables:
    for record in table.records:
        results.append(
               [   record.get_field(),
                   record.get_value(),
                   record.get_measurement(),
                   record.get_time(),
                   record.values.get("agency"),
                   record.values.get("sku"),
               ]
           ) 
