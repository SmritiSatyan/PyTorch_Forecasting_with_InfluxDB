import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Fill in the below attributes after creating an account on InfluxDB Cloud
token = "YOUR_TOKEN_HERE"
org = "YOUR_EMAIL_HERE"
bucket = "YOUR_BUCKET_HERE"
# URL depends on the region selected during sign up, an example is <url= "https://europe-west1-1.gcp.cloud2.influxdata.com">
url = "YOUR_URL_HERE"
client = InfluxDBClient(url=url, token=token, org=org, debug=True)
"""
Flux query execution/other source code
"""
