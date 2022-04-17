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
