    import influxdb_client
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS

    #  Generate an API token as shown in step 2 above
    # Fill in the below attributes after creating an account on InfluxDB Cloud
    token = " "
    org = " "
    bucket = " "
    # URL depends on the region selected during sign up
    url="https://europe-west1-1.gcp.cloud2.influxdata.com" 
    client = InfluxDBClient(url=url, token=token, org=org,debug=True)
    '''
    flux query execution/other source code
    '''
