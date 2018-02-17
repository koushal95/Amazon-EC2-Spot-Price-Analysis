import boto3
import datetime 
import pandas as pd
from sklearn import preprocessing
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import dateutil
import itertools

def handler(event, context):
    start_time = event['start_time']
    end_time = event ['end_time']
    region = event['region']
    product_description = event['product_description']
    client = boto3.client('ec2', region_name=region)
    response = client.describe_spot_price_history(
        InstanceTypes=event['instances_list'],
        ProductDescriptions=product_description,
        StartTime=start_time,
        EndTime = end_time,
        MaxResults=10000
    )
    return response['SpotPriceHistory']
def wrapper(instanceList, ProductDescriptionList, region, numberOfDays = 7):
    m4_list = []
    for i in range(1,90):
        output = (handler({
        'instances_list': instanceList,
        'start_time': datetime.datetime.now() - datetime.timedelta(i),
        'end_time': datetime.datetime.now() - datetime.timedelta(i-1),
        'product_description': ProductDescriptionList,
        'region': region
    }, ''))
        for j in range(0,len(output)):
            m4_list.append(output[j])

    df = pd.DataFrame(m4_list)
    df_uni = df.drop_duplicates()
    df_uni.reset_index(drop=True,inplace=True)
    availzone = df_uni.AvailabilityZone.unique() # this is for building the test set while deploying
    
    le = preprocessing.LabelEncoder()
    encode_ProductDescription = le.fit_transform(df_uni.ProductDescription)
    encode_InstanceType = le.fit_transform(df_uni.InstanceType)
    encode_AvailabilityZone = le.fit_transform(df_uni.AvailabilityZone)

    df_uni = df_uni.assign(year = df_uni.Timestamp.dt.year)
    df_uni = df_uni.assign(month = df_uni.Timestamp.dt.month)
    df_uni = df_uni.assign(day = df_uni.Timestamp.dt.day)
    df_uni = df_uni.assign(day_of_week = df_uni.Timestamp.dt.weekday)
    df_uni = df_uni.assign(hour = df_uni.Timestamp.dt.hour)
    df_uni = df_uni.assign(minute = df_uni.Timestamp.dt.minute)
    df_uni = df_uni.assign(second = df_uni.Timestamp.dt.second)
    df_uni = df_uni.assign(ProdDescEnc = encode_ProductDescription)
    df_uni = df_uni.assign(AvailZoneEnc = encode_AvailabilityZone)
    df_uni = df_uni.assign(InstanceTypeEnc = encode_InstanceType)

    ## Prepare data for model
    df_uni.sort_values(['Timestamp'], ascending=[True], inplace=True)
    
    y = df_uni.loc[:,'SpotPrice']
    X = df_uni.loc[:,['AvailZoneEnc','InstanceTypeEnc','ProdDescEnc','year','month','day','day_of_week','hour','minute','second']]
    
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)
    
    rng = pd.date_range(start = datetime.datetime.now(), end = datetime.datetime.now() + datetime.timedelta(numberOfDays), freq = 'H', tz=dateutil.tz.tzutc(), normalize = 'False')

    # convert the index into column from feature engineering
    real_test = pd.Series(rng)
    real_test_frame = real_test.to_frame()
    real_test_frame.columns = ['Timestamp']

    # feature engineering
    real_test_frame['year'] = real_test_frame.Timestamp.dt.year
    real_test_frame['month'] = real_test_frame.Timestamp.dt.month
    real_test_frame['day'] = real_test_frame.Timestamp.dt.day
    real_test_frame['day_of_week'] = real_test_frame.Timestamp.dt.weekday
    real_test_frame['hour'] = real_test_frame.Timestamp.dt.hour
    real_test_frame['minute'] = real_test_frame.Timestamp.dt.minute
    real_test_frame['second'] = real_test_frame.Timestamp.dt.second
    
    final_deploy = pd.DataFrame()
    [r, c] = real_test_frame.shape

    rowlists = [instanceList, ProductDescriptionList, availzone]
    i = 0
    for combination in (list(itertools.product(*rowlists))):
        final_deploy = final_deploy.append(real_test_frame, ignore_index=True)
        final_deploy.loc[i:i+r-1, 'InstanceType'] = combination[0]
        final_deploy.loc[i:i+r-1, 'ProductDescription'] = combination[1]
        final_deploy.loc[i:i+r-1, 'AvailabilityZone'] = combination[2]
        i = i + r
    
    encod_ProductDescription = le.fit_transform(final_deploy.ProductDescription)
    encod_InstanceType = le.fit_transform(final_deploy.InstanceType)
    encod_AvailabilityZone = le.fit_transform(final_deploy.AvailabilityZone)
    final_deploy = final_deploy.assign(ProdDescEnc = encod_ProductDescription)
    final_deploy = final_deploy.assign(AvailZoneEnc = encod_AvailabilityZone)
    final_deploy = final_deploy.assign(InstanceTypeEnc = encod_InstanceType)
    
    test_deploy  = final_deploy.loc[:,['AvailZoneEnc','InstanceTypeEnc','ProdDescEnc','year','month','day','day_of_week','hour','minute','second']]
    
    ## let the predictions begin!!
    future = regr.predict(test_deploy)
    future_series = pd.Series(future)
    
    pretty_predictions = pd.DataFrame()
    pretty_predictions = pretty_predictions.assign(Timestamp = final_deploy.Timestamp)
    pretty_predictions = pretty_predictions.assign(AvailabilityZone = final_deploy.AvailabilityZone)
    pretty_predictions = pretty_predictions.assign(InstanceType = final_deploy.InstanceType)
    pretty_predictions = pretty_predictions.assign(ProductDescription = final_deploy.ProductDescription)
    pretty_predictions = pretty_predictions.assign(Predicted_SpotPrice = future_series)
    
    return pretty_predictions

## enter the instance types you need below as list!
## give only one product description for one run
df = wrapper(['m4.large', 'm4.xlarge'],['Linux/UNIX (Amazon VPC)'], 'us-west-2')
df
