import boto3
import io
import pandas as pd


def load_data_from_s3(bucket_name: str, file_path: str) -> pd.DataFrame:
    client = boto3.client('s3')
    csv_obj = client.get_object(Bucket=bucket_name, Key=file_path)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(io.StringIO(csv_string), index_col=0)
    return data