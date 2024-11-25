import gzip
import json
import pickle

from decimal import Decimal
from fnmatch import fnmatch
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
import yaml

from dsp_interview_transcripts import logger


s3 = boto3.resource("s3")


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super(CustomJsonEncoder, self).default(obj)


def load_s3_data(bucket_name: str, file_name: str):
    """Load a file from an S3 location.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_name (str): Path to the file in the S3 bucket.

    Returns:
        Loaded data.
    """
    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return [json.loads(line) for line in file]
    if fnmatch(file_name, "*.yml") or fnmatch(file_name, "*.yaml"):
        file = obj.get()["Body"].read().decode()
        return yaml.safe_load(file)
    elif fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.json.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return json.load(file)
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_name, "*.csv"):
        return pd.read_csv("s3://" + bucket_name + "/" + file_name)
    elif fnmatch(file_name, "*.parquet"):
        return pd.read_parquet("s3://" + bucket_name + "/" + file_name)
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        with obj.get()["Body"] as f:
            return pickle.load(f)
    elif fnmatch(file_name, "*.npy"):
        with obj.get()["Body"] as f:
            return np.load(f)
    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.parquet", "*.jsonl.gz", "*.jsonl", or "*.json"'
        )


def save_to_s3(bucket_name: str, output_var, output_file_dir: str):
    """Saves a file to S3.

    Args:
        bucket_name (str): Bucket name.
        output_var (_type_): Output variable to save.
        output_file_dir (str): Path to save the file to.
    """

    obj = s3.Object(bucket_name, output_file_dir)

    if fnmatch(output_file_dir, "*.csv"):
        output_var.to_csv("s3://" + bucket_name + "/" + output_file_dir, index=False)
    elif fnmatch(output_file_dir, "*.parquet"):
        output_var.to_parquet("s3://" + bucket_name + "/" + output_file_dir, index=False)
    elif fnmatch(output_file_dir, "*.pkl") or fnmatch(output_file_dir, "*.pickle"):
        obj.put(Body=pickle.dumps(output_var))
    elif fnmatch(output_file_dir, "*.gz"):
        obj.put(Body=gzip.compress(json.dumps(output_var).encode()))
    elif fnmatch(output_file_dir, "*.txt"):
        obj.put(Body=output_var)
    elif fnmatch(output_file_dir, "*.npy"):
        buffer = BytesIO()
        np.save(buffer, output_var)
        buffer.seek(0)  # Move the cursor to the beginning of the buffer
        obj.put(Body=buffer.getvalue())
    else:
        obj.put(Body=json.dumps(output_var, cls=CustomJsonEncoder))

    logger.info(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


def upload_file_to_s3(bucket_name: str, local_file: str, output_file_dir: str):
    s3_client = boto3.client("s3")

    s3_client.upload_file(local_file, bucket_name, output_file_dir)
