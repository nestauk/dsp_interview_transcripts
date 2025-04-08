import base64

from io import StringIO

import pandas as pd


def read_data(contents):
    """If you have used an upload box to pick a csv file, you will
    need this logic to read in the csv as a pandas dataframe.
    """
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(StringIO(decoded.decode("utf-8")))
    return df
