import pandas as pd
import pickle
from common import get_data_path


def process_data(ticker_price_file_name, ticker_detail_file_name='ticker_detail.pkl'):
    with open(get_data_path() + ticker_price_file_name, 'rb') as f:
        df_price = pickle.load(f)

    with open(get_data_path() + ticker_detail_file_name, 'rb') as f:
        df_ticker = pickle.load(f)

    df = pd.merge(df_price, df_ticker, on='ticker')


    return df


if __name__ == '__main__':
    df = process_data(ticker_price_file_name='ticker_price_237773.pkl')
