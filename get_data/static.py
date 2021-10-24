import os
from polygon import RESTClient
import pandas as pd
import time
import logging
import pickle
import matplotlib.pyplot as plt
import requests
import datetime
from common import get_data_path

logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)
pd.options.display.width = 0


class RESTfulProcessor:
    key = os.environ['polygon_key']
    data_folder = get_data_path()
    holidays = [
        '2020-07-03',
        '2020-09-07',
        '2020-11-26',
        '2020-12-25',
        '2021-01-01',
        '2021-01-18',
        '2021-02-15',
        '2021-04-02',
        '2021-05-31',
        '2021-07-05',
        '2021-09-06'
    ]

    def __init__(self):
        self.ticker = []
        self.next_url = None
        self.ticker_file_name = self.data_folder + 'ticker.pkl'

        self.ticker_detail = []
        self.ticker_detail_file_name = self.data_folder + 'ticker_detail.pkl'

        self.ticker_price = []
        self.ticker_price_file_name = self.data_folder + 'ticker_price.pkl'
        self.ticker_price_error_counter = {}

    def set_next_url(self, next_url):
        logging.info('next url is {}'.format(next_url))
        self.next_url = next_url

    def get_ticker(self):
        if os.path.isfile(self.ticker_file_name):
            self.ticker = self.load_data(file_name=self.ticker_file_name)
        else:
            with RESTClient(self.key) as client:
                counter = 0
                while counter == 0 or len(ls_ticker) == 100:
                    res = client.reference_tickers_v3(next_url=self.next_url)
                    ls_ticker = list(pd.DataFrame(res.results).loc[:, 'ticker'])
                    logging.info('loaded tickers from {s} to {e}'.format(s=ls_ticker[0], e=ls_ticker[-1]))
                    self.ticker.extend(ls_ticker)

                    if hasattr(res, 'next_url'):
                        self.set_next_url(res.next_url)
                        counter += 1
                        time.sleep(0.1)
                    else:
                        break
            self.store_data(data=self.ticker, file_name=self.ticker_file_name)
        return

    @staticmethod
    def store_data(data, file_name):
        logging.info('Storing {} data into pickle'.format(file_name))
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info('Storing is done')

    @staticmethod
    def load_data(file_name: str):
        logging.info('Found stored {} data, start loading'.format(file_name))
        with open(file_name, 'rb') as f:
            ticker = pickle.load(f)
        logging.info('Loading is done')

        return ticker

    def get_ticker_detail(self):
        if os.path.isfile(self.ticker_detail_file_name):
            self.ticker_detail = self.load_data(file_name=self.ticker_detail_file_name)
        else:
            error_counter = 0
            counter = 0
            ticker_detail = []
            with RESTClient(self.key) as client:
                for t in self.ticker:
                    if counter % 100 == 0:
                        logging.info('counter = {c}, start loading {t}'.format(c=counter, t=t))
                    try:
                        res = client.reference_ticker_details_vx(symbol=t)
                        ticker_detail.append(res.results)
                    except Exception as e:
                        print(e)
                        error_counter += 1
                        if error_counter > 200:
                            break
                    counter += 1
                    time.sleep(0.05)

            self.ticker_detail = pd.DataFrame(ticker_detail)
            self.store_data(data=self.ticker_detail, file_name=self.ticker_detail_file_name)

        return

    def ticker_filter(self):
        df_ticker = self.ticker_detail.copy()
        df_ticker = df_ticker[~df_ticker['sic_code'].isna()]

        df_ticker = df_ticker[[
            'ticker_root', 'ticker', 'name', 'market', 'locale', 'primary_exchange', 'type',
            'currency_name', 'cik', 'outstanding_shares', 'market_cap', 'address', 'sic_code', 'sic_description',
            'ticker_suffix', 'base_currency_symbol'
        ]]
        # df_ticker = df_ticker[df_ticker['market_cap'] > 1e9]

        return df_ticker

    def get_price(self):
        if os.path.isfile(self.ticker_price_file_name):
            self.ticker_price = self.load_data(file_name=self.ticker_price_file_name)
        else:
            df = self.ticker_filter()
            tickers = df['ticker'].values
            ticker_price = []
            biz_dates = pd.bdate_range('2020-06-01', '2021-10-15')
            biz_dates = [x.date().strftime('%Y-%m-%d') for x in biz_dates]
            biz_dates = [x for x in biz_dates if x not in self.holidays]

            with RESTClient(self.key) as client:
                for t in tickers:
                    logging.info('start loading {t}'.format(t=t))
                    for bd in biz_dates:
                        try:
                            res = client.stocks_equities_daily_open_close(symbol=t, date=bd)
                            res_dict = {
                                'ticker': t,
                                'date': bd,
                                'after_hours': res.after_hours,
                                'high': res.high,
                                'low': res.low,
                                'open': res.open,
                                'pre_market': res.pre_market,
                                'volume': res.volume
                            }
                            ticker_price.append(res_dict)
                        except requests.exceptions.HTTPError as e:
                            self.ticker_price_error_counter[t] = self.ticker_price_error_counter.get(t, 0) + 1
                            print(e)
                            if self.ticker_price_error_counter[t] > 10:
                                print('too many errors, jump to next ticker')
                                break
                        time.sleep(0.01)

            self.ticker_price = pd.DataFrame(ticker_price)
            self.store_data(data=self.ticker_price, file_name=self.ticker_price_file_name)
        return

    def run(self):
        self.get_ticker()
        self.get_ticker_detail()
        self.get_price()

        return


if __name__ == '__main__':
    resp = RESTfulProcessor()
    df = resp.run()
