import os
from polygon import RESTClient
import pandas as pd
import time
import logging
import pickle
from common import get_data_path

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
pd.options.display.width = 0


class RESTfulProcessor:
    key = os.environ['polygon_key']
    data_folder = get_data_path()

    def __init__(self):
        self.ticker = []
        self.next_url = None
        self.counter = 0

    def set_next_url(self, next_url):
        logging.info('next url is {}'.format(next_url))
        self.next_url = next_url

    def get_ticker(self):
        if file
        with RESTClient(self.key) as client:
            while self.counter == 0 or len(ls_ticker) == 100:
                res = client.reference_tickers_v3(next_url=self.next_url)
                ls_ticker = list(pd.DataFrame(res.results).loc[:, 'ticker'])
                logging.info('loaded tickers from {s} to {e}'.format(s=ls_ticker[0], e=ls_ticker[-1]))
                self.ticker.extend(ls_ticker)

                if hasattr(res, 'next_url'):
                    self.set_next_url(res.next_url)
                    self.counter += 1
                    time.sleep(0.1)
                else:
                    break
        return

    def store_ticker(self):
        logging.info('Storing ticker data into pickle')
        with open(self.data_folder + 'ticker.pickle', 'wb') as f:
            pickle.dump(self.ticker, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info('Storing is done')

    def load_ticker(self):
        with open(self.data_folder + 'ticker.pickle') as f:
            ticker = pickle.load(f)

        return ticker

    def run(self):
        self.get_ticker()
        self.store_ticker()

    def get_ticker_detail(self):
        for t in self.ticker:


if __name__ == '__main__':
    resp = RESTfulProcessor()
    resp.run()
