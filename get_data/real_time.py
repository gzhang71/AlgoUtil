import time
import os
from polygon import WebSocketClient, STOCKS_CLUSTER


def error_handler(ws, error):
    print("this is my custom error handler", error)


def close_handler(ws):
    print("this is my custom close handler")


def process_message(message):
    print("this is my custom message processing", message)


class SocketProcessor:
    key = os.environ['polygon_key']

    def __init__(self):
        self.client = WebSocketClient(
            cluster=STOCKS_CLUSTER,
            auth_key=self.key,
            process_message=process_message,
            on_close=close_handler,
            on_error=error_handler
        )

    def run(self, ticker: str = 'T.MSFT'):
        self.client.run_async()
        self.client.subscribe()
        time.sleep(1)
        self.client.close_connection()


if __name__ == '__main__':
    ticker = 'T.MSFT'
    sp = SocketProcessor()
    sp.run()
