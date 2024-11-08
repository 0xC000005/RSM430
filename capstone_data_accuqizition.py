# This is a python example algorithm using DMA REST API for the RIT ALGO1 Case

import signal
import requests
from time import sleep
import base64

# this class definition allows us to print error messages and stop the program when needed
class ApiException(Exception):
    pass

# this signal handler allows for a graceful shutdown when CTRL+C is pressed
def signal_handler(signum, frame):
    global shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True

# use your RIT API key here
API_KEY = {'X-API-key': 'GKWRRO4P'}
# Make sure the RIT client uses the same 9999 port
API_ENDPOINT = 'http://localhost:9999/v1'


shutdown = False

# this helper method handles rate-limiting to pause for the next cycle.
def handle_rate_limit(response):
    if response.status_code == 429:
        wait_time = float(response.headers.get('Retry-After', response.json().get('wait', 1)))
        print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying.")
        sleep(wait_time)
        return True
    return False

# this helper method handles authorization failure.
def handle_auth_failure(response):
    if response.status_code == 401:
        print("Authentication failed. Please check your username and password.")
        global shutdown
        shutdown = True
        return True
    return False

# this helper method compiles possible API responses and handlers. 
def api_request(session, method, endpoint, params=None):
    while True:
        url = f"{API_ENDPOINT}/{endpoint}"
        if method == 'GET':
            resp = session.get(url, params=params)
        elif method == 'POST':
            resp = session.post(url, params=params)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        if handle_auth_failure(resp):
            return None
        if handle_rate_limit(resp):
            continue
        if resp.ok:
            return resp.json()
        raise ApiException(f"API request failed: {resp.text}")

# this helper method returns the current 'tick' of the running case
def get_tick(session):
    response = api_request(session, 'GET', 'case')
    return response['tick'] if response else None


# this helper method returns the current 'period' of the running case
def get_period(session):
    response = api_request(session, 'GET', 'case')
    return response['period'] if response else None

# this helper method returns the bid and ask for a given security
def ticker_bid_ask(session, ticker):
    response = api_request(session, 'GET', 'securities/book', params={'ticker': ticker})
    if response:
        return response['bids'][0]['price'], response['asks'][0]['price']
    return None, None

# this helper method returns all available assets at the current tick
def get_available_assets(session):
    response = api_request(session, 'GET', 'assets')
    return response if response else None


# this helper method returns all the available securities at the current tick
'''
[
  {
    "ticker": "string",
    "type": "SPOT",
    "size": 0,
    "position": 0,
    "vwap": 0,
    "nlv": 0,
    "last": 0,
    "bid": 0,
    "bid_size": 0,
    "ask": 0,
    "ask_size": 0,
    "volume": 0,
    "unrealized": 0,
    "realized": 0,
    "currency": "string",
    "total_volume": 0,
    "limits": [
      {
        "name": "string",
        "units": 0
      }
    ],
    "interest_rate": 0,
    "is_tradeable": true,
    "is_shortable": true,
    "start_period": 0,
    "stop_period": 0,
    "description": "string",
    "unit_multiplier": 0,
    "display_unit": "string",
    "start_price": 0,
    "min_price": 0,
    "max_price": 0,
    "quoted_decimals": 0,
    "trading_fee": 0,
    "limit_order_rebate": 0,
    "min_trade_size": 0,
    "max_trade_size": 0,
    "required_tickers": "string",
    "bond_coupon": 0,
    "interest_payments_per_period": 0,
    "base_security": "string",
    "fixing_ticker": "string",
    "api_orders_per_second": 0,
    "execution_delay_ms": 0,
    "interest_rate_ticker": "string",
    "otc_price_range": 0
  }
]
'''
def get_available_securities(session):
    response = api_request(session, 'GET', 'securities')
    return response if response else None


# this helper method returns the news at the current tick
def get_news(session):
    response = api_request(session, 'GET', 'news')
    return response[0] if response else None

def place_order(session, ticker, order_type, quantity, action):
    api_request(session, 'POST', 'orders', 
                params={'ticker': ticker, 'type': order_type, 'quantity': quantity, 'action': action})

def main():
    with requests.Session() as s:
        s.headers.update(API_KEY)
        tick = get_tick(s)
        period = get_period(s)
        while period >= 0 and period <= 2 and not shutdown:
            try:
                # if there is a news, print it
                news = get_news(s)
                prices = get_available_assets(s)
                # print all the available securities
                securities = get_available_securities(s)
                print(securities)
                break
                
                
                # IMPORTANT to update the tick at the end of the loop to check that the algorithm should still run or not
                tick = get_tick(s)
                period = get_period(s)

            except ApiException as e:
                print(f"API error: {str(e)}")
                sleep(1)

if __name__ == '__main__':
    # register the custom signal handler for graceful shutdowns
    signal.signal(signal.SIGINT, signal_handler)
    main()