# This is a python example algorithm using DMA REST API for the RIT ALGO1 Case
import signal
import requests
from time import sleep
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
def get_available_securities(session):
    response = api_request(session, 'GET', 'securities')
    return response if response else None


# this helper method returns the news at the current tick
def get_news(session):
    response = api_request(session, 'GET', 'news')
    return response[0] if response else None


def get_tradable_securities(securities):
    """
    Returns a dictionary with only the securities where 'tradable' is True.

    Parameters:
        securities (dict): All securities as a dictionary.

    Returns:
        dict: Dictionary containing only tradable securities.
    """
    return {ticker: info for ticker, info in securities.items() if info['is_tradeable']}


def get_ask_price(session, securities):
    """
    Retrieves the ask prices for each security in the given dictionary.

    Parameters:
        session (requests.Session): The session to use for API requests.
        securities (dict): A dictionary of tradable securities.

    Returns:
        dict: A dictionary containing the ask prices for each security.
    """
    ask_prices = {}
    for ticker in securities.keys():
        ask_price, _ = ticker_bid_ask(session, ticker)
        ask_prices[ticker] = ask_price
    return ask_prices


def news_dict_to_string(news):
    """
    Converts a dictionary of news to a string.

    Parameters:
        news (dict): The news dictionary.

    Returns:
        str: The news as a string.
    """
    return f"{news['headline']} - {news['body']}"


def place_order(session, ticker, order_type, quantity, action):
    api_request(session, 'POST', 'orders', 
                params={'ticker': ticker, 'type': order_type, 'quantity': quantity, 'action': action})


def compile_trainable_data(session_start_datestr, period, ticker, tradable_securities_ask_prices, news):
    """
    Compiles the data to a dict be used for training the model.
    
    """
    trainable_data = {
        'session_start': session_start_datestr,
        'period': period,
        'ticker': ticker,
        'ask_prices': tradable_securities_ask_prices,
        'news': news
    }
    return trainable_data


def case_active(session):
    response = api_request(session, 'GET', 'case')
    # if status is ACTIVE, return True
    return response['status'] == 'ACTIVE'


def main():
    while True:
        with requests.Session() as s:
            session_start_datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            previous_news = None
            previous_trainable_data = None
            s.headers.update(API_KEY)

            inited = False

            data_stored = False

            trainable_data_list = []

            counter = 0
    
            while case_active(s) and not shutdown and counter < 10:
                try:
                    tick = get_tick(s)
                    period = get_period(s)

                    if not inited:
                        securities = get_available_securities(s)
                        securities_dict = {}
                        for security in securities:
                            securities_dict[security['ticker']] = security
                        # get only the tradable securities
                        tradable_securities = get_tradable_securities(securities_dict)
                        inited = True
                    

                    # if there is a news, print it
                    news = get_news(s)
                    news = news_dict_to_string(news)
                    
                    if previous_news is None:
                        previous_news = news
                    elif previous_news != news:
                        previous_news = news
                    else:
                        news = None
                    tradable_securities_ask_prices = get_ask_price(s, tradable_securities)
                    # compile the data to be used for training the model
                    trainable_data = compile_trainable_data(session_start_datestr, period, tick, tradable_securities_ask_prices, news)
                    
                    # check if the ticker in the previous trainable data is the same as the current one
                    if previous_trainable_data is None:
                        previous_trainable_data = trainable_data
                        print(trainable_data)
                        counter += 1
                        trainable_data_list.append(trainable_data)
                    elif previous_trainable_data['ticker'] != trainable_data['ticker']:
                        previous_trainable_data = trainable_data
                        print(trainable_data)
                        counter += 1
                        trainable_data_list.append(trainable_data)

                

                except ApiException as e:
                    print(f"API error: {str(e)}")
                    sleep(1)

                sleep(0.2)

            if not data_stored and trainable_data_list != []:
                    print("Storing the data...")

                    # turn the list of trainable data into a pandas dataframe

                    df = pd.DataFrame(trainable_data_list)
                    table = pa.Table.from_pandas(df)
                    pq.write_table(table, f"data\data_session_{session_start_datestr}.parquet")
                    
                    data_stored = True
    
            while not case_active(s) and not shutdown:
                print("Waiting for the case to start...")
                sleep(1)

if __name__ == '__main__':
    # register the custom signal handler for graceful shutdowns
    signal.signal(signal.SIGINT, signal_handler)
    main()