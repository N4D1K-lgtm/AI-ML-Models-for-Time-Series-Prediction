import pandas as pd
import os

def get_vol(prices, span=100, hours=1):
    delta=pd.Timedelta(hours=hours)
    # 1. compute returns of the form prices[t]/prices[t-1] - 1
    # 1.1 find the timestamps of prices[t-1] values
    df0 = prices.index.searchsorted(prices.index - delta)
    df0 = df0[df0 > 0]
    # 1.2 align timestamps of prices[t-1] to timestamps of prices[t]
    df0 = pd.Series(prices.index[df0-1], index=prices.index[prices.shape[0]-df0.shape[0] : ])
    # 1.3 get values by timestamps, then compute returns
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values - 1
    # 2. estimate rolling standard deviation
    df0 = df0.ewm(span=span).std()
    return df0

def get_horizons(prices, minutes=5):
    delta=pd.Timedelta(minutes=minutes)
    t1 = prices.index.searchsorted(prices.index + delta)
    t1 = t1[t1 < prices.shape[0]]
    t1 = prices.index[t1]
    t1 = pd.Series(t1, index=prices.index[:t1.shape[0]])
    return t1

def get_touches(prices, events, factors=[1, 1]):
    '''
    events: pd dataframe with columns
    t1: timestamp of the next horizon
    threshold: unit height of top and bottom barriers
    side: the side of each bet
    factors: multipliers of the threshold to set the height of top/bottom barriers
    '''
    out = events[['t1']].copy(deep=True)
    if factors[0] > 0: thresh_uppr = factors[0] * events['threshold']
    else: thresh_uppr = pd.Series(index=events.index) # no upper threshold
    if factors[1] > 0: thresh_lwr = -factors[1] * events['threshold']
    else: thresh_lwr = pd.Series(index=events.index)  # no lower threshold
    for loc, t1 in events['t1'].iteritems():
        df0=prices[loc:t1]                              # path prices
        df0=(df0 / prices[loc] - 1) * events.side[loc]  # path returns
        out.loc[loc, 'stop_loss'] = df0[df0 < thresh_lwr[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'take_profit'] = df0[df0 > thresh_uppr[loc]].index.min() # earliest take profit
    return out

def get_labels(touches):
    out = touches.copy(deep=True)
    # pandas df.min() ignores NaN values
    first_touch = touches[['stop_loss', 'take_profit']].min(axis=1)
    for loc, t in first_touch.iteritems():
        if pd.isnull(t):
            out.loc[loc, 'label'] = 0
        elif t == touches.loc[loc, 'stop_loss']: 
            out.loc[loc, 'label'] = -1
        else:
            out.loc[loc, 'label'] = 1
    return out

def load_data(path):
    # Import and Process Data
    out = pd.DataFrame()
    path = path
    files = os.listdir(path)
    names = []

    for file in files:
        
        name = str(file)
        name = name.replace(".json", "")
        names.append(name)
        df = pd.read_json(path + str(file))
        df.rename(columns = {0 : 'Time', 1 : 'Open', 2 : 'High', 3 : 'Low', 4 : 'Close', 5 : 'Volume'}, inplace=True)
        df["Timestamp"] = pd.to_datetime(df["Time"], unit="ms")
        df.drop("Time", axis=1, inplace=True)
        df.set_index('Timestamp', inplace=True)
        series = pd.Series(df['Close'], index=df.index)
        out[name] = series

    # out = out.fillna()
    return out, names

def label_series(df, span=50, hours=1, minutes=15):
    events = pd.DataFrame()
    labels = pd.DataFrame()
    touches = pd.DataFrame()

    events = events.assign(threshold=get_vol(df, span=span, hours=hours)).dropna()
    events = events.assign(t1=get_horizons(df, minutes=minutes)).dropna()
    events = events.assign(side=pd.Series(1., events.index)) # long only

    touches = get_touches(df, events, [.5, .5])
    touches = get_labels(touches)
    labels = touches.label
    
    return labels