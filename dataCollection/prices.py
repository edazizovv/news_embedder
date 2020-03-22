import pandas as pd
from datetime import datetime, timedelta
import requests
import calendar
from functools import partial
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def printTimes(closestBefore, closestAfter, newsDate):
    closestDateBefore = datetime.utcfromtimestamp(closestBefore / 1000)
    closestDateAfter = datetime.utcfromtimestamp(closestAfter / 1000)
    print(f"NewsTime: {newsDate}({calendar.day_name[newsDate.weekday()]}) " + 
        f"ClosestBefore: {closestDateBefore} ({calendar.day_name[closestDateBefore.weekday()]}) " +            
        f"ClosestAFter: {closestDateAfter} ({calendar.day_name[closestDateAfter.weekday()]})")

def timeChooserBefore(newsTime, val):
    res = newsTime - val
    if (res < 3600000):
        return 10000000000 + res # just big number
    return res

def timeChooserAfter(newsTime, val):
    res = val - newsTime
    if (res < 3600000):
        return 10000000000 + res # just big number
    return res

def getTimestamp(news):
    utcTime = news.PublishDate
    newsTime = utcTime.timestamp()
    past = (utcTime - timedelta(days=3)).timestamp()
    future = (utcTime + timedelta(days=3)).timestamp()
    return (int(past), int(newsTime), int(future))

def getHistory(symbol, past, future, session):
    url = f"https://api.alor.ru/md/history?code={symbol}&exchange=MOEX&tf=3600&from={past}&to={future}&format=TV"
    r = session.get(url)
    # If stock got delisted or never existed, we got 404. For example: PSKB
    if(r.status_code == 200):
        json = r.json()["history"]
        if (len(json) == 0):
            # Beware of happy new year (MOEX exhange don't work about a week)
            return None
        return json    

def requestsRetrySession(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

dataFull = pd.read_excel('data/dataFull.xlsx')

print(f"All entries: {dataFull.Id.size}, unique symbols: {dataFull.Symbols.unique().size}")
candles = pd.DataFrame(columns=['Id', 'openBefore', 'timeBefore', 'openAfter', 'timeAfter'])
session = requestsRetrySession(retries=10)
for _, news in dataFull.iterrows():
    # All timestamps returned by API is in UTC
    (past, newsTime, future) = getTimestamp(news)
    # they'll cut you when you try to open to many simultanious connections\
    # so it gonna be slow (5k entities costs me 10 minutes with slow connection)
    requestResult = getHistory(news.Symbols, past, future, session)
    if (requestResult is None):
        continue
    history = pd.DataFrame(requestResult)
    
    # Returns the most recent candle time BEFORE and AFTER a news event.
    # Usually it's one hour before and 1 hour after, so trades have about 2 hours to react
    closestBefore = min(history.time, key=partial(timeChooserBefore, newsTime * 1000))
    closestAfter = min(history.time, key=partial(timeChooserAfter, newsTime * 1000))

    # Debug stuff
    printTimes(closestBefore, closestAfter, news.PublishDate)

    before = history[history.time == closestBefore]
    after = history[history.time == closestAfter]
    newCandle = [{
        'Id': news.Id, 
        'openBefore': before.open.values[0], 
        'timeBefore': datetime.utcfromtimestamp(closestBefore / 1000), 
        'openAfter':after.open.values[0], 
        'timeAfter': datetime.utcfromtimestamp(closestAfter / 1000)
    }]
    candles = candles.append(newCandle, ignore_index=True,sort=False)
result = pd.merge(dataFull, candles, how='inner', on='Id')
result.to_excel("result.xlsx")
