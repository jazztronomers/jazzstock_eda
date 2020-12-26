import jazzstock_bot.common.connector_db as db
import pandas as pd
import numpy as np
import socket

'''
월별 PSMAR, VSMAR 모양을 살펴보는 함수

'''


ip = socket.gethostbyname('jazztronomers.iptime.org')


def get_date_by_cnt(cnt):
    date = db.selectSingleValue(
        '''SELECT CAST(DATE AS CHAR) AS DATE FROM jazzdb.T_DATE_INDEXED WHERE CNT = "%s"''' % (cnt))
    return date


def get_stockcode_alerted(the_date):
    df = db.selectpd(f'''
    SELECT STOCKCODE, CAST(DATE AS CHAR) AS DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOLUME, 
            PSMAR5, PSMAR20, PSMAR60, VSMAR5, VSMAR20, VSMAR60 
    FROM jazzdb.T_STOCK_MIN_05_SMAR 
    JOIN jazzdb.T_STOCK_OHLC_MIN USING(STOCKCODE, DATE, TIME) 
    WHERE 1=1 
    AND DATE = "{the_date}"
    AND PSMAR60 > 0.02
    AND VSMAR20 > 5
    ''')
    df['TRADINGVALUE'] = df['CLOSE'] * df['VOLUME']

    return df[df['TRADINGVALUE'] > 100000000].reset_index(drop=True)


def get_stockcode_well_snd(cnt):
    query = """

                    SELECT STOCKCODE
                    FROM jazzdb.T_STOCK_SND_ANALYSIS_RESULT_TEMP
                    JOIN jazzdb.T_DATE_INDEXED USING(DATE)
                    WHERE 1=1
                    AND CNT = '%s'
                    AND (I5>0.003 OR F5 > 0.003)
                                                    """ % (cnt)

    return db.selectpd(query)


