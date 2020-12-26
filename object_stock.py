import jazzstock_bot.common.connector_db as db
import pandas as pd
import numpy as np
import time
import socket
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import mean_squared_error


pd.options.display.max_rows = 1000
pd.options.display.max_columns= 500
pd.options.display.expand_frame_repr=False

class stock_markdown:
    '''
    조건식 만드는 객체
    '''
    def __init__(self, stockcode,
                 the_date,
                 prev_date,
                 start_date,
                 df_alert=None,
                 dev=False,
                 verbose=False):

        self.stockname = db.selectSingleValue(
            'SELECT STOCKNAME FROM jazzdb.T_STOCK_CODE_MGMT WHERE 1=1 AND STOCKCODE = "%s"' % (stockcode))
        self.stockcode = stockcode
        self.the_date = the_date
        self.start_date = start_date
        self.prev_date = prev_date
        self.df_alert = df_alert
        self.verbose = verbose
        self.df_bb_desc = None

        self.message_dic = {
            'LD': '볼밴 하단 하향 돌파',
            'LU': '볼밴 하단 상향 돌파',
            'MD': '볼밴 중단 상향 돌파',
            'MU': '볼밴 중단 하향 돌파',
            'UU': '볼밴 상단 상향 돌파',
            'UD': '볼밴 상단 하향 돌파'
        }

        if verbose:
            self.get_stock_info()

        if not dev:
            self.load_all()

    def get_stock_info(self, verbose=True):
        print('=' * 100)
        print(f'{self.stockname}({self.stockcode}) MARKDOWN')
        print(f'THE DATE (ALERT) : {self.the_date}')
        print(f'PREV PERIOD : {self.start_date} ~ {self.prev_date}')

    
    
    # ==============================================================================
    # DB에서 종목정보를 뽑아서 Dataframe을 구성하는 함수를 모두 호출
    # ==============================================================================
    def load_all(self):

        self.df_min_last_few_days = self.get_min_last_few_days(self.stockcode, self.start_date, self.prev_date)
        self.df_min_the_day = self.get_min_the_day(self.stockcode, self.the_date)
        self.df_snd_last_day, self.df_snd_previous_period = self.get_snd_previous(self.stockcode, self.start_date, self.prev_date)
        self.df_future = self.get_future(self.stockcode, self.the_date)
        self.get_snd_cumsum(self.stockcode, self.start_date, self.prev_date)
        self.get_bb_description(verbose=False)
        self.get_bb_event_interval()
        
    # ==============================================================================
    # DB에서 종목정보를 뽑아서 Dataframe을 구성하는 개별함수
    # ==============================================================================
    def get_min_last_few_days(self, stockcode, start_date, prev_date):

        df_min_last_few_days = db.selectpd(f'''
                SELECT STOCKCODE, DATE, TIME, CLOSE, VOLUME, 
                    PSMAR5, PSMAR5, PSMAR20, PSMAR60, VSMAR5, VSMAR20, VSMAR60
                FROM jazzdb.T_STOCK_MIN_05_SMAR 
                JOIN jazzdb.T_STOCK_OHLC_MIN USING(STOCKCODE, DATE, TIME) 
                WHERE 1=1 
                AND DATE BETWEEN "{start_date}" AND "{prev_date}"
                AND STOCKCODE = "{stockcode}"
             ''')

        return df_min_last_few_days

    def get_min_the_day(self, stockcode, the_date):

        df_min_the_day = db.selectpd(
            f'SELECT TIME, OPEN, HIGH, LOW, CLOSE, VOLUME FROM jazzdb.T_STOCK_OHLC_MIN WHERE 1=1 AND DATE = "{the_date}" AND STOCKCODE = "{stockcode}"')
        return df_min_the_day

    def get_snd_previous(self, stockcode, start_date, prev_date):

        df_snd_previous = db.selectpd(f'''
        SELECT DATE, C.OPEN, C.HIGH, C.LOW, C.CLOSE,
                MA5, MA20, MA60, MA120, VMA5, VMA20, VMA60, VMA120, 
                BBU, BBL, BBP, BBW,
                P1, P5, P20, P60, P120,
                I1, I5, I20, I60, I120, IR,
                F1, F5, F20, F60, F120, FR

        FROM jazzdb.T_STOCK_SND_ANALYSIS_RESULT_TEMP A
        JOIN jazzdb.T_STOCK_SND_ANALYSIS_LONGTERM B USING (STOCKCODE, DATE)

        JOIN jazzdb.T_STOCK_OHLC_DAY C USING (STOCKCODE, DATE)
        JOIN jazzdb.T_STOCK_BB D USING (STOCKCODE, DATE)
        JOIN jazzdb.T_STOCK_MA E USING (STOCKCODE, DATE)
        WHERE 1=1
        AND STOCKCODE = "{stockcode}" 
        AND DATE BETWEEN "{start_date}" AND "{prev_date}"
        ''')

        df_snd_last_day = df_snd_previous.tail(1)
        df_snd_previous_period = df_snd_previous[['DATE',
                                                  'OPEN', 'HIGH', 'LOW', 'CLOSE',
                                                  'MA5', 'MA20', 'MA60', 'MA120',
                                                  'VMA5', 'VMA20', 'VMA60', 'VMA120',
                                                  'BBU', 'BBL', 'BBP', 'BBW',
                                                  'IR', 'FR']]

        return df_snd_last_day, df_snd_previous_period

    def get_snd_cumsum(self, stockcode, start_date, prev_date):

        df_snd = db.selectpd(f'''
        SELECT DATE, INS, FOREI, PER, YG, SAMO, TUSIN, FINAN
        FROM jazzdb.T_STOCK_SND_DAY
        WHERE 1=1
        AND STOCKCODE = "{stockcode}" 
        AND DATE BETWEEN "{start_date}" AND "{prev_date}"
        ''')

        for each_col in "INS, FOREI, PER, YG, SAMO, TUSIN, FINAN".split(', '):
            self.df_snd_previous_period[each_col] = df_snd[each_col].cumsum()

    def get_future(self, stockcode, the_date):

        df_future = db.selectpd(f'''
        SELECT PRO1, PRO3, PRO5, PRO10, PRO20, PRH1, PRH3, PRH5, PRH10, PRH20, PRL1, PRL3, PRL5, PRL10, PRL20
        FROM jazzdb.T_STOCK_FUTURE_PRICE
        WHERE 1=1
        AND STOCKCODE = "{stockcode}"
        AND DATE = "{the_date}"
        ''')

        return df_future

    def mean(self, lst):
        return sum(lst) / len(lst)

    def _calc_rank(self, value, itrlist):

        mean_top_20 = self.mean(sorted(itrlist, reverse=True)[:20])
        itrlist.append(value)
        itrlist = sorted(itrlist, reverse=True)
        percentage = (value - mean_top_20) / mean_top_20

        return itrlist.index(value), percentage

    def get_chart(self, feature_columns=['CLOSE', 'BBP', 'BBW', 'INS', 'FOREI', 'SAMO', 'YG'], show=True, threshold=0,
                  mid=0.5, save=False):
        '''

        수급창구별 RANK 추이 확인
        수급창구별 독립비율 확인

        BBP 추이 확인
            상승중인지
            하락중인지
            횡보중인지

        BBW 추이 확인
            변동성이 줄어들고 있는지
                BBW가 커지고 있는지
            변동성이 커지고 있는지
                BBW가 줄어들고 있는지

        '''

        fig, axs = plt.subplots(len(feature_columns), figsize=(10, 15))

        for i, column in enumerate(feature_columns):
            axs[i].plot(np.arange(len(self.df_snd_previous_period)), self.df_snd_previous_period[column])

            if column == 'CLOSE':
                axs[i].plot(np.arange(len(self.df_snd_previous_period)), self.df_snd_previous_period['BBU'])
                axs[i].plot(np.arange(len(self.df_snd_previous_period)), self.df_snd_previous_period['BBL'])

            elif column == 'BBP':

                axs[i].plot(np.arange(len(self.df_snd_previous_period)),
                            np.asarray([mid] * len(self.df_snd_previous_period)), 'r--')
                axs[i].plot(np.arange(len(self.df_snd_previous_period)),
                            np.asarray([threshold] * len(self.df_snd_previous_period)), 'r--')
                axs[i].plot(np.arange(len(self.df_snd_previous_period)),
                            np.asarray([1 - threshold] * len(self.df_snd_previous_period)), 'r--')

            axs[i].set_title(column)

        fig.tight_layout()
        if show:
            plt.show()

        if save:
            plt.savefig('output/%s_%s.png' % (self.stockcode, self.the_date))

    def get_bb_raw_data(self):

        self.df_bb = self.df_snd_previous_period[['BBP', 'BBW']]
        self.df_bb['BBP_GRAD'] = np.gradient(self.df_snd_previous_period['BBP'].values)
        self.df_bb['BBW_GRAD'] = np.gradient(self.df_snd_previous_period['BBW'].values)
        self.df_bb['BBW_IS_INFLECT'] = np.gradient(self.df_snd_previous_period['BBW'].values)

        print(self.df_bb)
        
        
        

    def get_sumamry_alert(self, df_alert=None, verbose=False):
        '''
        알람 관련 Dataframe이 설정되었다면 써머리를 정리해주는 함수
        '''

        if self.verbose or verbose:
            print('=' * 100)
            print(' ** ALERT_SUMMARY')
            print(' *  5분봉지표 VSMAR20, PSMAR60을 근거하여 전송하는 ALERT에 대한 요약')
            print('=' * 100)

        df_alert = df_alert or self.df_alert

        for i, each_row in df_alert.iterrows():
            close_alert = each_row.CLOSE

            rank, percentage = self._calc_rank(each_row.VOLUME, self.df_min_last_few_days.VOLUME.values.tolist())
            df_after_alert = self.df_min_the_day[self.df_min_the_day.TIME > each_row.TIME]

            high_after_alert = df_after_alert.HIGH.max()
            close_the_day = self.df_min_the_day.tail(1).CLOSE.values[0]

            result = {

                "each alert volume이 최근 n거래일중 순위": '%s/%s' % (rank, len(self.df_min_last_few_days)),
                "each alert volume이 최근 n거래일중 max보다 몇퍼센트나 적은지": percentage,
                "당일 종가가 alert가격종가보다 높은지": close_the_day > close_alert,
                "당일 종가가 alert가격종가보다 얼마나 높은지": (close_the_day - close_alert) / close_alert,
                "alert후 고가가 alert가격종가보다 높은지": high_after_alert > close_alert,
                "alert후 고가가 alert가격종가보다 얼마나 높은지": (high_after_alert - close_alert) / close_alert,

            }

        if self.verbose:
            for k, v in result.items():
                print(' ', k, '\t', v)
                
    # ==============================================================================
    # 최종 SUMMARY, TEXT
    # ==============================================================================
    def get_sumamry_day_index(self, header=False, verbose=False, save=False):
        '''
        최종 SUMMARY, TEXT로 구성됨, MARKDOWN 형태로 수정 필요
        '''

        sr = self.df_snd_last_day.round(3).iloc[0]

        bbp = self.df_snd_previous_period.BBP.values.tolist()
        bbw = self.df_snd_previous_period.BBW.values.tolist()

        message = ''
        message = message + '=' * 100 + '\n'
        message = message + '## %s(%s) %s SUMMARY' % (self.stockname, self.stockcode, self.the_date) + '\n'
        message = message + '=' * 100 + '\n'

        message = message + 'OHLC | %5d %5d %5d %5d' % (sr.OPEN, sr.HIGH, sr.LOW, sr.CLOSE) + '\n'
        message = message + 'BBP  | %+.3f %+.3f %+.3f %+.3f ------' % (
        bbp[-1], self.mean(bbp[-4:]), self.mean(bbp[-19:]), self.mean(bbp)) + '\n'
        message = message + 'BBW  | %+.3f %+.3f %+.3f %+.3f ------' % (
        bbw[-1], self.mean(bbw[-4:]), self.mean(bbw[-19:]), self.mean(bbw)) + '\n'
        message = message + 'PRF  | %+.3f %+.3f %+.3f %+.3f %+.3f' % (sr.P1, sr.P5, sr.P20, sr.P60, sr.P120) + '\n'
        message = message + 'INS  | %+.3f %+.3f %+.3f %+.3f %+.3f | %5d' % (
        sr.I1, sr.I5, sr.I20, sr.I60, sr.I120, sr.IR) + '\n'
        message = message + 'FOR  | %+.3f %+.3f %+.3f %+.3f %+.3f | %5d' % (
        sr.F1, sr.F5, sr.F20, sr.F60, sr.F120, sr.FR) + '\n'
        message = message + '------------------------' * 3 + '\n'
        message = message + '## BB EVENT:  ' + '\n'
        message = message + '------------------------' * 3 + '\n'
        for each_event in ['LD', 'LU', 'MU', 'MD', 'UU', 'UD']:

            if self.df_bb_recent_event_elapsed_days.iloc[0][each_event] != -1:
                message = message + '%s %3d 거래일전' % (
                self.message_dic[each_event], self.df_bb_recent_event_elapsed_days.iloc[0][each_event]) + '\n'
            else:
                message = message + '%s 기록없음' % (self.message_dic[each_event]) + '\n'

        message = message + '## BB EVENT DETAIL:  ' + '\n'
        message = message + str(self.df_bb_desc) + '\n'

        message = message + '## FUTURE:  ' + '\n'

        message = message + str(self.df_future) + '\n'

        if self.verbose or verbose:
            print(message)

        if save:
            f = open('output/%s_%s.md' % (self.stockcode, self.the_date), 'w', encoding='utf-8')
            f.write(message)
            f.close()

            

    def get_bb_description(self, bbp=None, bbw=None, threshold=0, mid=0.5, dev=False, verbose=False):
        '''
        볼린저밴드 해석하기 위한 함수
        
        return
            1) 특이점 x,y 좌표, 상태값

        '''

        if bbp is None:
            bbp = self.df_snd_previous_period.BBP.values
        if bbw is None:
            bbw = self.df_snd_previous_period.BBW.values

        if isinstance(bbp, list):
            bbp = np.asarray(bbp)
        if isinstance(bbp, list):
            bbw = np.asarray(bbw)

        ret_df = pd.DataFrame(columns=['PREV_IDX', 'DATE_IDX', 'INTERVAL', 'EVENT', 'PREVENT', 'GRAD', 'BBW'])
        bbp_grad = np.gradient(bbp).round(2)

        prev_point = 0
        prev_event = '-'

        '''
        LD: BB LOWERBAND DOWNWARD BREAKTHROUGH, 볼린저밴드 하단 하향돌파
        LU: BB LOWERBAND UPWARD BREAKTHROUGH, 볼린저밴드 하단 상향돌파
        MD: 20MA BAND DOWNWARD BREAKTHROUGH, 20일이평선 상향 돌파
        MU: 20MA BAND DOWNWARD BREAKTHROUGH, 20일이평선 하향 돌파
        UU: BB UPPERBAND DOWNWARD BREAKTHROUGH, 볼린저밴드 상단 상향돌파
        UD: BB UPPERBAND UPWARD BREAKTHROUGH,볼린저밴드 상단 하향돌파

        LD, UU

        '''
        for i in range(0, len(bbp) - 1):

            # BBP 상향돌파
            if bbp[i + 1] > bbp[i]:

                if bbp[i] < 1 - threshold <= bbp[i + 1]:
                    ret_df.loc[len(ret_df)] = [prev_point, i, i - prev_point, 'UU', prev_event, bbp_grad[i + 1],
                                               bbw[i + 1]]
                    prev_point = i
                    prev_event = 'UU'

                elif bbp[i] < mid <= bbp[i + 1]:
                    ret_df.loc[len(ret_df)] = [prev_point, i, i - prev_point, 'MU', prev_event, bbp_grad[i + 1],
                                               bbw[i + 1]]
                    prev_point = i
                    prev_event = 'MU'

                elif bbp[i] < threshold <= bbp[i + 1]:
                    ret_df.loc[len(ret_df)] = [prev_point, i, i - prev_point, 'LU', prev_event, bbp_grad[i + 1],
                                               bbw[i + 1]]
                    prev_point = i
                    prev_event = 'LU'


            # BBP 하향돌파
            elif bbp[i + 1] < bbp[i]:

                if bbp[i] > threshold >= bbp[i + 1]:
                    ret_df.loc[len(ret_df)] = [prev_point, i, i - prev_point, 'LD', prev_event, bbp_grad[i + 1],
                                               bbw[i + 1]]
                    prev_point = i
                    prev_event = 'LD'

                elif bbp[i] > mid >= bbp[i + 1]:
                    ret_df.loc[len(ret_df)] = [prev_point, i, i - prev_point, 'MD', prev_event, bbp_grad[i + 1],
                                               bbw[i + 1]]
                    prev_point = i
                    prev_event = 'MD'

                elif bbp[i] > 1 - threshold >= bbp[i + 1]:
                    ret_df.loc[len(ret_df)] = [prev_point, i, i - prev_point, 'UD', prev_event, bbp_grad[i + 1],
                                               bbw[i + 1]]
                    prev_point = i
                    prev_event = 'UD'

        ret_df.loc[len(ret_df)] = [prev_point, i, i - prev_point, '--', prev_event, bbp_grad[i + 1], bbw[i + 1]]

        if dev:
            spoint = [0]
            for i in range(len(bbp) - 1):
                change_size = (bbp_grad[i] - bbp_grad[i + 1]).round(2)
                sign_change = change_size > 0
                if change_size > 0.09 and sign_change:
                    spoint.append(i)
                    plt.scatter(i, bbp[i])

            spoint.append(len(bbp) - 1)
            prev_slope = 0
            for i in range(len(spoint) - 1):
                x1 = spoint[i]
                x2 = spoint[i + 1]
                y1 = bbp[spoint[i]]
                y2 = bbp[spoint[i + 1]]

                x_values = [x1, x2]
                y_values = [y1, y2]
                plt.plot(x_values, y_values)

        self.df_bb_desc = ret_df.copy()
        return ret_df[['PREV_IDX', 'DATE_IDX', 'INTERVAL', 'PREVENT', 'EVENT', 'GRAD', 'BBW']]

    def get_bb_event_interval(self, verbose=False):
        self.df_bb_recent_event_elapsed_days = pd.DataFrame(
            columns=['STOCKCODE', 'DATE', 'LD', 'LU', 'MU', 'MD', 'UU', 'UD'],
            data=[[self.stockcode, self.prev_date, -1, -1, -1, -1, -1, -1]])

        for each_event in ['LD', 'LU', 'MU', 'MD', 'UU', 'UD']:

            if each_event in self.df_bb_desc['PREVENT'].unique():
                self.df_bb_recent_event_elapsed_days.at[0, each_event] = 60 - self.df_bb_desc[
                    self.df_bb_desc['PREVENT'] == each_event].tail(1).PREV_IDX.values[0]

        return self.df_bb_recent_event_elapsed_days

    # ==============================================================================
    # FEATURE COLUMNS 생성부
    # ==============================================================================
    def is_low_volatility(self, threshold = 0.05):
        '''
        최근 10거래일간 변동성이 크지 않은지 True False로 반환하는 함수
        '''

        high_5 = self.df_snd_previous_period.tail(5).HIGH.max()
        high_10 = self.df_snd_previous_period.tail(10).HIGH.max()
        high_15 = self.df_snd_previous_period.tail(15).HIGH.max()

        low_5 = self.df_snd_previous_period.tail(5).LOW.min()
        low_10 = self.df_snd_previous_period.tail(10).LOW.min()
        low_15 = self.df_snd_previous_period.tail(15).LOW.min()

        close = self.df_snd_last_day.tail(1).CLOSE.values[0]

        fluct_high_5 = (high_5 - close) / close
        fluct_high_10 = (high_10 - close) / close
        fluct_high_15 = (high_15 - close) / close

        fluct_low_5 = (low_5 - close) / -close
        fluct_low_10 = (low_10 - close) / -close
        fluct_low_15 = (low_15 - close) / -close

        fluct_high = max(fluct_high_5, fluct_high_10) < threshold
        fluct_low = max(fluct_low_5, fluct_low_10) < threshold

        return fluct_high and fluct_low
    
    
    

    def is_snd_ins_positive(self, threshold=0.008):        
        '''
        최근 5거래일간 기관수급이 양수인지 판단하는 함수
        '''

        if self.df_snd_last_day.I5.values[0] > threshold:
            return True

        else:
            return False

    def is_snd_for_positive(self, threshold=0.008):
        '''
        최근 5거래일간 외인수급이 양수인지 판단하는 함수
        '''


        if self.df_snd_last_day.F5.values[0] > threshold:
            return True

        else:
            return False

    def is_bbp_positive(self):
        '''
        최근 거래일에 볼린저밴드상 주가흐름의 패턴이 어떠한지 판단하는 함수
        
        LUMU : 밴드하단 상향돌파, 밴드 중단 상향돌파
        LDLUMU : 밴드하단 하향돌파후 밴드하단 상향돌파, 밴드중단 상향돌파
        
        일단은 True False 만 판단하고 있지만, 밴드중단을 상향 하향 왔다리 갔다리 하는게 있는지
        하단을 긁고 내려가고 있는지 등을 판단하도록 추가 구현 하면 좋을듯
        '''
        ret = ''.join(self.df_bb_desc.PREVENT.values.tolist())
        if ret[-4:] in ['LUMU'] or ret[-6] in ['LDLUMU']:

            return True
        else:
            return False

    def is_bbw_narrow(self):
        '''
        최근거래일간 볼린저밴드 폯이 좁은지 판단하는 함수
        
        '''

        # print(self.df_snd_last_day.BBW.values[0], self.df_snd_previous_period.tail(10).BBW.mean())

        if self.df_snd_last_day.BBW.values[0] < 0.15 and self.df_snd_previous_period.tail(10).BBW.mean() < 0.18:
            return True
        else:
            return False

    def is_close_under_ma(self, window=60):
        '''
        현주가가 이동평균선 이하인지 판단하는 함수
        '''

        if self.df_snd_last_day.CLOSE.values[0] < self.df_snd_last_day['MA60'].values[0]:
            return True
        else:
            return False

    def is_close_over_ma(self, window=5):
        '''
        현주가가 이동평균선 이상인지 판단하는 함수
        '''
        if self.df_snd_last_day.CLOSE.values[0] > self.df_snd_last_day['MA5'].values[0]:
            return True
        else:
            return False

