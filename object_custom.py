from jazzstock_eda.object_stock import stock_markdown



# 메인객체를 상속받아서 나만의 함수를 작성하는 템플릿
class stock_custom(stock_markdown):


    def __init__(self, stockcode,
                 the_date,
                 prev_date,
                 start_date,
                 df_alert=None,
                 dev=False,
                 verbose=False):

        # 상속받은 부모객체의 INIT 부터 먼저 실행하고
        super().__init__(stockcode, the_date, prev_date, start_date, df_alert, dev, verbose)

        # 나만의 __init__을 실행한다
        print('* 매인객체에는 없는 프린트문을 추가해보았지')

    def load_all(self):

        # 부모객체의 load_all()을 먼저 실행하고
        super().load_all()

        # 나의 load_all() 을 실행한다
        self.get_dummy_column() # get_dummy_column은 그냥 true를 반환하지만 self.df_snd_previous_period 에다 my_index라는 컬럼을 추가해주었다
        print('* 내것도 LOAD 해줘!')


    def my_function(self):
        # 지표생성할땐 df_snd_previos_period 만 사용하면 됨, 나머지는 메인객체 직접 참고하도록

        print(self.df_snd_previous_period.columns)
        print('* 나만의 함수를 실행해보았다')

        return True

    def get_dummy_column(self):
        # df_snd_previos_period 에 지표를 추가하고 싶다!
        # 그러면 다음과 같이 해보셈
        print('BEFORE', self.df_snd_previous_period.columns)
        self.df_snd_previous_period['my_index']= 5
        print('AFTER', self.df_snd_previous_period.columns)
        return True


    def is_last_trading_day_p5_positive(self, threshold=0):
        # 논리함수 CUSTOM 정의
        print(self.df_snd_last_day.transpose())

        return self.df_snd_last_day.P5.values[0] > threshold

if __name__ == '__main__':

    stockcode = '079940'
    the_date = '2020-12-24'
    prev_date = '2020-12-24'  #  운영시뮬레이션이 아닌, 장종료후 분석하면 전일자가 최근일자 이므로 the_date 와 같은 날짜를 주면됨
    start_date = '2020-09-28' #  60거래일전

    stock = stock_custom(stockcode  = stockcode,
                 the_date  = the_date,
                 prev_date = prev_date,
                 start_date = start_date)


    # ===============================================================================
    # 부모객체에서 만들어주는 dataframe들을 참고하려면 아래 프린트문을 활성화 해보도록
    print(stock.df_snd_previous_period.tail(3))
    # print(stock.df_snd_last_day)
    # print(stock.df_min_the_day)
    # print(stock.df_min_last_few_days)
    # print(stock.df_bb_desc)
    # print(stock.df_bb_recent_event_elapsed_days)
    # ===============================================================================



    ret = stock.my_function()
    print('='*100)

    result = stock.is_last_trading_day_p5_positive(threshold=0)
    print(F'\n\n그래서, P5는 0보다 큽니까? {result}')

