from jazzstock_eda.util_stockcode_manager import get_date_by_cnt, get_stockcode_alerted, get_stockcode_well_snd
from jazzstock_eda.object_stock import stock_markdown
import pandas as pd



mode = ['단일종목','알람종목']

mode = 'p'
the_date_cnt = 0
if mode == 'p':
    day_gap = 0
else:
    day_gap = 1


the_date = get_date_by_cnt(the_date_cnt)
prev_date = get_date_by_cnt(the_date_cnt+ day_gap)
start_date = get_date_by_cnt(the_date_cnt+ day_gap + 59)


def true_counter(l):

    cond_cnt = 0
    for each in l:
        if each:
            cond_cnt+=1

    return cond_cnt, l


stockcodes = ['019540']
stockcodes_len = len(stockcodes)

result_df = pd.DataFrame(columns=['STOCKCODE','COUNT', 'LOW_VOLA', 'POS_SND', 'ASC_BBP', 'NAR_BBW', 'UNDER_MA60', 'OVER_MA5'])

for j, stockcode in enumerate(stockcodes):

    try:
        stock = stock_markdown(stockcode=stockcode, \
                               the_date=the_date, \
                               prev_date=prev_date, \
                               start_date=start_date, \
                               df_alert=None, \
                               verbose=False)

        low_volatility = stock.is_low_volatility()  # 금일종가 상하 플마 4% 밴드범위 안에 10일고점, 저점이 포함되는지
        positive_ins_snd = stock.is_snd_ins_positive()  # 금일 기관수급 또는 외인수급 (jazzstock 지표 I5, F5)가 0.008 이상인지  (OR 조건)
        positive_for_snd = stock.is_snd_for_positive()  # 금일 기관수급 또는 외인수급 (jazzstock 지표 I5, F5)가 0.008 이상인지  (OR 조건)
        positive_snd = positive_for_snd or positive_ins_snd
        positive_bbp = stock.is_bbp_positive()  # 30거래일이네 볼린저밴드 하단밴드를 상향돌파한 이력이 있으며, 5거래일이내 20이평선을 상향돌파하였는지
        bbw_narrow = stock.is_bbw_narrow()
        close_under_ma60 = stock.is_close_under_ma(window=60)
        close_over_ma5 = stock.is_close_over_ma(window=5)

        feature_columns = [low_volatility, positive_snd, positive_bbp, bbw_narrow, close_under_ma60, close_over_ma5]

        cond_cnt, l = true_counter(feature_columns)
        result_df.loc[len(result_df)]= [stockcode, cond_cnt, low_volatility, positive_snd, positive_bbp, bbw_narrow, close_under_ma60, close_over_ma5]

        print(j, '/', len(stockcodes), stockcode, cond_cnt, l, '**' if cond_cnt==6 else '')
        # print(result_df.tail(5))
        if cond_cnt > -1:
            stock.get_sumamry_day_index(verbose=False, save=True)
            stock.get_chart(show=False, save=True)


    except Exception as e:
        print(stockcode, 'ERROR', e)


# result_df.to_csv('result_%s.csv'%(the_date), encoding='utf-8'))