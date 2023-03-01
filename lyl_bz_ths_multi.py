from typing import Tuple
import pandas as pd
from random import sample
from tqdm import tqdm
import os
from time import time
from multiprocessing import Pool

# 卡牌数组
CARDS = []
for i in range(4):
    CARDS.extend([(i, j) for j in range(1, 14)])

# 顺子组合
SHUNZI = [{i + 1, i + 2, i + 3} for i in range(11)] + [{12, 13, 1},]

# 模拟次数
NUM = 10 ** 5


def one_round(*args) -> Tuple[pd.Series, pd.Series]:
    """一轮抽卡后 不同数量玩家下的豹子和同花顺统计结果"""
    # 抽卡
    cards_this_round = sample(CARDS, 51)
    cards_split_players = [cards_this_round[i: i + 3] for i in range(0, 51, 3)]
    cards_split_data = [[*i[0], *i[1], *i[2]] for i in cards_split_players]
    df = pd.DataFrame(
        cards_split_data,
        columns=['suit_1', 'num_1', 'suit_2',
                 'num_2', 'suit_3', 'num_3']
    )

    def _tonghuashun(s: pd.Series) -> bool:
        """同花顺判定"""
        if (s['suit_1'] == s['suit_2'] == s['suit_3']) and \
                {s['num_1'], s['num_2'], s['num_3']} in SHUNZI:
            return True
        return False

    # 判定豹子和同花顺
    df.eval('baozi = (num_1 == num_2 == num_3)', inplace=True)
    df['tonghuashun'] = df.apply(_tonghuashun, axis=1)

    # 统计不同人数下的豹子和同花顺结果
    return df['baozi'].cumsum(), df['tonghuashun'].cumsum()


def get_result(num: int) -> pd.DataFrame:
    """获得数轮洗牌后的结果"""
    cup_total = os.cpu_count()
    cpu_used = cup_total - 2 if cup_total > 3 else 1
    print(f'cpu count: {os.cpu_count()}, used: {cpu_used}')
    with Pool(cpu_used) as pool:
        # 多核并用 洗牌并记录
        print('shuffling and recoding...')
        l = [i for i in
             tqdm(pool.imap_unordered(one_round, range(num)), total=num)]
        bz_data, ths_data = zip(*l)

    # 统计最终结果
    df_bz_data: pd.DataFrame = pd.concat(bz_data, axis=1)
    df_ths_data: pd.DataFrame = pd.concat(ths_data, axis=1)

    stat = pd.concat(
        [
            df_bz_data.sum(axis=1).rename('bz_count'),
            df_ths_data.sum(axis=1).rename('ths_count')
        ],
        axis=1
    )

    stat.eval(
        '''
        bz_percentage = bz_count / @num
        ths_percentage = ths_count / @num
        ''',
        inplace=True
    )
    return stat


if __name__ == '__main__':
    print('李永乐老师的豹子、同花顺模拟')
    print('原视频地址 https://www.ixigua.com/7201798361501205048')
    try:
        x = int(eval(input('请输入模拟次数: ')))
    except ValueError:
        print('输入的不是数字 将使用默认次数 {NUM}')
        x = NUM
    else:
        print(f'模拟次数为: {x}')

    start_time = time()
    stat = get_result(x)
    file_name = f'stat_m_{x}.csv'
    stat.to_csv(file_name, index=False)
    print(f'模拟结果的文件名为: {file_name}')
    print(f'耗时总计: {time() - start_time}秒')
