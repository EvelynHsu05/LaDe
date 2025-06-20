# -*- coding: utf-8 -*-
import geohash2
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic

from utils.util import ws, dir_check, write_list_list, dict_merge, multi_thread_work

# 基礎輔助函數
# 獲取 DataFrame 中指定列名的索引位置
def idx(df, col_name):
    _idx_ = list(df.columns).index(col_name)
    return _idx_

# 將時間字符串轉換為兩個數值：日期編碼和一天中的分鐘數
def time2min(t):
    """
    time string -> date, minute of a day
    :example input: 03-29 17:45:49
    :example output: 329, 1065.8
    """
    M, d = t.split(' ')[0].split('-')
    h, m, s = t.split(' ')[1].split(':')
    return int(f'{M}{d}'), 60 * int(h) + int(m) + int(s) / 60

# 將資料按快遞員ID和日期分組，每個組形成一條"軌跡"
def split_trajectory(df):
    """
    split the dataframe into trajectories of different couriers
    """
    courier_l = []
    temp = df.values[0]
    c_idx=idx(df, 'courier_id')
    ds_idx =  idx(df, 'ds')
    f = 0
    t = 0
    for row in df.values: #df.values: data in the format of numpy arrary
        if row[c_idx] != temp[c_idx] or row[ds_idx] != temp[ds_idx]:
            courier_l.append(df[f:t]) # pick out the data when enters a new courier id or a new date
            f = t
        t = t + 1
        temp = row
    courier_l.append(df[f:t])
    return courier_l

# 檢查連續的速度異常點，標記需要保留的數據點
def check_adjacent_speed(speed_index):
    keep_index = []
    for i in range(len(speed_index) - 1):
        if (speed_index[i] == 1) and (speed_index[i+1] == 1):
            keep_index.extend([False])
        else:
            keep_index.extend([True])
    keep_index.extend([True])
    return keep_index

# 多線程處理GPS異常點
def drop_abnormal_gps_kernel(args: dict):
    c_lst = args['c_lst']
    result = {}
    pbar = tqdm(total=len(c_lst))
    for c in c_lst:
        pbar.update(1)
        c = c.reset_index()
        c['speed_adjacent'] = c.apply(lambda r: r['dis_to_last_package'] / r['time_to_last_package'] if r['time_to_last_package'] !=0 else 0, axis=1)
        c['keep'] = check_adjacent_speed(((c['speed_adjacent'] > 500).values + 0).tolist())
        for o_id, is_keep in zip(c['order_id'], c['keep']):
            result[o_id] = is_keep
    return result

# 將非連續的ID或類別映射為連續的數字索引（從0開始）
def reindex(dic):
    idx = 0
    map_dic = {}
    for k in dic.keys():
        map_dic[k] = idx
        idx += 1
    return map_dic

# 過濾異常數據和異常快遞員
def drop_unnormal(df, fout):

    # delete abnormal gps drift points

    courier_l = split_trajectory(df)
    # thread_num：線程數量，表示要使用多少個並行線程來處理任務
    thread_num = 20
    n = len(courier_l)
    # task_num：每個線程要處理的任務數量
    task_num = n // thread_num
    # 將所有任務分割成task_num長度的字典
    args_lst = [{'c_lst': courier_l[i: min(i + task_num, n)]} for i in range(0, n, task_num)]
    # 
    results = multi_thread_work(args_lst, drop_abnormal_gps_kernel, thread_num)
    result_dict = dict_merge(results)
    df['keep'] = df['order_id'].apply(lambda x: result_dict[x])
    df = df[df['keep']==True]

    # get couriers features
    print('Begin to construct courier feature')
    couriers, couriers_feature = courier_info(df)
    print('courier feature constructed')

    # filter couriers
    rmv_wd = set(filter(lambda c: couriers_feature['work_days'][c] < 5, couriers))       # 檢查快遞員work days < 5，符合回傳True，不符合（工作時長大於5天）回傳False，並將符合的快遞員ID轉換成set
    rmv_dmd = set(filter(lambda c: couriers_feature['dis_avg_day'][c] < 50, couriers))   # average travel distance per day < 50m
    rmv_omd = set(filter(lambda c: couriers_feature['order_avg_day'][c] < 5, couriers))  # average package number < 5
    rmv_tmo = set(filter(lambda c: couriers_feature['time_avg_order'][c] < 5, couriers)) # average got time difference between two consecutive packages in the trajectory< 5 min
    rmv_dmo = set(filter(lambda c: couriers_feature['dis_avg_order'][c] < 20, couriers)) # average distance between two consecutive packages in the trajectory < 20m
    remove_c = rmv_wd & rmv_dmd & rmv_omd & rmv_tmo & rmv_dmo


    # filter courier data
    print('Begin to filter data')
    keep = []
    similar_cnt, remove_cnt = 0, 0
    courier_l = split_trajectory(df)
    pbar = tqdm(total=len(courier_l))
    for c in courier_l:
        pbar.update(1)
        c_v =  c.reset_index()
        for n, row in c_v.iterrows():
            if row['courier_id'] in remove_c:
                keep.append(False)
                remove_cnt += 1
                continue
            if n != 0 and row['dis_to_last_package'] == 0 and row['time_to_last_package'] < 1:  ##consider as one order within one minute
                keep.append(False)
                similar_cnt += 1
                continue
            keep.append(True)
    print('Filter data finish')
    df = df[keep]


    str_ = f" filter couriers: couriers/total: {len(remove_c)}/{len(couriers)}, filter number of orders {remove_cnt}, delete redundant order:  {similar_cnt}."
    print(str_)
    write_list_list(fout +'/data_info.txt', [[str_]], 'w')

    report_df = pd.DataFrame({'filter condition': ['work days < 5', 'average travel distance per day < 50m', 'average package number < 3',
                                                   'average got time difference between two consecutive packages in the trajectory< 5 min',
                                                   'average distance between two two consecutive packages in the trajectory < 20m'],
                              'number':[ len(x & remove_c) for x in [rmv_wd, rmv_dmd, rmv_omd, rmv_tmo, rmv_dmo]],
                              'ratio': [  len(x & remove_c) / len(remove_c) if len(remove_c) != 0 else 0 for x in [rmv_wd, rmv_dmd, rmv_omd, rmv_tmo, rmv_dmo]]
                              })
    print('Report of filtered couriers：\n', report_df )

    return df, (couriers, couriers_feature)

# 將列表轉換成字符
def list2str(l):
    # list to str
    return '.'.join(map(str, l))

# 將字符轉換成列表
def str2list(s):
    # str to list
    return [] if s == '' else list(map(int, s.split('.')))

# 多線程計算每個時間點的未完成任務
def get_todo_kernel(args: dict):
    result = {}
    def get_a_todo(x):
        # 獲取當前訂單完成時間，視為當下時間，將其他接受時間小於當下時間（已接受）且完成時間大於當下時間（還未送達）的訂單列出，為待辦任務清單
        now_time = x['finish_time_minute']
        accepted = c_v[c_v['accept_time_minute'] < now_time]  # accepted orders at current time
        todo = accepted[accepted['finish_time_minute'] > now_time]  # unfinished order at current time
        todo_lst = list(todo['index'])

        o_id = x['index']
        result[(o_id, 'todo_task')] = list2str(todo_lst)
        result[(o_id, 'todo_task_num')] = len(todo_lst)
        return x

    c_lst = args['c_lst']
    pbar = tqdm(total=len(c_lst))
    for c in c_lst:
        pbar.update(1)
        c_v = c.reset_index()
        c_v = c_v.apply(lambda x: get_a_todo(x), axis=1)
    return result

# 計算每個快遞員的統計特徵
def courier_info(df):
    """
    get courier's feature
    """
    couriers=list(set(df['courier_id']))
    feature_dict = {} # init the feature dict
    for key in ['index', 'id', 'order_sum', 'dis_sum', 'work_days', 'order_avg_day', 'dis_avg_day', 'time_avg_order', 'dis_avg_order', 'speed_avg_order']:
        feature_dict[key] = {}
    # index: index in the courier list, may be used for embedding table
    # order_sum: total order number of a courier in the dataset
    # dis_sum: total travel distance of a courier
    # work_days: work days of a courier
    # order_avg_day: average order number of a courier each day
    # dis_avg_day: average distance per day of a courier
    # time_avg_order: average delta t (in minute) between two consecutive packages of a courier
    # dis_avg_order: average distance (in minute) between two consecutive packages of a courier
    # speed_avg_order: average speed of a courier
    pbar = tqdm(total=len(couriers))
    for idx, c in enumerate(couriers):
        pbar.update(1)
        c_df = df[df['courier_id'] == c]
        feature_dict['index'][c] = idx
        feature_dict['id'][c] =  c
        feature_dict['order_sum'][c] = c_df.shape[0]
        feature_dict['dis_sum'][c] = sum(c_df['dis_to_last_package'])
        
        # 每筆訂單都有一個日期，但同一天可能有多筆訂單，我們想知道這個快遞員總共工作了幾天，
所以要去重複後計算天數
        # 比起work_days，更應該命名為活躍天數
        feature_dict['work_days'][c]=len(set(c_df['ds']))

        feature_dict['order_avg_day'][c]=feature_dict['order_sum'][c]/feature_dict['work_days'][c]
        feature_dict['dis_avg_day'][c]=feature_dict['dis_sum'][c]/feature_dict['work_days'][c]
        feature_dict['time_avg_order'][c]=np.mean(c_df['time_to_last_package'])
        feature_dict['dis_avg_order'][c] = np.mean(c_df['dis_to_last_package'])

        # 如果時間為０，預設速度為５
        feature_dict['speed_avg_order'][c] = feature_dict['dis_sum'][c] / (sum(c_df['time_to_last_package'])) if sum(c_df['time_to_last_package']) != 0 else 5
    return couriers, feature_dict

# 多線程處理每條軌跡，計算訂單間的關係特徵
def process_traj_kernel(args ={}):
    # process a trajectory of a courier
    c_lst = args['c_lst']
    pbar = tqdm(total=len(c_lst))
    result = {}
    for c in c_lst:
        pbar.update(1)
        c_v = c.reset_index()
        for n, row in c_v.iterrows():
            date_gt, gt = time2min(row['pickup_time'])  # pickup_time in minute
            date_at, at = time2min(row['accept_time'])  # accept_time in minute
            if date_gt != date_at:  # got_date != accept_date, means that the order is placed a day another day
                at = at - 60 * 24

            if (str(row['time_window_start'])) == 'nan':
                et = 1440
            else:
                _, et = time2min(row['time_window_end'])  # expect_got_time in minute

            # information of last package
            last_idx = max(0, n - 1)  # last_idx = 0 if n==0 else n-1
            last_ft = c_v.iloc[last_idx]['finish_time_minute']
            last_lon_ = c_v.iloc[last_idx]['lng']
            last_lat_ = c_v.iloc[last_idx]['lat']

            o_id =  row['order_id']
            result[(o_id, 'accept_time_minute')] = at
            result[(o_id, 'expect_finish_time_minute')] = et
            result[(o_id, 'time_to_last_package')] = row['finish_time_minute']- last_ft
            result[(o_id, 'dis_to_last_package')] = int(geodesic((last_lat_, last_lon_), (row['lat'], row['lng'])).meters)

    return result


name_dict = {
        'from_dipan_id': 'region_id',
        'from_city_name': 'city',
        'delivery_user_id': 'courier_id',
        'poi_lng': 'lng',
        'poi_lat': 'lat',
        'typecode': 'aoi_type',
        'got_time': 'pickup_time', # got_time,got_gps_time,got_gps_lng,got_gps_lat
        'got_gps_time': 'pickup_gps_time',
        'got_gps_lng': 'pickup_gps_lng',
        'got_gps_lat': 'pickup_gps_lat',
        'book_start_time': 'time_window_start',
        'expect_got_time': 'time_window_end'
    }

# 建立AOI（Area of Interest）字典和特徵
def make_aoi_dict(fin_temp):
    print('start make aoi dict')
    df = pd.read_csv(fin_temp + "/package_feature.csv", sep=',', encoding='utf-8')  # package features
    aois = df['aoi_id'].value_counts().to_dict()
    aoi_dict = reindex(aois)
    df['aoi_id'] = df['aoi_id'].apply(lambda x: aoi_dict[x])

    courier_l = split_trajectory(df)
    
    aoi_nums = df["aoi_id"].nunique()
    aoi_frequency_adj = np.zeros([aoi_nums, aoi_nums])  # AOI轉換頻率矩陣
    aoi_time_adj = np.zeros([aoi_nums, aoi_nums])       # AOI轉換時間矩陣
    aoi_order_num = np.zeros([aoi_nums])                # 每個AOI的訂單數
    aoi_type = np.zeros([aoi_nums])                     # AOI類型
    aoi_actime = np.zeros([aoi_nums])                   # AOI累計時間
    aoi_lng = np.zeros([aoi_nums])                      # AOI累計經度
    aoi_lat = np.zeros([aoi_nums])                      # AOI累計緯度

    for i in range(len(courier_l)):
        pbar.update(1)
        for j in range(len(courier_l[i])):
            aoi = courier_l[i].iloc[j]["aoi_id"]
            aoi_order_num[aoi] += 1                    # 訂單數+1
            aoi_type[aoi] = courier_l[i].iloc[j]["aoi_type"]  # 記錄AOI類型
            aoi_actime[aoi] += courier_l[i].iloc[j]["time_to_last_package"]  # 累計時間
            aoi_lng[aoi] += courier_l[i].iloc[j]["lng"]  # 累計經度
            aoi_lat[aoi] += courier_l[i].iloc[j]["lat"]  # 累計緯度
            if j != 0:  # 不是第一個訂單
                from_aoi = courier_l[i].iloc[j - 1]["aoi_id"]  # 前一個AOI
                to_aoi = courier_l[i].iloc[j]["aoi_id"]        # 當前AOI
                aoi_frequency_adj[from_aoi][to_aoi] += 1       # 轉換頻率+1
                aoi_time_adj[from_aoi][to_aoi] = aoi_time_adj[from_aoi][to_aoi] + courier_l[i].iloc[j]["time_to_last_package"]
    aoi_frequency_adj[aoi_frequency_adj == 0] = 1  # 避免除以0
    aoi_time_adj = np.divide(aoi_time_adj, aoi_frequency_adj)  # 平均轉換時間
    aoi_actime = np.divide(aoi_actime, aoi_order_num)          # 平均處理時間
    aoi_lng = np.divide(aoi_lng, aoi_order_num)                # 平均經度
    aoi_lat = np.divide(aoi_lat, aoi_order_num)                # 平均緯度

    aoi_feature = np.zeros([aoi_nums, 5])
    for i in tqdm(range(aoi_nums)):
        aoi_feature[i][0] = i                # AOI索引
        aoi_feature[i][1] = aoi_type[i]      # AOI類型
        aoi_feature[i][2] = aoi_lng[i]       # AOI平均經度
        aoi_feature[i][3] = aoi_lat[i]       # AOI平均緯度
        aoi_feature[i][4] = aoi_actime[i]    # AOI平均處理時間

    data = {
        "aoi_dict": aoi_dict,              # AOI ID對應字典
        "aoi_feature": aoi_feature,        # AOI特徵矩陣
        "aoi_time_adj": aoi_time_adj,      # AOI轉換時間矩陣
        "aoi_frequency_adj": aoi_frequency_adj  # AOI轉換頻率矩陣
    }
    fout = fin_temp + "/aoi_feature.npy"
    np.save(fout, data)

# 整合所有數據處理步驟的主函數
def pre_process(fin, fout, is_test=False, thread_num = 20):
    print('Raw input file:' + fin)
    print('Temporary file:' + fout)
    df = pd.read_csv(fin, sep=',', encoding='utf-8')
    if 'got_time' in list(df.columns):
        df = df.rename(columns=name_dict)

    print('length of the original data: ', len(df))

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # rank data by date, courier, and got time
    print('Expand basic information...')
    df['finish_time_minute'] = df['pickup_time'].apply(lambda t: time2min(t)[1]) # here pick up time is the task finish time
    df['expect_finish_time_minute'] = df['time_window_end'].apply(lambda t: time2min(t)[1])
    df = df.sort_values(by=['ds', 'courier_id', 'finish_time_minute'])
    if is_test:
        df = df[:2000]

    courier_l = split_trajectory(df)

    n = len(courier_l)
    task_num = n // thread_num
    args_lst = [{'c_lst': courier_l[i: min(i+task_num, n)]}   for i in range(0, n, task_num)]
    results = multi_thread_work(args_lst, process_traj_kernel, thread_num)
    result_dict = dict_merge(results)

    init_value = [0] * len(df)
    expand_column = ['accept_time_minute', 'expect_finish_time_minute', 'time_to_last_package', 'dis_to_last_package']
    for col in expand_column:
        df[col] = init_value
        df[col] = df['order_id'].apply(lambda x: result_dict[(x, col)])
    print('Basic information expanded...')

    print('Filter data...')
    ## Remove too short trajectories, duplicate packages/orders, and remove GPS drift points, Filter couriers data
    df, courier_information = drop_unnormal(df, fout)
    couriers, couriers_feature  = courier_information
    print('Filter finished...')

    # insert order index
    df.insert(0, 'index', range(1, df.shape[0] + 1))

    ## get unfinished tasks, (packages)
    print('Get unfinished tasks ...')
    courier_l = split_trajectory(df)
    n =  len(courier_l)
    args_lst = [{'c_lst': courier_l[i: min(i + task_num, n)]} for i in range(0, n, task_num)]
    results = multi_thread_work(args_lst, get_todo_kernel, thread_num)
    result_dict = dict_merge(results)

    init_value = [0] * len(df)
    expand_column = ['todo_task', 'todo_task_num']
    for col in expand_column:
        df[col] = init_value
        df[col] = df['index'].apply(lambda x: result_dict[(x, col)])
    print('Get unfinished tasks done...')

    # courier's information
    df['dis_avg_day'] = [couriers_feature['dis_avg_day'][c] for c in df['courier_id']]
    df['time_avg_order'] = [couriers_feature['time_avg_order'][c] for c in df['courier_id']]

    print('Update information...')
    courier_l = split_trajectory(df)
    n = len(courier_l)
    task_num = n // thread_num
    args_lst = [{'c_lst': courier_l[i: min(i + task_num, n)]} for i in range(0, n, task_num)]
    results = multi_thread_work(args_lst, process_traj_kernel, thread_num)
    result_dict = dict_merge(results)

    init_value = [0] * len(df)
    expand_column = ['time_to_last_package', 'dis_to_last_package']
    for col in expand_column:
        df[col] = init_value
        df[col] = df['order_id'].apply(lambda x: result_dict[(x, col)])

    df['relative_dis_to_last_package'] = df.apply(lambda r: r['dis_to_last_package'] / r['dis_avg_day'] * 100 if r['dis_avg_day'] !=0 else 0, axis=1)

    # 將經緯度轉換為geohash編碼，便於空間分析
    df['geohash_6'] = [geohash2.encode(lat, lon, 6) for lat, lon in zip(df['lat'], df['lng'])]
    df['geohash_3'] = [geohash2.encode(lat, lon, 3) for lat, lon in zip(df['lat'], df['lng'])]
    geo3_dict = {}
    geo3 = df['geohash_3'].unique()
    for i in range(len(geo3)):
        geo3_dict[geo3[i]] = i
    df['geohash_3'] = df['geohash_3'].map(geo3_dict)

    geo6_dict = {}
    geo6 = df['geohash_6'].unique()
    for i in range(len(geo6)):
        geo6_dict[geo6[i]] = i
    df['geohash_6'] = df['geohash_6'].map(geo6_dict)

    print('Features between adjacent tasks constructed...')

    days = sorted(list(set(df['ds'])))
    df['days'] =  [days.index(d) for d in df['ds']]

    ##Generate courier's feature
    cou_df_dic = {}
    for fea, fea_dict in couriers_feature.items():
        fea_lst = [fea_dict[c] for c in couriers]
        cou_df_dic[fea] = fea_lst
    cou_df = pd.DataFrame(cou_df_dic)

    if fout != '':
        dir_check(fout)
        df.to_csv(fout+'package_feature.csv', index=False)
        cou_df.to_csv(fout+'courier_feature.csv', index=False)
        make_aoi_dict(fout)
    print('Data preprocessing is done...')

    return df, cou_df





