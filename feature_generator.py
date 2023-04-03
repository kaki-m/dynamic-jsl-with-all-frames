'''
2023/2/23
特徴量生成コード
指文字を表現するデータを時間軸で4分割し、それぞれに対する特徴量を生成する
'''
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
from natsort import natsorted
import copy

STATIC_FILE_USE_NUM = 12  # 静的指文字の特徴量生成に使用するフレーム数
START_INDEX = 20  # 静的指文字の特徴量生成に使用するフレームの開始位置

Name = input("特徴抽出するディレクトリ名を入力: ")
srcdir = '../../kakizaki/data/' + str(Name) + '/preprocessed_coordinates_m/'
wrtdir = './features/features1/' + str(Name) + '/'

try:
    os.makedirs(wrtdir)
except FileExistsError:
    print('ディレクトリ(' + str(wrtdir) + ')はすでに存在しているため作成しません')
    pass

file = os.listdir(srcdir)  # srcdirのファイル名をリストですべて持って来る
srcFiles = natsorted(file)  # ファイル名順にソート(この場合数字でソートされるのでlabel順、フレーム順になる)

'''
ここからデータをそれぞれのhand_typeごとのリストにする処理
データ格納先を宣言 -> ファイル名で分けてそれぞれのhand_typeに分ける
'''
hand_data_list = [[] for i in range(47)]

for f in srcFiles:
    hand_type = int(f.split('_')[1])
    hand_data_list[hand_type].append(f)

'''
hand_dataごとに分けてたリストhand_data_listのデータ数を調べることで、
最もフレーム数の少ないhand_typeのフレーム数を取得する
これらは時間軸データ分割の4分割に耐えられるかを確認するため
'''
minDataNum = 1000
minDataHandType = 0
for i in range(1,47):
    if len(hand_data_list[i]) < minDataNum:
        minDataNum = len(hand_data_list[i])
        minDataHandType = i

# もし最小のフレーム数が4を下回る場合にはその旨を通知し、処理を続行するかどうか決める
if minDataNum < 4:
    user_input = input("データ数が4を下回り分割が行えません。処理を続けますか? y/n : ")
    if user_input == 'y':
        print("処理を続行します")
    else:
        print("データ数が十分にないため特徴量生成を中止します")
        exit(1)


'''
静的指文字のフレーム数を動的指文字のフレーム数に合わせる処理を行う
静的指文字の特徴抽出に使用するフレーム数は動的指文字の平均フレーム数から算出され、
コード上部のSTATIC_FILE_USE_NUMに定義される
使用フレームはだいたい真ん中にしたいが、そのスタート位置はコード上部START_INDEXに定義する
'''
for i in range(1,42):  # 静的指文字のファイル名を12個に減らす
    new_file_list = []
    if len(hand_data_list[i]) <= 12:
        j = 0
    else:
        j = START_INDEX  # カウンタ変数
    while len(new_file_list) < STATIC_FILE_USE_NUM:  # 規定ファイル数を確保するまで
        try:
            new_file_list.append(hand_data_list[i][j])
        except:
            print("静的指文字のファイル数が足りませんでした")
            print("hand_type: " + str(i))
            print("ファイル箇所: " + str(j))
            #exit(1)
            #ここで強制終了せずに12以下のファイル数でも実行する
            break
        j = j + 1
    # new_file_listに置き換える
    hand_data_list[i] = copy.deepcopy(new_file_list)

'''
ここから実際に特徴抽出を行っていく
'''
# まずはファイル名リストhand_data_listを使って実際のデータをnumpy配列に取ってくる

for i in tqdm(range(1,len(hand_data_list))):
    frame_num = len(hand_data_list[i])
    split_num = frame_num / 4  # データを時間軸で4分割するため、何個ずつにするか
    raw_data_list = []
    for j in range(len(hand_data_list[i])):
        # データ読み込み
        data = np.loadtxt(srcdir + hand_data_list[i][j], delimiter=',')
        raw_data_list.append(data)
    
    # print(raw_data_list)
    # ここまででraw_data_listに座標データ、frame_numにそのhand_typeのフレーム数が入っている
    start_indexes = [0]
    for j in range(3):
        if j == 3:
            start_indexes.append(int(start_indexes[j] + split_num))
        else:
            start_indexes.append(math.ceil(start_indexes[j] + split_num))
    
    start_indexes.append(frame_num)
    # print("start_indexes: " + str(start_indexes))
    # start_indexesに入っている数字から次の数字までの間で一つの特徴量を生成する
    '''
    Distance: average
    Angle: average
    Variation: 最初のフレームと最後のフレームの移動ベクトル
    Direction: average, 各指が向いている方向を方向ベクトルで示す
    '''
    all_columns = []
    all_features = np.array([], dtype = 'float')
    distance_average = np.array([], dtype = 'float')
    angle_average = np.array([], dtype = 'float')
    variation = np.array([], dtype = 'float')
    thumb_direction_averages = np.array([], dtype = 'float')
    index_finger_direction_averages = np.array([], dtype = 'float')
    middle_finger_direction_averages = np.array([], dtype = 'float')
    ring_finger_direction_averages = np.array([], dtype = 'float')
    pinkie_finger_direction_averages = np.array([], dtype = 'float')

    # 各区切りで得られた特徴量を一時的に保存する変数
    distance_average_tmp = np.array([0 for j in range(190)], dtype = 'float')
    angle_average_tmp = np.array([0 for j in range(630)], dtype = 'float')

    used_frame_counter = 0
    # csvでculumnの説明として保存する部分を保存
    mark1=[]  # 距離
    mark2=[]  # 角度
    mark3=[]  # variation
    mark4=[]  # direction

    thumb_direction = np.array([0,0,0], dtype = 'float')
    index_finger_direction = np.array([0,0,0], dtype = 'float')
    middle_finger_direction = np.array([0,0,0], dtype = 'float')
    ring_finger_direction = np.array([0,0,0], dtype = 'float')
    pinkie_finger_direction = np.array([0,0,0], dtype = 'float')
    # raw_data_list[0 -> 1/4]
    for j in range(start_indexes[0], start_indexes[1]-1):
        '''
        distanceの生成
        '''
        used_frame_counter += 1 #平均を出すためにいくつフレームがあったのかを記録
        distance_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    #もし最初の一回目のdistance生成ならmarkを作る
                    mark1.append("1of4_distance" + str(k) +":"+ str(l))
                # 3次元の点と点同士の距離をdistance_average[1-2などに入れる]
                distance_average_tmp[distance_index] += math.sqrt((raw_data_list[j][k][0]-raw_data_list[j][l][0])**2 + (raw_data_list[j][k][1]-raw_data_list[j][l][1])**2 + (raw_data_list[j][k][2]-raw_data_list[j][l][2])**2)
                distance_index += 1

        '''
        angleの生成
        '''
        angle_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # x角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # y角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # z角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # 最初の1フレームならmark2を作る
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    mark2.append("1of4_angle_x" + str(k) +":"+ str(l))
                    mark2.append("1of4_angle_y" + str(k) +":"+ str(l))
                    mark2.append("1of4_angle_z" + str(k) +":"+ str(l))
        
        '''
        directionの生成
        averageを求める。
        三次元単位ベクトル
        '''
        thumb_direction_tmp = np.array([0,0,0], dtype = 'float')
        index_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        middle_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        ring_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        pinkie_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        # 親指 1 -> 4
        # 指先 - 指元でベクトルを計算し、その後で単位ベクトルに変換
        thumb_direction_tmp[0] = raw_data_list[j][4][0] - raw_data_list[j][1][0]
        thumb_direction_tmp[1] = raw_data_list[j][4][1] - raw_data_list[j][1][1]
        thumb_direction_tmp[2] = raw_data_list[j][4][2] - raw_data_list[j][1][2]
        thumb_vector_size = math.sqrt((thumb_direction_tmp[0] ** 2) + (thumb_direction_tmp[1] ** 2) + (thumb_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        thumb_direction_tmp[0] /= thumb_vector_size
        thumb_direction_tmp[1] /= thumb_vector_size
        thumb_direction_tmp[2] /= thumb_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        thumb_direction += thumb_direction_tmp

        # 人差し指 5 -> 8
        index_finger_direction_tmp[0] = raw_data_list[j][8][0] - raw_data_list[j][5][0]
        index_finger_direction_tmp[1] = raw_data_list[j][8][1] - raw_data_list[j][5][1]
        index_finger_direction_tmp[2] = raw_data_list[j][8][2] - raw_data_list[j][5][2]
        index_finger_vector_size = math.sqrt((index_finger_direction_tmp[0] ** 2) + (index_finger_direction_tmp[1] ** 2) + (index_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        index_finger_direction_tmp[0] /= index_finger_vector_size
        index_finger_direction_tmp[1] /= index_finger_vector_size
        index_finger_direction_tmp[2] /= index_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        index_finger_direction += index_finger_direction_tmp

        # 中指 9 -> 12
        middle_finger_direction_tmp[0] = raw_data_list[j][12][0] - raw_data_list[j][9][0]
        middle_finger_direction_tmp[1] = raw_data_list[j][12][1] - raw_data_list[j][9][1]
        middle_finger_direction_tmp[2] = raw_data_list[j][12][2] - raw_data_list[j][9][2]
        middle_finger_vector_size = math.sqrt((middle_finger_direction_tmp[0] ** 2) + (middle_finger_direction_tmp[1] ** 2) + (middle_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        middle_finger_direction_tmp[0] /= middle_finger_vector_size
        middle_finger_direction_tmp[1] /= middle_finger_vector_size
        middle_finger_direction_tmp[2] /= middle_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        middle_finger_direction += middle_finger_direction_tmp

        # 薬指　13 -> 16
        ring_finger_direction_tmp[0] = raw_data_list[j][16][0] - raw_data_list[j][13][0]
        ring_finger_direction_tmp[1] = raw_data_list[j][16][1] - raw_data_list[j][13][1]
        ring_finger_direction_tmp[2] = raw_data_list[j][16][2] - raw_data_list[j][13][2]
        ring_finger_vector_size = math.sqrt((ring_finger_direction_tmp[0] ** 2) + (ring_finger_direction_tmp[1] ** 2) + (ring_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        ring_finger_direction_tmp[0] /= ring_finger_vector_size
        ring_finger_direction_tmp[1] /= ring_finger_vector_size
        ring_finger_direction_tmp[2] /= ring_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        ring_finger_direction += ring_finger_direction_tmp

        # 小指 17 -> 20
        pinkie_finger_direction_tmp[0] = raw_data_list[j][20][0] - raw_data_list[j][17][0]
        pinkie_finger_direction_tmp[1] = raw_data_list[j][20][1] - raw_data_list[j][17][1]
        pinkie_finger_direction_tmp[2] = raw_data_list[j][20][2] - raw_data_list[j][17][2]
        pinkie_finger_vector_size = math.sqrt((pinkie_finger_direction_tmp[0] ** 2) + (pinkie_finger_direction_tmp[1] ** 2) + (pinkie_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        pinkie_finger_direction_tmp[0] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[1] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[2] /= pinkie_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        pinkie_finger_direction += pinkie_finger_direction_tmp
    
    '''
    variationの生成
    区切った中での最後のフレーム - 最初のフレームで変化したベクトルを21個のlandmarkごとに生成する
    '''
    variation_tmp = []
    for j in range(21):
        mark3.append("1of4_variationX_" + str(j+1) )
        try:
            variation_tmp.append(raw_data_list[start_indexes[1]-1][j][0] - raw_data_list[start_indexes[0]][j][0])  # x座標の最初のフレームと最後のフレームの差
        except:
            print(raw_data_list)
        mark3.append("1of4_variationY_" + str(j+1) )
        variation_tmp.append(raw_data_list[start_indexes[1]-1][j][1] - raw_data_list[start_indexes[0]][j][1])  # y座標
        mark3.append("1of4_variationZ_" + str(j+1) )
        variation_tmp.append(raw_data_list[start_indexes[1]-1][j][2] - raw_data_list[start_indexes[0]][j][2])  # z座標
    
    if used_frame_counter == 0:
        print("0で割るパターンが実行")
        print("hand_type: " + str(i))
        print("frame_num: " + str(frame_num))
        print("split_num: " + str(split_num))
        print(start_indexes)
    #今まで計算した特徴はすべてのフレームの合計なので、使用　フレーム数で割って平均にする
    for j in range(len(distance_average)):
        distance_average_tmp[j] = distance_average_tmp[j] / used_frame_counter
    for j in range(len(angle_average)):
        angle_average_tmp[j] = angle_average_tmp[j] / used_frame_counter
    
    for j in range(3):
        thumb_direction[j] /= used_frame_counter
        index_finger_direction[j] /= used_frame_counter
        middle_finger_direction[j] /= used_frame_counter
        ring_finger_direction[j] /= used_frame_counter
        pinkie_finger_direction[j] /= used_frame_counter
    
    # データ確認
    # confirm_variable_name = variation_tmp
    # print("データ確認")
    # print(len(confirm_variable_name))
    # print(confirm_variable_name)

    '''
    distance_average_tmp: 190個の実数データ
    angle_average:        630個の実数データ
    thumb_direction:      
    '''
    # この分割域で計算した特徴を一つの変数にまとめる
    distance_average = np.append(distance_average, distance_average_tmp)
    angle_average = np.append(angle_average, angle_average_tmp)
    thumb_direction_averages = np.append(thumb_direction_averages, thumb_direction)
    index_finger_direction_averages = np.append(index_finger_direction_averages, index_finger_direction)
    middle_finger_direction_averages = np.append(middle_finger_direction_averages, middle_finger_direction)
    ring_finger_direction_averages = np.append(ring_finger_direction_averages,ring_finger_direction)
    pinkie_finger_direction_averages = np.append(pinkie_finger_direction_averages, pinkie_finger_direction)
    variation = np.append(variation, variation_tmp)

    '''
    2/4
    '''
    # 各区切りで得られた特徴量を一時的に保存する変数
    distance_average_tmp = np.array([0 for j in range(190)], dtype = 'float')
    angle_average_tmp = np.array([0 for j in range(630)], dtype = 'float')
    thumb_direction = np.array([0,0,0], dtype = 'float')
    index_finger_direction = np.array([0,0,0], dtype = 'float')
    middle_finger_direction = np.array([0,0,0], dtype = 'float')
    ring_finger_direction = np.array([0,0,0], dtype = 'float')
    pinkie_finger_direction = np.array([0,0,0], dtype = 'float')
    # raw_data_list[1/4 -> 2/4] (separate 2)
    for j in range(start_indexes[1], start_indexes[2]-1):
        '''
        distanceの生成(2/4)
        '''
        used_frame_counter += 1 #平均を出すためにいくつフレームがあったのかを記録
        distance_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    #もし最初の一回目のdistance生成ならmarkを作る
                    mark1.append("1of4_distance" + str(k) +":"+ str(l))
                # 3次元の点と点同士の距離をdistance_average[1-2などに入れる]
                distance_average_tmp[distance_index] += math.sqrt((raw_data_list[j][k][0]-raw_data_list[j][l][0])**2 + (raw_data_list[j][k][1]-raw_data_list[j][l][1])**2 + (raw_data_list[j][k][2]-raw_data_list[j][l][2])**2)
                distance_index += 1

        '''
        angleの生成
        '''
        angle_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # x角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # y角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # z角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # 最初の1フレームならmark2を作る
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    mark2.append("1of4_angle_x" + str(k) +":"+ str(l))
                    mark2.append("1of4_angle_y" + str(k) +":"+ str(l))
                    mark2.append("1of4_angle_z" + str(k) +":"+ str(l))
        
        '''
        directionの生成
        averageを求める。
        三次元単位ベクトル
        '''
        thumb_direction_tmp = np.array([0,0,0], dtype = 'float')
        index_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        middle_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        ring_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        pinkie_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        # 親指 1 -> 4
        # 指先 - 指元でベクトルを計算し、その後で単位ベクトルに変換
        thumb_direction_tmp[0] = raw_data_list[j][4][0] - raw_data_list[j][1][0]
        thumb_direction_tmp[1] = raw_data_list[j][4][1] - raw_data_list[j][1][1]
        thumb_direction_tmp[2] = raw_data_list[j][4][2] - raw_data_list[j][1][2]
        thumb_vector_size = math.sqrt((thumb_direction_tmp[0] ** 2) + (thumb_direction_tmp[1] ** 2) + (thumb_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        thumb_direction_tmp[0] /= thumb_vector_size
        thumb_direction_tmp[1] /= thumb_vector_size
        thumb_direction_tmp[2] /= thumb_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        thumb_direction += thumb_direction_tmp

        # 人差し指 5 -> 8
        index_finger_direction_tmp[0] = raw_data_list[j][8][0] - raw_data_list[j][5][0]
        index_finger_direction_tmp[1] = raw_data_list[j][8][1] - raw_data_list[j][5][1]
        index_finger_direction_tmp[2] = raw_data_list[j][8][2] - raw_data_list[j][5][2]
        index_finger_vector_size = math.sqrt((index_finger_direction_tmp[0] ** 2) + (index_finger_direction_tmp[1] ** 2) + (index_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        index_finger_direction_tmp[0] /= index_finger_vector_size
        index_finger_direction_tmp[1] /= index_finger_vector_size
        index_finger_direction_tmp[2] /= index_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        index_finger_direction += index_finger_direction_tmp

        # 中指 9 -> 12
        middle_finger_direction_tmp[0] = raw_data_list[j][12][0] - raw_data_list[j][9][0]
        middle_finger_direction_tmp[1] = raw_data_list[j][12][1] - raw_data_list[j][9][1]
        middle_finger_direction_tmp[2] = raw_data_list[j][12][2] - raw_data_list[j][9][2]
        middle_finger_vector_size = math.sqrt((middle_finger_direction_tmp[0] ** 2) + (middle_finger_direction_tmp[1] ** 2) + (middle_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        middle_finger_direction_tmp[0] /= middle_finger_vector_size
        middle_finger_direction_tmp[1] /= middle_finger_vector_size
        middle_finger_direction_tmp[2] /= middle_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        middle_finger_direction += middle_finger_direction_tmp

        # 薬指　13 -> 16
        ring_finger_direction_tmp[0] = raw_data_list[j][16][0] - raw_data_list[j][13][0]
        ring_finger_direction_tmp[1] = raw_data_list[j][16][1] - raw_data_list[j][13][1]
        ring_finger_direction_tmp[2] = raw_data_list[j][16][2] - raw_data_list[j][13][2]
        ring_finger_vector_size = math.sqrt((ring_finger_direction_tmp[0] ** 2) + (ring_finger_direction_tmp[1] ** 2) + (ring_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        ring_finger_direction_tmp[0] /= ring_finger_vector_size
        ring_finger_direction_tmp[1] /= ring_finger_vector_size
        ring_finger_direction_tmp[2] /= ring_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        ring_finger_direction += ring_finger_direction_tmp

        # 小指 17 -> 20
        pinkie_finger_direction_tmp[0] = raw_data_list[j][20][0] - raw_data_list[j][17][0]
        pinkie_finger_direction_tmp[1] = raw_data_list[j][20][1] - raw_data_list[j][17][1]
        pinkie_finger_direction_tmp[2] = raw_data_list[j][20][2] - raw_data_list[j][17][2]
        pinkie_finger_vector_size = math.sqrt((pinkie_finger_direction_tmp[0] ** 2) + (pinkie_finger_direction_tmp[1] ** 2) + (pinkie_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        pinkie_finger_direction_tmp[0] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[1] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[2] /= pinkie_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        pinkie_finger_direction += pinkie_finger_direction_tmp
    
    '''
    variationの生成
    区切った中での最後のフレーム - 最初のフレームで変化したベクトルを21個のlandmarkごとに生成する
    '''
    variation_tmp = []
    for j in range(21):
        mark3.append("2of4_variationX_" + str(j+1) )
        variation_tmp.append(raw_data_list[start_indexes[2]-1][j][0] - raw_data_list[start_indexes[1]][j][0])  # x座標の最初のフレームと最後のフレームの差
        mark3.append("2of4_variationY_" + str(j+1) )
        variation_tmp.append(raw_data_list[start_indexes[2]-1][j][1] - raw_data_list[start_indexes[1]][j][1])  # y座標
        mark3.append("2of4_variationZ_" + str(j+1) )
        variation_tmp.append(raw_data_list[start_indexes[2]-1][j][2] - raw_data_list[start_indexes[1]][j][2])  # z座標
    

    #今まで計算した特徴はすべてのフレームの合計なので、使用　フレーム数で割って平均にする
    for j in range(len(distance_average)):
        distance_average_tmp[j] = distance_average_tmp[j] / used_frame_counter
    for j in range(len(angle_average)):
        angle_average_tmp[j] = angle_average_tmp[j] / used_frame_counter
    
    for j in range(3):
        thumb_direction[j] /= used_frame_counter
        index_finger_direction[j] /= used_frame_counter
        middle_finger_direction[j] /= used_frame_counter
        ring_finger_direction[j] /= used_frame_counter
        pinkie_finger_direction[j] /= used_frame_counter
    
    # # データ確認
    # confirm_variable_name = variation_tmp
    # print("データ確認")
    # print(len(confirm_variable_name))
    # print(confirm_variable_name)

    '''
    distance_average_tmp: 190個の実数データ
    angle_average:        630個の実数データ
    thumb_direction:      
    '''
    # この分割域で計算した特徴を一つの変数にまとめる
    distance_average = np.append(distance_average, distance_average_tmp)
    angle_average = np.append(angle_average, angle_average_tmp)
    thumb_direction_averages = np.append(thumb_direction_averages, thumb_direction)
    index_finger_direction_averages = np.append(index_finger_direction_averages, index_finger_direction)
    middle_finger_direction_averages = np.append(middle_finger_direction_averages, middle_finger_direction)
    ring_finger_direction_averages = np.append(ring_finger_direction_averages,ring_finger_direction)
    pinkie_finger_direction_averages = np.append(pinkie_finger_direction_averages, pinkie_finger_direction)
    variation = np.append(variation, variation_tmp)

    '''
    3/4
    '''
    # 各区切りで得られた特徴量を一時的に保存する変数
    distance_average_tmp = np.array([0 for j in range(190)], dtype = 'float')
    angle_average_tmp = np.array([0 for j in range(630)], dtype = 'float')
    thumb_direction = np.array([0,0,0], dtype = 'float')
    index_finger_direction = np.array([0,0,0], dtype = 'float')
    middle_finger_direction = np.array([0,0,0], dtype = 'float')
    ring_finger_direction = np.array([0,0,0], dtype = 'float')
    pinkie_finger_direction = np.array([0,0,0], dtype = 'float')
    # raw_data_list[2/4 -> 3/4]
    for j in range(start_indexes[2], start_indexes[3]-1):
        '''
        distanceの生成
        '''
        used_frame_counter += 1 #平均を出すためにいくつフレームがあったのかを記録
        distance_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    #もし最初の一回目のdistance生成ならmarkを作る
                    mark1.append("1of4_distance" + str(k) +":"+ str(l))
                # 3次元の点と点同士の距離をdistance_average[1-2などに入れる]
                distance_average_tmp[distance_index] += math.sqrt((raw_data_list[j][k][0]-raw_data_list[j][l][0])**2 + (raw_data_list[j][k][1]-raw_data_list[j][l][1])**2 + (raw_data_list[j][k][2]-raw_data_list[j][l][2])**2)
                distance_index += 1

        '''
        angleの生成
        '''
        angle_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # x角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # y角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # z角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # 最初の1フレームならmark2を作る
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    mark2.append("1of4_angle_x" + str(k) +":"+ str(l))
                    mark2.append("1of4_angle_y" + str(k) +":"+ str(l))
                    mark2.append("1of4_angle_z" + str(k) +":"+ str(l))
        
        '''
        directionの生成
        averageを求める。
        三次元単位ベクトル
        '''
        thumb_direction_tmp = np.array([0,0,0], dtype = 'float')
        index_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        middle_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        ring_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        pinkie_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        # 親指 1 -> 4
        # 指先 - 指元でベクトルを計算し、その後で単位ベクトルに変換
        thumb_direction_tmp[0] = raw_data_list[j][4][0] - raw_data_list[j][1][0]
        thumb_direction_tmp[1] = raw_data_list[j][4][1] - raw_data_list[j][1][1]
        thumb_direction_tmp[2] = raw_data_list[j][4][2] - raw_data_list[j][1][2]
        thumb_vector_size = math.sqrt((thumb_direction_tmp[0] ** 2) + (thumb_direction_tmp[1] ** 2) + (thumb_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        thumb_direction_tmp[0] /= thumb_vector_size
        thumb_direction_tmp[1] /= thumb_vector_size
        thumb_direction_tmp[2] /= thumb_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        thumb_direction += thumb_direction_tmp

        # 人差し指 5 -> 8
        index_finger_direction_tmp[0] = raw_data_list[j][8][0] - raw_data_list[j][5][0]
        index_finger_direction_tmp[1] = raw_data_list[j][8][1] - raw_data_list[j][5][1]
        index_finger_direction_tmp[2] = raw_data_list[j][8][2] - raw_data_list[j][5][2]
        index_finger_vector_size = math.sqrt((index_finger_direction_tmp[0] ** 2) + (index_finger_direction_tmp[1] ** 2) + (index_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        index_finger_direction_tmp[0] /= index_finger_vector_size
        index_finger_direction_tmp[1] /= index_finger_vector_size
        index_finger_direction_tmp[2] /= index_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        index_finger_direction += index_finger_direction_tmp

        # 中指 9 -> 12
        middle_finger_direction_tmp[0] = raw_data_list[j][12][0] - raw_data_list[j][9][0]
        middle_finger_direction_tmp[1] = raw_data_list[j][12][1] - raw_data_list[j][9][1]
        middle_finger_direction_tmp[2] = raw_data_list[j][12][2] - raw_data_list[j][9][2]
        middle_finger_vector_size = math.sqrt((middle_finger_direction_tmp[0] ** 2) + (middle_finger_direction_tmp[1] ** 2) + (middle_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        middle_finger_direction_tmp[0] /= middle_finger_vector_size
        middle_finger_direction_tmp[1] /= middle_finger_vector_size
        middle_finger_direction_tmp[2] /= middle_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        middle_finger_direction += middle_finger_direction_tmp

        # 薬指　13 -> 16
        ring_finger_direction_tmp[0] = raw_data_list[j][16][0] - raw_data_list[j][13][0]
        ring_finger_direction_tmp[1] = raw_data_list[j][16][1] - raw_data_list[j][13][1]
        ring_finger_direction_tmp[2] = raw_data_list[j][16][2] - raw_data_list[j][13][2]
        ring_finger_vector_size = math.sqrt((ring_finger_direction_tmp[0] ** 2) + (ring_finger_direction_tmp[1] ** 2) + (ring_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        ring_finger_direction_tmp[0] /= ring_finger_vector_size
        ring_finger_direction_tmp[1] /= ring_finger_vector_size
        ring_finger_direction_tmp[2] /= ring_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        ring_finger_direction += ring_finger_direction_tmp

        # 小指 17 -> 20
        pinkie_finger_direction_tmp[0] = raw_data_list[j][20][0] - raw_data_list[j][17][0]
        pinkie_finger_direction_tmp[1] = raw_data_list[j][20][1] - raw_data_list[j][17][1]
        pinkie_finger_direction_tmp[2] = raw_data_list[j][20][2] - raw_data_list[j][17][2]
        pinkie_finger_vector_size = math.sqrt((pinkie_finger_direction_tmp[0] ** 2) + (pinkie_finger_direction_tmp[1] ** 2) + (pinkie_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        pinkie_finger_direction_tmp[0] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[1] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[2] /= pinkie_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        pinkie_finger_direction += pinkie_finger_direction_tmp
    
    '''
    variationの生成
    区切った中での最後のフレーム - 最初のフレームで変化したベクトルを21個のlandmarkごとに生成する
    '''
    variation_tmp = []
    for j in range(21):
        mark3.append("1of4_variationX_" + str(j+1) )
        variation_tmp.append(raw_data_list[start_indexes[3]-1][j][0] - raw_data_list[start_indexes[2]][j][0])  # x座標の最初のフレームと最後のフレームの差
        mark3.append("1of4_variationY_" + str(j+1) )
        variation_tmp.append(raw_data_list[start_indexes[3]-1][j][1] - raw_data_list[start_indexes[2]][j][1])  # y座標
        mark3.append("1of4_variationZ_" + str(j+1) )
        variation_tmp.append(raw_data_list[start_indexes[3]-1][j][2] - raw_data_list[start_indexes[2]][j][2])  # z座標
    

    #今まで計算した特徴はすべてのフレームの合計なので、使用　フレーム数で割って平均にする
    for j in range(len(distance_average_tmp)):
        distance_average_tmp[j] = distance_average_tmp[j] / used_frame_counter
    for j in range(len(angle_average_tmp)):
        angle_average_tmp[j] = angle_average_tmp[j] / used_frame_counter
    
    for j in range(3):
        thumb_direction[j] /= used_frame_counter
        index_finger_direction[j] /= used_frame_counter
        middle_finger_direction[j] /= used_frame_counter
        ring_finger_direction[j] /= used_frame_counter
        pinkie_finger_direction[j] /= used_frame_counter
    
    # # データ確認
    # confirm_variable_name = variation_tmp
    # print("データ確認")
    # print(len(confirm_variable_name))
    # print(confirm_variable_name)

    '''
    distance_average_tmp: 190個の実数データ
    angle_average:        630個の実数データ
    thumb_direction:      
    '''
    # この分割域で計算した特徴を一つの変数にまとめる
    distance_average = np.append(distance_average, distance_average_tmp)
    angle_average = np.append(angle_average, angle_average_tmp)
    thumb_direction_averages = np.append(thumb_direction_averages, thumb_direction)
    index_finger_direction_averages = np.append(index_finger_direction_averages, index_finger_direction)
    middle_finger_direction_averages = np.append(middle_finger_direction_averages, middle_finger_direction)
    ring_finger_direction_averages = np.append(ring_finger_direction_averages,ring_finger_direction)
    pinkie_finger_direction_averages = np.append(pinkie_finger_direction_averages, pinkie_finger_direction)
    variation = np.append(variation, variation_tmp)
    


    '''
    4/4
    '''
    # 各区切りで得られた特徴量を一時的に保存する変数
    distance_average_tmp = np.array([0 for j in range(190)], dtype = 'float')
    angle_average_tmp = np.array([0 for j in range(630)], dtype = 'float')
    thumb_direction = np.array([0,0,0], dtype = 'float')
    index_finger_direction = np.array([0,0,0], dtype = 'float')
    middle_finger_direction = np.array([0,0,0], dtype = 'float')
    ring_finger_direction = np.array([0,0,0], dtype = 'float')
    pinkie_finger_direction = np.array([0,0,0], dtype = 'float')
    # raw_data_list[3/4 -> 4/4]
    for j in range(start_indexes[3], start_indexes[4]-1):
        '''
        distanceの生成
        '''
        used_frame_counter += 1 #平均を出すためにいくつフレームがあったのかを記録
        distance_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    #もし最初の一回目のdistance生成ならmarkを作る
                    mark1.append("1of4_distance" + str(k) +":"+ str(l))
                # 3次元の点と点同士の距離をdistance_average[1-2などに入れる]
                distance_average_tmp[distance_index] += math.sqrt((raw_data_list[j][k][0]-raw_data_list[j][l][0])**2 + (raw_data_list[j][k][1]-raw_data_list[j][l][1])**2 + (raw_data_list[j][k][2]-raw_data_list[j][l][2])**2)
                distance_index += 1

        '''
        angleの生成
        '''
        angle_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # x角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # y角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # z角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # 最初の1フレームならmark2を作る
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    mark2.append("1of4_angle_x" + str(k) +":"+ str(l))
                    mark2.append("1of4_angle_y" + str(k) +":"+ str(l))
                    mark2.append("1of4_angle_z" + str(k) +":"+ str(l))
        
        '''
        directionの生成
        averageを求める。
        三次元単位ベクトル
        '''
        thumb_direction_tmp = np.array([0,0,0], dtype = 'float')
        index_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        middle_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        ring_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        pinkie_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        # 親指 1 -> 4
        # 指先 - 指元でベクトルを計算し、その後で単位ベクトルに変換
        thumb_direction_tmp[0] = raw_data_list[j][4][0] - raw_data_list[j][1][0]
        thumb_direction_tmp[1] = raw_data_list[j][4][1] - raw_data_list[j][1][1]
        thumb_direction_tmp[2] = raw_data_list[j][4][2] - raw_data_list[j][1][2]
        thumb_vector_size = math.sqrt((thumb_direction_tmp[0] ** 2) + (thumb_direction_tmp[1] ** 2) + (thumb_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        thumb_direction_tmp[0] /= thumb_vector_size
        thumb_direction_tmp[1] /= thumb_vector_size
        thumb_direction_tmp[2] /= thumb_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        thumb_direction += thumb_direction_tmp

        # 人差し指 5 -> 8
        index_finger_direction_tmp[0] = raw_data_list[j][8][0] - raw_data_list[j][5][0]
        index_finger_direction_tmp[1] = raw_data_list[j][8][1] - raw_data_list[j][5][1]
        index_finger_direction_tmp[2] = raw_data_list[j][8][2] - raw_data_list[j][5][2]
        index_finger_vector_size = math.sqrt((index_finger_direction_tmp[0] ** 2) + (index_finger_direction_tmp[1] ** 2) + (index_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        index_finger_direction_tmp[0] /= index_finger_vector_size
        index_finger_direction_tmp[1] /= index_finger_vector_size
        index_finger_direction_tmp[2] /= index_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        index_finger_direction += index_finger_direction_tmp

        # 中指 9 -> 12
        middle_finger_direction_tmp[0] = raw_data_list[j][12][0] - raw_data_list[j][9][0]
        middle_finger_direction_tmp[1] = raw_data_list[j][12][1] - raw_data_list[j][9][1]
        middle_finger_direction_tmp[2] = raw_data_list[j][12][2] - raw_data_list[j][9][2]
        middle_finger_vector_size = math.sqrt((middle_finger_direction_tmp[0] ** 2) + (middle_finger_direction_tmp[1] ** 2) + (middle_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        middle_finger_direction_tmp[0] /= middle_finger_vector_size
        middle_finger_direction_tmp[1] /= middle_finger_vector_size
        middle_finger_direction_tmp[2] /= middle_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        middle_finger_direction += middle_finger_direction_tmp

        # 薬指　13 -> 16
        ring_finger_direction_tmp[0] = raw_data_list[j][16][0] - raw_data_list[j][13][0]
        ring_finger_direction_tmp[1] = raw_data_list[j][16][1] - raw_data_list[j][13][1]
        ring_finger_direction_tmp[2] = raw_data_list[j][16][2] - raw_data_list[j][13][2]
        ring_finger_vector_size = math.sqrt((ring_finger_direction_tmp[0] ** 2) + (ring_finger_direction_tmp[1] ** 2) + (ring_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        ring_finger_direction_tmp[0] /= ring_finger_vector_size
        ring_finger_direction_tmp[1] /= ring_finger_vector_size
        ring_finger_direction_tmp[2] /= ring_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        ring_finger_direction += ring_finger_direction_tmp

        # 小指 17 -> 20
        pinkie_finger_direction_tmp[0] = raw_data_list[j][20][0] - raw_data_list[j][17][0]
        pinkie_finger_direction_tmp[1] = raw_data_list[j][20][1] - raw_data_list[j][17][1]
        pinkie_finger_direction_tmp[2] = raw_data_list[j][20][2] - raw_data_list[j][17][2]
        pinkie_finger_vector_size = math.sqrt((pinkie_finger_direction_tmp[0] ** 2) + (pinkie_finger_direction_tmp[1] ** 2) + (pinkie_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        pinkie_finger_direction_tmp[0] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[1] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[2] /= pinkie_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        pinkie_finger_direction += pinkie_finger_direction_tmp
    
    '''
    variationの生成
    区切った中での最後のフレーム - 最初のフレームで変化したベクトルを21個のlandmarkごとに生成する
    '''
    # variation_tmp = []
    # for j in range(21):
    #     mark3.append("1of4_variationX_" + str(j+1) )
    #     try:
    #         variation_tmp.append(raw_data_list[start_indexes[4]-1][j][0] - raw_data_list[start_indexes[3]][j][0])  # x座標の最初のフレームと最後のフレームの差
    #     except:
    #         print("raw_data_listの長さ: " + str(len(raw_data_list)))
    #         print(start_indexes)
    #         print(raw_data_list[start_indexes[4]-1][j][0])
    #         print(raw_data_list)
    #         print(raw_data_list[start_indexes[3]])
    #     mark3.append("1of4_variationY_" + str(j+1) )
    #     variation_tmp.append(raw_data_list[start_indexes[4]-1][j][1] - raw_data_list[start_indexes[3]][j][1])  # y座標
    #     mark3.append("1of4_variationZ_" + str(j+1) )
    #     variation_tmp.append(raw_data_list[start_indexes[4]-1][j][2] - raw_data_list[start_indexes[3]][j][2])  # z座標
    

    #今まで計算した特徴はすべてのフレームの合計なので、使用　フレーム数で割って平均にする
    for j in range(len(distance_average_tmp)):
        distance_average_tmp[j] = distance_average_tmp[j] / used_frame_counter
    for j in range(len(angle_average_tmp)):
        angle_average_tmp[j] = angle_average_tmp[j] / used_frame_counter
    
    for j in range(3):
        thumb_direction[j] /= used_frame_counter
        index_finger_direction[j] /= used_frame_counter
        middle_finger_direction[j] /= used_frame_counter
        ring_finger_direction[j] /= used_frame_counter
        pinkie_finger_direction[j] /= used_frame_counter
    
    # # データ確認
    # confirm_variable_name = variation_tmp
    # print("データ確認")
    # print(len(confirm_variable_name))
    # print(confirm_variable_name)

    '''
    distance_average_tmp: 190個の実数データ
    angle_average:        630個の実数データ
    thumb_direction:      
    '''
    # この分割域で計算した特徴を一つの変数にまとめる
    distance_average = np.append(distance_average, distance_average_tmp)
    angle_average = np.append(angle_average, angle_average_tmp)
    thumb_direction_averages = np.append(thumb_direction_averages, thumb_direction)
    index_finger_direction_averages = np.append(index_finger_direction_averages, index_finger_direction)
    middle_finger_direction_averages = np.append(middle_finger_direction_averages, middle_finger_direction)
    ring_finger_direction_averages = np.append(ring_finger_direction_averages,ring_finger_direction)
    pinkie_finger_direction_averages = np.append(pinkie_finger_direction_averages, pinkie_finger_direction)
    # variation = np.append(variation, variation_tmp)

    
    #print("ここからファイルに書き込む")
    #計算した特徴をファイルに書き込むために一つの変数にまとめる
    all_features = np.array([], dtype="float")
    all_features = np.append(all_features, distance_average)
    all_features = np.append(all_features, angle_average)
    all_features = np.append(all_features, thumb_direction_averages)
    all_features = np.append(all_features, index_finger_direction_averages)
    all_features = np.append(all_features, middle_finger_direction_averages)
    all_features = np.append(all_features, ring_finger_direction_averages)
    all_features = np.append(all_features, pinkie_finger_direction_averages)
    # all_features = np.append(all_features, variation)

    # print(all_features)
    np.savetxt(wrtdir + Name + '_' + str(i) + '_' + 'feature.csv', all_features, delimiter=',')
    # print(wrtdir + Name + '_' + str(i) + '_' + 'feature.csv')