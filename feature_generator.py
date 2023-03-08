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
srcdir = '../kakizaki/data/' + str(Name) + '/preprocessed_coordinates_m/'
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
for i in range(1,42):  # 静的指文字のファイル名を減らしていく
    new_file_list = []
    j = START_INDEX  # カウンタ変数
    while len(new_file_list) < STATIC_FILE_USE_NUM:  # 規定ファイル数を確保するまで
        try:
            new_file_list.append(hand_data_list[i][j])
        except:
            print("静的指文字のファイル数が足りませんでした")
            print("hand_type: " + str(i))
            print("ファイル箇所: " + str(j))
            exit(1)
        j = j + 1
    # new_file_listに置き換える
    hand_data_list[i] = copy.deepcopy(new_file_list)

'''
ここから実際に特徴抽出を行っていく
'''
# まずはファイル名リストhand_data_listを使って実際のデータをnumpy配列に取ってくる

all_distance_max = []
all_distance_min = []
all_distance_average = []
all_angle_max = []
all_angle_min = []
all_angle_average = []
all_variation = []
all_direction_average = []

for i in range(len(hand_data_list)):
    frame_num = len(hand_data_list[i])
    split_num = frame_num / 4  # データを時間軸で4分割するため、何個ずつにするか
    raw_data_list = []
    for j in range(len(hand_data_list[i])):
        # データ読み込み
        data = np.loadtxt(srcdir + hand_data_list[i][j], delimiter=',')
        raw_data_list.append(data)
    # ここまででraw_data_listに座標データ、frame_numにそのhand_typeのフレーム数が入っている
    start_indexes = [0]
    for j in range(3):
        start_indexes.append(int(start_indexes[j] + split_num))
    
    start_indexes.append(frame_num)
    # start_indexesに入っている数字から次の数字までの間で一つの特徴量を生成する
    '''
    Distance: average
    Angle: average
    Variation: 最初のフレームと最後のフレームの移動ベクトル
    Direction: average, 各指が向いている方向を方向ベクトルで示す
    '''
    distance_average = [0 for j in range(190)]
    angle_average = [0 for j in range(630)]
    variation = []
    direction_average = []

    # 各区切りで得られた特徴量を一時的に保存する変数
    distance_average_tmp = [0 for j in range(190)]
    angle_average_tmp = [0 for j in range(630)]
    variation_tmp = 0
    direction_average_tmp = 0

    used_frame_counter = 0
    # csvでculumnの説明として保存する部分を保存
    mark1=[]  # 距離
    mark2=[]  # 角度
    mark3=[]  # variation
    mark4=[]  # direction

    # raw_data_list[0 -> 1/4]
    for j in range(start_indexes[0], start_indexes[1]):
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
                    mark1.append("distance" + str(k) + str(l))
                # 3次元の点と点同士の距離をdistance_average[1-2などに入れる]
                distance_average[distance_index] += math.sqrt((raw_data_list[j][k][0]-raw_data_list[j][l][0])**2 + (raw_data_list[j][k][1]-raw_data_list[j][l][1])**2 + (raw_data_list[j][k][2]-raw_data_list[j][l][2])**2)
                distance_index += 1

        '''
        angleの生成
        '''
        angle_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # x角度
                angle_average[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # y角度
                angle_average[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # z角度
                angle_average[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # 最初の1フレームならmark2を作る
                if (j==start_indexes[0]) and (k == 0) and (l == 0):
                    mark2.append("angle_x" + str(k) + str(l))
                    mark2.append("angle_y" + str(k) + str(l))
                    mark2.append("angle_z" + str(k) + str(l))
        
        '''
        directionの生成
        averageを求める。単位ベクトル
        '''
        direction_index = 0

                

    #今まで計算した特徴はすべてのフレームの合計なので、使用フレーム数で割って平均にする
    for j in range(len(distance_average)):
        distance_average[j] = distance_average[j] / used_frame_counter
    for j in range(len(angle_average)):
        angle_average[j] = angle_average[j] / used_frame_counter


    
    # raw_data_list[1/4 -> 2/4]
    for j in range(start_indexes[1], start_indexes[2]):

    # raw_data_list[2/4 -> 3/4]
    for j in range(start_indexes[2], start_indexes[3]):
        

    # raw_data_list[3/4 -> 4/4]
    for j in range(start_indexes[3], start_indexes[4]):
