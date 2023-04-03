# dynamic_JSL
今回使用するデータは学部の卒論研究でしようしたものと同じなため、以下のディレクトリとなる

../kakizaki/data/

## 新たな特徴量を使う
2フレームのみを使用するvariationは消して、一つの動的指文字の特徴量ファイルにすべてのフレームを使用することにする。

### わたなべ先輩の方法
すべてのフレームを使用すると、被験者によって動的指文字の表現に要する時間が違うことから同じ量の特徴量を生成することが困難である。

この問題を解決し、指文字に要する時間が違っても一定量の特徴量を獲得する方法としてわたなべ先輩の時間軸に対してデータを??%区切りにして、それぞれ区切った場所で抽出するという方法を採用する。

区切った場所で抽出する特徴量の種類は以下を考えた。

- landmark同士の方向ベクトル(指の向き)
- 区切った中でのlandmarkのmax座標,min座標, またそれの次の区切りまでの変化
- 今までも使用していたangleの区切りの中でのmax, min

今回は動的手話のフレーム数が最小で4枚だったため、データの変更がない限り時間軸に対し4分割で対応する。

動いているかいないかを分けるために分散を特徴量に用いることも案の一つ

以前まで使っていたdistance, angle, variationはすべて1 or 2フレームに対するものだったため、これらをすべて区切った中でのmax, min, averageなどに置き換えることとする。


## 2/23
### 特徴量生成ファイルを作成する


## 3/22
ファイル生成プログラムが作成できたため、特徴ファイルの生成を行う。

まだ生成された特徴が正しいものかもわからないが、ファル生成を行っていく。このときファイル数が足りない人のデータなどからは処理が途中で止まるため、今後対応を考えなければいけない。該当エラーが出る人の名前をメモする。

- konnai (静的ファイル数不足)
- fujiwara (動的ファイル4以下のため分割不可)
- baba (invalid value encountered)
- satya (invalid value encountered)
- suzuki (invalid value encountered)
- sekizawa (invalid value encountered)
- keiji (invalid value encountered)
- mina (invalid value encountered)
- hirooka (invalid value encountered)
- moniru (動的ファイル4以下のため分割不可)
- jozume (動的ファイル4以下のため分割不可)
- watanabe (invalid value encountered)
- yuta (invalid value encountered)
- aosi (静的ファイル数不足)
- akiba (invalid value encountered)

かなりのデータに対してエラーになってしまうことがわかった。

invalid value encounterredは0で割ったときに起きるエラーで、該当箇所ではused_frame_counterで割っているので、フレームを一つも使っていないという状況を回避する必要がある。

variationは4分割したときに3つしか出ないということが忘れていたので、variationの生成を

```py
raw_data_list[4] - raw_data_list[3]を除いた最初の３つにすることで解決できた
```

ここまでで
- konnai (静的ファイル数不足)
- fujiwara (動的ファイル4以下のため分割不可)
- sekizawa (index error in variation generate)
- keiji (index error in variation generate)
- mina (invalid value encountered)
- moniru (動的ファイル4以下のため分割不可)
- aosi (静的ファイル数不足)

までエラーが出るファイルが減った

静的ファイルの数が12を下回っているものが処理が止まってしまうので、12より下回っていて、4よりは多いものについては動的指文字と同じような操作をして4分割にしていくことにする

これでkonnaiが直った。aosiはzero divitionになった

# 関数化することによって実行できるようになった
- fujiwara
46: ファイル数不足
15: ファイル数不足
- moniru
14: ファイル数不足
20: ファイル数不足
39: ファイル数不足
- jozume
10: ファイル数不足

特徴のアルゴリズムの正誤は不明だが、以上を除きすべてのファイルに対し実行ができるようになった
