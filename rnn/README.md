## Train
~~~sh
python main.py --use_gpu --train --train_text train-of-ja-wiki.wakati.txt --model ja-wiki --vocab_file vocab.txt --n_epoch 5
~~~

## Test
~~~sh
python main.py --use_gpu --evaluate --evaluate_text test-of-ja-wiki.wakati.txt --model ja-wiki-5epoch.model --vocab_file vocab.txt
~~~

loss:4.19947195053  sentence: iPhone6 の おすすめ カラー は ローズ ゴールド です 。
loss:4.42352819443  sentence: iPhone6 は カラー の おすすめ が ローズ ゴールド の カラー です 。
loss:4.70171880722  sentence: iPhone6 の 人気 は おすすめ カラー で ローズ ゴールド です 。
loss:4.86169385910  sentence: iPhone6 が おすすめ な 人気 は ローズ ゴールド で カラー です 。
loss:4.80426645279  sentence: iPhone6 に 人気 な の は おすすめ カラー ローズ ゴールド です 。
loss:4.57957983017  sentence: iPhone6 を おすすめ が カラー の ローズ ゴールド で 人気 です 。
loss:5.54693555832  sentence: Xperia は 高 機能 な のに 安い ので 価格 を 気 に し て いる 人 に は おすすめ です 。
loss:5.09127521515  sentence: Xperia が 高 機能 な のに 安い の は 価格 を 気 に し て いる 人で は おすすめ です 。
loss:5.29798030853  sentence: Xperia は 安い から 高 機能 な ので 価格 を 気 に し て いる 人 に は おすすめ です 。
loss:5.36294460297  sentence: Xperia は 高 機能 で も 価格 を 安い 人 は 気 に し て いる ので おすすめ です 。
loss:5.57070827484  sentence: Xperia の 高 機能 な のに 価格 は 安い ので 気 に し て いる おすすめ の 人 です 。
loss:6.33925390244  sentence: Mac は ディスプレイ が 超 きれい と 評判 です よ 。
loss:6.11912536621  sentence: Mac の ディスプレイ に 超 きれい が 評判 です よ 。
loss:6.36941289902  sentence: Mac で ディスプレイ と 超 きれい に 評判 です よ 。
loss:6.36844825745  sentence: Mac は きれいな 超 ディスプレイ が 評判 です よ 。
loss:6.35109186172  sentence: Mac は 評判 で 超 きれい と ディスプレイ です よ 。
loss:3.39383363724  sentence: iPhone6 の おすすめ カラー は ローズ ピンク です 。
loss:4.57050514221  sentence: iPhone6 の おすすめ カラー は ディスプレイ です 。
loss:5.23725128174  sentence: iPhone6 の おすすめ カラー は 豊富 な アプリ です 。
loss:3.54550409317  sentence: iPhone6 の おすすめ カラー は セキュリティ です 。
loss:4.65393924713  sentence: iPhone6 の おすすめ カラー は 価格 です 。
loss:5.62628173828  sentence: Xperia は カメラ の 性能 が 高い から おすすめ です 。
loss:5.87420463562  sentence: Xperia は 人気 色 の 性能 が 高い から おすすめ です 。
loss:5.63660669327  sentence: Xperia は 価格 の 性能 が 高い から おすすめ です 。
loss:5.67155313492  sentence: Xperia は アプリ の 種類 の 性能 が 高い から おすすめ です 。
loss:5.93311738968  sentence: Xperia は 若者 人気 の 性能 が 高い から おすすめ です 。
