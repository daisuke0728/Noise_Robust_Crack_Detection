# Noise_Robust_Crack_Detection
ノイズに対して頑健な コンクリートひび割れ検出

※研究とは別に実施したものです

## タスク概要・背景

### 背景 ひび割れ検出
橋や建物などの社会インフラの安全管理は重要であるが、社会インフラ、特に、コンクリート構造物の⽼朽化が進んでいるといえる。
コンクリートのひび割れ点検作業は人手によって行われているものが多く、効率が悪いので、近年は画像を利用した方式が注目され、DeepLearningを使用した手法が高い精度を出している。

一方、必ずしもAIに適したきれいな画像を撮影できるとは限らず、これらのノイズは、モデルの精度を下げてしまう。

以上のことから、今回は、**ノイズの⼊った画像に対しても、ひびの有無を判断できるモデル**を作成しました。

## 開発環境
```
python=3.8.10
torch=1.10.1+cu113
torchvision=0.11.2+cu113
numpy=1.20.3
pandas=1.3.1
matplotlib=3.4.2
```

## 実行方法

1. SDNET2018をダウンロード
https://digitalcommons.usu.edu/all_datasets/48/

2. ベースラインの実行
ノイズ無し
`python ./main.py <data_dir of SDNET> ./result/original` 
ノイズあり(ぼかし)
`python ./main.py <data_dir of SDNET> ./result/blur_noise`
ノイズあり(ガウスノイズ)
`python ./main.py <data_dir of SDNET> ./result/gaussian_noise`
ノイズあり(ぼかし・ガウスノイズ)
`python ./main.py <data_dir of SDNET> ./result/blur_gaussian_noise` 

## Reference
Dorafshan S, Thomas RJ, Maguire M. SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks. Data Brief. 2018;21:1664-1668. Published 2018 Nov 6. doi:10.1016/j.dib.2018.11.015


