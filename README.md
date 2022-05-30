# FiarseqTutorial
Fairseq初心者のための日本語チュートリアルです．
NTTが公開している対話システム用大規模言語モデルであるJapanese Dialog Transformerのファインチューニング，及びCNN+LSTMのレガシーな画像キャプション生成モデルの作成と学習を行います．このチュートリアルを通して  
- 基本的なfairseq-cliの使い方  
- fairseqの大まかな仕組み  
- fairseqのカスタマイズ方法  

の習得を目指します．  

# 想定読者  
- pytorchでモデル実装を行ったことがある．  
- 言語モデルの学習や自然言語処理そのものについて、自然言語処理100本ノック程度の知識・実装の理解がある  
- pytorchのクラスやデコレータなどの応用的な知識がある   

## フォルダ構成  
`/materials`: チュートリアルスライド(pdf形式)置き場  
`/work_dialog`: Japanese Dialog Transformerのファインチューニングのハンズオンの作業ディレクトリ  
`/work_caption`: 画像キャプション生成のハンズオンの作業ディレクトリ  
