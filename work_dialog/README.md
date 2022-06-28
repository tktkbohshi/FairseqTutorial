# 作業手順  
## データのダウンロードと前処理  
1. `$sh data_download.sh`を実行（時間がかかります）．  
2. `$python scripts/extract_persona_chat.py japanese_persona_chat.xlsx data/perchat/raw/`でデータセットの分割を行う．  
3. `$python scripts/tokenize.sh`でトークナイズを行う．  

## 学習の実行  
`$sh train.sh`を実行する．  

## モデルを試す  
`$sh interactive.sh`，または`$sh generate.sh`を実行する．  
