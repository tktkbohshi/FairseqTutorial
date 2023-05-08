### 学習時にエンコーダー（デコーダー）のパラメーターを固定する  
fairseq/fairseq/trainer.pyのload_checkpoint関数について  
修正前
```
self.model.load_state_dict(
    state["model"], strict=True, model_cfg=self.cfg.model
)
```
↓  
修正後  
```
self.model.load_state_dict(
    state["model"], strict=True, model_cfg=self.cfg.model
)

#encoderを固めたいならば
for param in self.model.encoder.parameters():
    param.requires_grad = False

#decoderを固めたいならば
for param in self.model.decoder.parameters():
    param.requires_grad = False
```  

self.modelの中身を知りたい人はfairseq/fairseq/models/fairseq_model.pyのベースモデル実装（特にFairseqEncoderDecoderModelクラス）を参考のこと  