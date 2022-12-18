torch-model-archiver \
--model-name BertFillMask \
--version 1.0 \
--serialized-file  Transformer_model/pytorch_model.bin \
--handler ModelHandler.py \
--extra-files "Transformer_model/config.json,Transformer_model/special_tokens_map.json,Transformer_model/tokenizer_config.json,Transformer_model/tokenizer.json,Transformer_model/vocab.txt" \
--export-path model_store
