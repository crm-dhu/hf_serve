python -m transformers.onnx --model="Salesforce/codegen-6B-nl" --feature=causal-lm --framework=pt --preprocessor=tokenizer --atol=1e-4 codegen_6b_text_generation_onnx/
