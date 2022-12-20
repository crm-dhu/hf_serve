from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "codegen_text_generation"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model_inputs = tokenizer("This is a great", return_tensors='pt', return_attention_mask=False)
result = model.generate(**model_inputs, max_new_tokens=24, do_sample=True, top_p=0.95, temperature=0.75,
                        num_return_sequences=5, no_repeat_ngram_size=2)
print(tokenizer.batch_decode(result))
