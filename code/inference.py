from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def model_fn(model_dir):
    model_8bit = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model_8bit, tokenizer


def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = data.pop("inputs", data)
    encoded_input = tokenizer(text, return_tensors='pt')
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), **data)
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)
