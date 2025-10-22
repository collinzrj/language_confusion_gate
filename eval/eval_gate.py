import os
import sys
import requests
import json
# Import necessary libraries
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# --- Configuration ---

# SET VLLM_USE_V1=0 before run (if using vLLM locally)
# os.environ['VLLM_USE_V1'] = '0' # Uncomment if needed

# Ensure you set your OpenRouter API key as an environment variable
# export OPENROUTER_API_KEY='your_actual_api_key_here'
GREEDY_SAMPLING = True
print("GREEDY_SAMPLING", GREEDY_SAMPLING)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
USE_OPENROUTER = os.environ.get("USE_OPENROUTER", '').lower() == 'true'
USE_INTERVENTION_PROMPT = os.environ.get("USE_INTERVENTION_PROMPT", '').lower() == 'true'
if USE_INTERVENTION_PROMPT:
    intervention_prefix = """Language confusion in the context of large language models (LLMs) refers to the phenomenon where a model mixes or confuses elements of multiple languages inappropriately during generation. Pay attention to prevent it. For example, "곧 방호복을 입은 경찰관들이 yard에 들어와 타격 가스로 수감자들을 몰아넣었다." is incorrect. It should be "곧 방호복을 입은 경찰관들이 마당에 들어와 최루가스로 수감자들을 몰아넣었다." Pay attention to avoid language confusion during generation.\n\n"""
else:
    intervention_prefix = ''

# --- Data Preparation ---
def prepare_include(tokenizer=None, think=False):
    # If tokenizer is None, we are likely using OpenRouter and don't need to apply chat template here
    def get_prefix(lang):
        answer_trans = {
            'Arabic': 'إجابة',
            'Hebrew': 'תשובה',
            'Greek': 'Απάντηση',
            'Russian': 'Ответ',
            'Vietnamese': 'Câu trả lời'
        }
        return f"Answer the following multiple choice question. The last line of your response should be of the following format: '{answer_trans[lang]}: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering, reply in {lang}."
    langs = ['Arabic', 'Hebrew', 'Greek', 'Russian', 'Vietnamese']
    answers = []
    prompts = []
    for lang in langs:
        lang_ds = load_dataset("CohereLabs/include-base-44", lang)
        for row in lang_ds['test']:
            prompt = get_prefix(lang) + row['question'] + '\n\n' + f"A) {row['option_a']}\n" + f"B) {row['option_b']}\n" + f"C) {row['option_c']}\n" + f"D) {row['option_d']}"
            prompt = intervention_prefix + prompt
            if tokenizer:
                prompt = tokenizer.apply_chat_template([
                    {'role': 'user', 'content': prompt}
                ], add_generation_prompt=True, tokenize=False, enable_thinking=False)
            if think:
                prompt += '  <think>' # Add thinking token if needed by the model (Note: space added for clarity, adjust if needed)
            prompts.append(prompt)
            answers.append(row['answer'])
    return prompts, answers

def prepare_flores_no_latin(tokenizer=None, think=False, add_prompt=False):
    df = pd.read_json('./data/flores_no_latin_eval.jsonl', lines=True)
    prompts = []
    answers = []
    def format_prompt(prompt):
        prompt = intervention_prefix + prompt
        if tokenizer:
            return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        else:
             # If no tokenizer, assume raw prompt is fine for OpenRouter or model handles it
            return prompt
    for _, row in df.iterrows():
        prompts.append(format_prompt(row['query']))
        answers.append(row['query_response'])
    return prompts, answers

def prepare_flores_with_latin(tokenizer=None, think=False, add_prompt=False):
    df = pd.read_json('./data/flores_with_latin_eval.jsonl', lines=True)
    prompts = []
    answers = []
    def format_prompt(prompt):
        prompt = intervention_prefix + prompt
        if tokenizer:
            return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        else:
             # If no tokenizer, assume raw prompt is fine for OpenRouter or model handles it
            return prompt
    for _, row in df.iterrows():
        prompts.append(format_prompt(row['query']))
        answers.append(row['query_response'])
    return prompts, answers

def prepare_humaneval_xl(tokenizer=None, think=False):
    print('start prepare_humaneval_xl')
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # pls = ['python', 'java', 'javascript', 'csharp', 'go', 'kotlin', 'perl', 'php', 'ruby', 'scala', 'swift', 'typescript']
    # languages = ["English", "Russian", "Chinese", "German", "Spanish", "French", "Italian", "Portuguese", "Greek", "Hungarian", "Dutch", "Finnish", "Indonesian", "Turkish", "Arabic", "Vietnamese", "Bulgarian", "Persian", "Malay", "Hebrew", "Estonian", "Tagalog", "Afrikaans"]
    pls = ['python', 'perl']
    langs = ['Arabic', 'Hebrew']
    prompts = []
    answers = []
    def format_prompt(prompt):
        prompt = intervention_prefix + prompt
        if tokenizer:
            return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
        else:
             # If no tokenizer, assume raw prompt is fine for OpenRouter or model handles it
            return prompt
    for _ in range(2):
        dataset = load_dataset("floatai/HumanEval-XL", "python", trust_remote_code=True)
        for lang in langs:
            lang_ds = dataset[lang]
            for sample in lang_ds:
                # 5 trials on each prompt
                for _ in range(5):
                    prompts.append(format_prompt(sample['prompt']))
                    answers.append(sample['task_id'])
    print('finish prepare_humaneval_xl')
    return prompts, answers

# --- Generation Logic ---
def generate_with_vllm(llm, prompts, sampling_params):
    """Generate using vLLM."""
    return llm.generate(prompts, sampling_params)

# --- Unified Main Execution Logic ---

def run_eval(dataset_name, model_path):
    """
    Run a test for a given dataset and model, automatically detecting backend.
    """
    print(f"Running '{dataset_name}' eval for model: {model_path}...")

    # Prepare Data
    if dataset_name == 'include':
        prepare_fn = prepare_include
        default_max_tokens = 8000
    elif dataset_name == 'flores-no-latin':
        prepare_fn = prepare_flores_no_latin
        default_max_tokens = 2000
    elif dataset_name == 'flores-with-latin':
        prepare_fn = prepare_flores_with_latin
        default_max_tokens = 2000
    elif dataset_name == 'humaneval':
        prepare_fn = prepare_humaneval_xl
        default_max_tokens = 8000
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    # --- Use vLLM ---
    print("Detected local model (vLLM)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    should_think = 'think' in model_path

    prompts, answers = prepare_fn(tokenizer, should_think)
    max_tokens = default_max_tokens # Use default, could make configurable if needed

    # Setup LLM and Sampling Params

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=8192, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=max_tokens)
    outputs = generate_with_vllm(llm, prompts, sampling_params)
    res_arr = []
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        thinking = getattr(output.outputs[0], 'thinking', '')
        res_arr.append({
            'query': prompt,
            'query_response': generated_text,
            'answer': answer,
            'thinking': thinking
        })
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_path = f'./data/{model_path}_{dataset_name}_res_{timestamp}.jsonl' # Sanitize filename
    with open(out_path, 'w', encoding='utf-8') as f:
        for row in res_arr:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print("Saved to:", out_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <dataset_name> <model_path>")
        print("       dataset_name: 'include' or 'flores'")
        print("       model_path: the path to the model")
        sys.exit(1)

    dataset_name = sys.argv[1].lower()
    model_path = sys.argv[2]

    run_eval(dataset_name, model_path)


# VLLM_USE_V1=0 TOK_PATH='Qwen/Qwen3-8B' python eval_gate.py flores-no-latin /share/shmatikov/collin/language_confusion_project/language_confusion_paper/gate_weights/gate-qwen3-8b-20k_95p_2025-08-25-13:26:19_plugged