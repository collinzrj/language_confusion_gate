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

# --- Unified Model Paths ---
# Use file paths for local models, OpenRouter model identifiers for remote models.
# Add 'openrouter_alias' if the identifier differs from the key.
model_dict = {
    'qwen3-8b-nogate': {
        'path': '/share/shmatikov/collin/language_confusion_project/language_confusion_paper/gate_weights/qwen3-8b'
    },
    'qwen3-8b-nonorm': {
        'path': '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-qwen3-8b-nonorm-20k_95p_2025-09-02-02:50:25_plugged'
    },
    'qwen3-8b-norm': {
        'path': '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-qwen3-8b-20k_95p_2025-08-25-13:26:19_plugged'
    },
    '30b-nogate': {
        'path': '/cpfs01/user/jiawei.lyt/ckpt/verl_checkpoints/lyt-rl-gen/qwen3-tpp-nothink-0721-distilled-data0706-recitex1-bothtrans-mixlangx2-GenRM-32B-sentcs-GSPO-ref-turbopp-LENGTH_FLIP_THRESHOLD1.3-LENGTH_FLIP_PROB0.75-REF_ANSWER_POSITION-A-expert-12k_bs512_minibs128_n8/global_step_60/actor_hf'
    },
    '30b-gate-oss': {
        'path': '/cpfs01/user/xiujian.zrj/cs_gate_train/models/opensource-turbo-nothink-gate-qwen3-controlfix-20k_95p_flores_2025-08-22-15:21:43_plugged'
    },
    '30b-gate-oss-norm': {
        'path': '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-qwen-30b-norm-20k_95p_2025-08-28-08:59:51_plugged'
    },
    '30b-think-nogate': {
        'path': '/cpfs01/user/jiawei.lyt/ckpt/verl_checkpoints/lyt-rl-gen/qwen3-tpp-thinking-fh0723-mkd035-distilled-data0706-recitex1-bothtrans-mixlangx2-GenRM-32B-sentcs-GSPO-ref-turbopp-THINK-FLIP1-2.4-LENGTH_FLIP_THRESHOLD1.3-LENGTH_FLIP_PROB0.75-REF_ANSWER_POSITION-A-expert-12k_bs512_minibs128_n8/global_step_90/actor_hf'
    },
    '30b-think-norm': {
        'path': '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-qwen-30b-think-norm-20k_95p_2025-09-01-12:07:35_plugged'
    },
    'llama-8b-oss': {
        'path': '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-llama3-8b-20k_95p_norm_2025-08-25-03:50:13_plugged'
    },
    'llama-8b-oss_nonorm': {
        'path': '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-llama3-8b-20k_95p_nonorm_2025-08-24-11:41:02_plugged'
    },
    'llama-8b-nogate': {
        'path': 'meta-llama/Llama-3.1-8B-Instruct'
    },
    'gemma-12b-oss': {
        'path': '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-gemma3-12b-20k_95p_2025-08-26-02:18:32_plugged'
    },
    'gemma-12b-nogate': {
        'path': 'google/gemma-3-12b-it'
    },
    'gemma-12b-oss-nonorm': {
        'path': '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-gemma3-12b-nonorm-20k_95p_2025-08-28-08:33:06_plugged'
    },
    'gpt-oss-20b-nogate-local': { # Renamed key to distinguish from OpenRouter version
        'path': '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gpt-oss-20b'
    },
    'gpt-oss-20b-norm': {
        'path': '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-gpt-oss-20b-norm-20k_95p_2025-08-29-12:31:23_plugged'
    },
}

# --- Data Preparation ---
def prepare_include(tokenizer=None, think=False):
    # If tokenizer is None, we are likely using OpenRouter and don't need to apply chat template here
    def get_prefix(lang):
        answer_trans = {
            'Arabic': 'إجابة',
            'Hebrew': 'תשובה',
            'Greek': 'Απάντηση',
            'Korean': '답변',
            'Russian': 'Ответ',
            'Vietnamese': 'Câu trả lời'
        }
        return f"Answer the following multiple choice question. The last line of your response should be of the following format: '{answer_trans[lang]}: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering, reply in {lang}."
    langs = ['Arabic', 'Hebrew', 'Greek', 'Korean', 'Russian', 'Vietnamese']
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

def run_test(dataset_name, model_name, num_workers=10):
    """
    Run a test for a given dataset and model, automatically detecting backend.
    """
    print(f"Running '{dataset_name}' test for model: {model_name}...")

    # --- Input Validation ---
    if model_name not in model_dict:
        print(f"Error: Model '{model_name}' not found in model_dict.")
        sys.exit(1)

    if dataset_name not in ['include', 'flores', 'flores-no-latin', 'flores-with-latin', 'humaneval']:
        print(f"Error: Unsupported dataset name '{dataset_name}'. Please use 'include' or 'flores'.")
        sys.exit(1)

    model_config = model_dict[model_name]

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

    # --- Determine Backend and Prepare Data/Sampling ---
    if 'path' in model_config:
        # --- Use vLLM ---
        print("Detected local model (vLLM)...")
        if 'gpt' not in model_name.lower():
            assert os.environ.get('VLLM_USE_V1') == '0', "Please set VLLM_USE_V1=0 for vLLM"
        else:
            assert os.environ.get('VLLM_USE_V1') == '1', "Please set VLLM_USE_V1=1 for vLLM for GPTOSS"

        model_path = model_config['path']
        # if dataset_name == 'include': # Set TOK_PATH only for include if needed by tokenizer
        #     os.environ['TOK_PATH'] = model_path
        tokenizer = AutoTokenizer.from_pretrained(os.environ['TOK_PATH'])
        should_think = 'think' in model_name

        prompts, answers = prepare_fn(tokenizer, should_think)
        max_tokens = default_max_tokens # Use default, could make configurable if needed

        # Setup LLM and Sampling Params
    
        if 'olmo' in model_name:
            llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096, tensor_parallel_size=2)
        else:
            llm = LLM(model=model_path, dtype="bfloat16", max_model_len=8192, tensor_parallel_size=2)

        if 'gemma' in model_name:
            if GREEDY_SAMPLING:
                top_k = 1
            else:
                top_k = 64
            sampling_params = SamplingParams(temperature=1, top_p=0.95, top_k=64, max_tokens=max_tokens)
        if 'gpt' in model_name:
            # p should be 1 and k should be none, but this doesn't fit to the intervention impl
            sampling_params = SamplingParams(temperature=1, top_p=0.9999, top_k=100, max_tokens=max_tokens)
        else:
            if GREEDY_SAMPLING:
                top_k = 1
            else:
                top_k = 20
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=top_k, max_tokens=max_tokens)

        if 'flores' in dataset_name:
            sampling_params.max_tokens = 512

        print("sampling_params", sampling_params)

        # Generate
        outputs = generate_with_vllm(llm, prompts, sampling_params)
    else:
        print(f"Error: Model configuration for '{model_name}' is invalid. It must have either 'path' or 'openrouter_alias'.")
        sys.exit(1)

    # --- Process and Save Results (Common for both backends) ---
    res_arr = []
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # Attempt to get thinking; default to empty if not present (e.g., for vLLM)
        thinking = getattr(output.outputs[0], 'thinking', '')
        # print("Prompt:", [prompt])
        # print("Response:", [generated_text])
        # print("Thinking:", [thinking]) # Optional: print thinking
        res_arr.append({
            'query': prompt,
            'query_response': generated_text,
            'answer': answer,
            'thinking': thinking # Add thinking to the result array
        })
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    if USE_INTERVENTION_PROMPT:
        prompt_note = 'useprompt_'
    else:
        prompt_note = ''
    out_path = f'./data/{safe_model_name}_{dataset_name}_res_{prompt_note}{timestamp}.jsonl' # Sanitize filename
    with open(out_path, 'w', encoding='utf-8') as f:
        for row in res_arr:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print("Saved to:", out_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <dataset_name> <model_name>")
        print("       dataset_name: 'include' or 'flores'")
        print("       model_name: A key from the model_dict (e.g., '30b-nogate', 'gpt-oss-20b-nogate')")
        print("Example: python script.py include 30b-nogate")
        print("Example: python script.py flores gpt-oss-20b-nogate")
        print("Note: Set OPENROUTER_API_KEY environment variable if testing OpenRouter models.")
        sys.exit(1)

    dataset_name = sys.argv[1].lower()
    model_name = sys.argv[2]

    run_test(dataset_name, model_name, 50) # Reduced default workers for testing