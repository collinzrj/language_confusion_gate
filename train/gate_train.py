import torch, unicodedata, sys, os, json
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator, Dict
from dataclasses import dataclass, field
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM,Qwen2RMSNorm
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.processing_utils import Unpack
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from transformers import TrainingArguments, Trainer, PreTrainedModel, Qwen3Model
from transformers.utils import ModelOutput
from peft import LoraConfig, get_peft_model, TaskType
from confusion_gate import Qwen3Gating, contains_cj, Qwen3MoeGating, LlamaGating, GemmaGating, OlmoGating
import torch.distributed as dist

import functools
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TRAIN_GPTOSS = os.environ.get('TRAIN_GPTOSS', '').lower() == 'true'
TRAIN_QWEN = os.environ.get('TRAIN_QWEN', '').lower() == 'true'
print("TRAIN_QWEN", TRAIN_QWEN)

# # Store the original function
# original_init = dist.init_process_group

# # Define a wrapper that triggers a breakpoint
# @functools.wraps(original_init)
# def debug_init(*args, **kwargs):
#     print("Breakpoint: torch.distributed.init_process_group called")
#     print("Call stack:")
#     traceback.print_stack()  # This prints the full trace
#     # breakpoint()  # Execution will pause here
#     return original_init(*args, **kwargs)

def load_model(base_path):
    config = AutoConfig.from_pretrained(base_path).__dict__
    if config['architectures'][0] == 'Qwen3ForCausalLM':
        print("load Qwen3Gating")
        model = Qwen3Gating.from_pretrained(
            base_path,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True
        )
    elif config['architectures'][0] == 'Qwen2MoeForCausalLM' or config['architectures'][0] == 'Qwen3MoeForCausalLM':
        print("load Qwen3MoeGating")
        model = Qwen3MoeGating.from_pretrained(
            base_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True
        )
    elif config['architectures'][0] == 'LlamaForCausalLM':
        model = LlamaGating.from_pretrained(
            base_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True
        )
    elif config['architectures'][0] == 'Gemma3ForConditionalGeneration':
        model = GemmaGating.from_pretrained(
            base_path,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True
        )
    elif config['architectures'][0] == 'Olmo2ForCausalLM':
        model = OlmoGating.from_pretrained(
            base_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True
        )
    elif config['architectures'][0] == 'GptOssGating':
        # from transformers import GptOssForCausalLM
        # model = GptOssForCausalLM.from_pretrained(base_path, torch_dtype="auto", output_hidden_states=True)
        from confusion_gate import GptOssGating
        model = GptOssGating.from_pretrained(base_path, torch_dtype="auto", output_hidden_states=True)
    else:
        raise NotImplementedError
    return model

# Define data collator
@dataclass
class CustomDataCollator:
    tokenizer: Any
    padding: Union[bool, str] = "max_length"
    max_length: Optional[int] = 2048
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # input_ids = [f["repeat_response_token_ids"][:self.max_length] for f in features]
        # input_ids2 = [f["token_ids"][:self.max_length] for f in features]
        # print([len(x) for x in input_ids])
        # print([len(x) for x in input_ids2])
        def get_token_ids(query, query_response):
            if not TRAIN_GPTOSS:
                return self.tokenizer.apply_chat_template([
                    {'role': 'user', 'content': query},
                    {'role': 'assistant', 'content': query_response},
                ])
            else:
                prefix = '<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>'
                if '</think>' in query_response:
                    thinking = query_response.split('</think>')[0].replace('<think>', '')
                    new_query_response = query_response.split('</think>')[1]
                    # print('thinking', [thinking])
                    # print('new_query_response', [new_query_response])
                else:
                    thinking = ''
                    new_query_response = query_response
                oss_text = prefix + f'<|start|>user<|message|>{query}<|end|>' + f'<|start|>assistant<|channel|>analysis<|message|>{thinking}<|end|>' + '<|start|>assistant<|channel|>final<|message|>{new_query_response}<|return|>'
                return self.tokenizer.encode(oss_text)

        # tokens have been computed for qwen, no need to recompute
        input_ids = [get_token_ids(f['query'], f['query_response'])[:self.max_length] for f in features]

        # Use built-in tokenizer padding
        self.tokenizer.padding_side  = 'left'
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return batch

if __name__ == "__main__":
    USE_OPEN_SOURCE_DS = os.environ.get('USE_OPEN_SOURCE_DS', 'false').lower() == 'true'
    print("USE_OPEN_SOURCE_DS", USE_OPEN_SOURCE_DS)
    base_path = sys.argv[1]
    model_name = sys.argv[2]
    print("gate_train Base model is", base_path)
    token_norm_str = os.environ.get('TOKEN_NORM', 'false').strip().lower()
    should_token_norm = token_norm_str == 'true'
    print('should_token_norm', should_token_norm)
    
    import yaml
    # Open and load the YAML file
    with open("./deepspeed_config.yaml", "r") as file:
        deepspeed_config = yaml.safe_load(file)
    print(deepspeed_config)
    from datetime import datetime
    now = datetime.now()
    top_k = 20
    top_p = 0.95
    formatted_str = now.strftime("%Y-%m-%d-%H:%M:%S")
    output_dir = f"./models/gate-{model_name}-{top_k}k_{int(top_p * 100)}p_{formatted_str}"
    print('output_dir', output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=deepspeed_config['deepspeed_config']['train_micro_batch_size_per_gpu'],
        per_device_eval_batch_size=deepspeed_config['deepspeed_config']['train_micro_batch_size_per_gpu'],
        learning_rate=2e-5,
        save_strategy="no",
        save_steps=0,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="no",
        report_to="tensorboard",
        push_to_hub=False,
        # fp16=True,
        bf16=True,
        gradient_accumulation_steps=deepspeed_config['deepspeed_config']['gradient_accumulation_steps'],
        warmup_steps=100,
        weight_decay=0.01,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        save_total_limit=2,
    )
    print("before load model")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(base_path)
    model.load_lang_masks(tokenizer)
    print("after load model")
    model.should_token_norm = should_token_norm
    model.top_k = top_k
    model.top_p = top_p
    model.generation_config.return_dict_in_generate = True
    for name, param in model.named_parameters():
        if 'code_switch' not in name:
            param.requires_grad = False
    print('after loading model')
    # Load dataset:
    train_dataset = pd.read_json('./data/codeswitch_gate_opensource_alpaca_train.jsonl', lines=True).to_dict('records')
    test_dataset = None
    print("finish load dataset")
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=CustomDataCollator(tokenizer),
        tokenizer=tokenizer,
    )
    print("current gpu memory", torch.cuda.memory_summary(device=None, abbreviated=True))
    # Start training
    trainer.train()
    trainer.save_model(output_dir)