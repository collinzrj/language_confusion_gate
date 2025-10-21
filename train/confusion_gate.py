import torch, unicodedata, os
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator, Dict
from dataclasses import dataclass, field
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM,Qwen2RMSNorm
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer
from transformers.processing_utils import Unpack
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from transformers import TrainingArguments, Trainer, PreTrainedModel, Qwen3ForCausalLM, Qwen3Model, Qwen3MoeForCausalLM, LlamaForCausalLM, Gemma3ForCausalLM, Olmo2ForCausalLM
from transformers.utils import ModelOutput
from peft import LoraConfig, get_peft_model, TaskType
from tok_contains_partial_cjk import contain_partial_cjk
import time
import sys

CACHE_HS = os.environ.get('CACHE_HS', False)
print("CACHE_HS", CACHE_HS)

def contains_cj(text):
    for char in text:
        if char == ' ':
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if not unicodedata.category(char).startswith('L'):
            continue
        if 'CJK' in name or 'KATAKANA' in name or 'HIRAGANA' in name:
            return 1
    return 0

def contains_latin(text):
    for char in text:
        if char == ' ':
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if not unicodedata.category(char).startswith('L'):
            continue
        if 'LATIN' in name:
            return 1
    return 0

def contains_only_special(text):
    for char in text:
        if char.isspace():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if unicodedata.category(char).startswith('L'):
            return 0
    return 1

# tokenizer = AutoTokenizer.from_pretrained('/cpfs01/user/xiujian.zrj/models/gate-qwen3-lowres-en')

def get_control_toks(vocab_size, tokenizer):
    special_toks = []
    for vs in tokenizer.special_tokens_map.values():
        if type(vs) is str:
            assert(len(tokenizer.encode(vs, add_special_tokens=False)) == 1)
            special_toks.append(tokenizer.encode(vs, add_special_tokens=False)[0])
        else:
            for v in vs:
                assert(len(tokenizer.encode(v, add_special_tokens=False)) == 1)
                special_toks.append(tokenizer.encode(v, add_special_tokens=False)[0])
    control_toks_mask = torch.zeros(vocab_size)
    print(special_toks)
    for tok in special_toks:
        control_toks_mask[tok] = 1
    control_toks_mask = torch.tensor(control_toks_mask).bool()
    return control_toks_mask

def get_lang_masks(vocab_size, tokenizer):
    cj_labels, latin_labels, special_labels, lowres_labels = [], [], [], []
    for tok in range(vocab_size):
        tok_text = tokenizer.decode(tok)
        cj_label = contains_cj(tok_text) or contain_partial_cjk(tokenizer, tok)
        # cj_label = contains_cj(tok_text)
        latin_label = contains_latin(tok_text) and not contains_cj(tok_text)
        special_label = not cj_label and contains_only_special(tok_text)
        cj_labels.append(cj_label)
        latin_labels.append(latin_label)
        special_labels.append(special_label)
        lowres_labels.append((cj_label == 0) and (latin_label == 0) and (special_label == 0))
    try:
        current_device = torch.cuda.current_device()
    except:
        current_device = 'cpu'
    print("Current device is", current_device)
    lang_masks = torch.tensor([cj_labels, latin_labels, special_labels, lowres_labels]).to(current_device)
    print('before', lang_masks[0].sum(), lang_masks[1].sum(), lang_masks[2].sum(), lang_masks[3].sum())
    control_toks_mask = get_control_toks(vocab_size, tokenizer)
    lang_masks[:, control_toks_mask] = 0
    lang_masks[2, control_toks_mask] = 1
    print('after', lang_masks[0].sum(), lang_masks[1].sum(), lang_masks[2].sum(), lang_masks[3].sum())
    return lang_masks

def contains_lang_name(text, lang_names):
    assert(type(lang_names) is list)
    assert(len(lang_names) > 0)
    for char in text:
        if char == ' ':
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if not unicodedata.category(char).startswith('L'):
            continue
        for lang_name in lang_names:
            if lang_name in name:
                return 1
    return 0

def get_lang_masks_lowres(vocab_size, tokenizer):
    labels = []
    control_toks_mask = get_control_toks(vocab_size, tokenizer)
    for tok in range(vocab_size):
        tok_text = tokenizer.decode(tok)
        if control_toks_mask[tok]:
            # this should be treated as special tok
            labels.append(2)
        if contains_cj(tok_text) or contain_partial_cjk(tokenizer, tok):
            labels.append(0)
        elif contains_lang_name(tok_text, ['ARABIC']):
            labels.append(5)
        elif contains_lang_name(tok_text, ['HANGUL']):
            labels.append(6)
        elif contains_lang_name(tok_text, ['THAI']):
            labels.append(7)
        elif contains_lang_name(tok_text, ['CYRILLIC']):
            labels.append(8)
        elif contains_latin(tok_text):
            labels.append(1)
        elif '�' in tok_text:
            labels.append(4)
        elif contains_only_special(tok_text):
            labels.append(2)
        else:
            # other lowres lang
            labels.append(3)
    current_device = torch.cuda.current_device()
    lang_masks = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=9)
    return lang_masks.t()

@torch.no_grad()
def batch_varied_top_k_p_in_logits(
    logits_batch: torch.Tensor,      # [N, V]
    lang_masks:   torch.Tensor,      # [C, V] bool
    top_ks:       torch.Tensor,      # [N] long
    top_ps:       torch.Tensor,      # [N] float
) -> torch.Tensor:
    """
    Vectorized version of batch_only_low_res_in_logits that handles
    different top_k and top_p values for each sequence in the batch.

    Returns
    -------
    labels : BoolTensor of shape [N, C]
    """
    N, V = logits_batch.shape
    C    = lang_masks.shape[0]
    device = logits_batch.device
    
    # Use the maximum top_k for the initial top-k operation
    max_k = int(top_ks.max())

    # 1) Top-k logits / indices (for max_k)
    topk_logits, topk_idx = torch.topk(logits_batch,
                                       k=max_k,
                                       dim=-1,
                                       sorted=True)  # [N, max_k]

    # 2) Probabilities of those k tokens
    # TODO: may this be different from topk?
    log_denom = torch.logsumexp(logits_batch.float(), dim=-1, keepdim=True)
    topk_probs = torch.exp((topk_logits.float() - log_denom))
    topk_probs = topk_probs.to(logits_batch.dtype) # [N, max_k]

    # 3) Create masks for sequence-specific top_k
    # This mask is True for tokens within the specific top_k of each sequence
    k_mask = torch.arange(max_k, device=device).expand(N, -1) < top_ks.unsqueeze(-1)
    
    # Apply k_mask to probabilities before nucleus sampling
    topk_probs.masked_fill_(~k_mask, 0.0)

    # 4) Nucleus (top_p) filter inside the k tokens
    cum_p = topk_probs.cumsum(dim=-1) # [N, max_k]
    
    # Remove tokens with cumulative probability > top_p
    # Keep the first token that exceeds the threshold
    prev_cum_p = torch.zeros_like(cum_p)
    prev_cum_p[:, 1:] = cum_p[:, :-1]
    
    # Skip nucleus filtering when top_p == 1.0
    p_mask = prev_cum_p < top_ps.unsqueeze(-1)  # [N, max_k] < [N, 1] -> [N, max_k]
    top_p_eq_1_mask = (top_ps == 1.0).unsqueeze(-1)  # [N] -> [N, 1]
    p_mask = p_mask | top_p_eq_1_mask  # [N, max_k] | [N, 1] -> [N, max_k] (broadcasting)
    
    # Final keep mask combines k and p filtering
    keep_mask = k_mask & p_mask # [N, max_k]

    # 5) Low-res check on the filtered tokens
    token_is_lang = lang_masks[:, topk_idx]             # [C, N, max_k]
    token_is_lang = token_is_lang.permute(1, 2, 0)      # [N, max_k, C]
    keep_mask_exp = keep_mask.unsqueeze(-1)             # [N, max_k, 1]
    any_lang = (keep_mask_exp & token_is_lang).any(dim=1) # [N, C]

    return any_lang

@torch.no_grad()
def chunked_batch_varied_top_k_p_in_logits(
    logits_batch: torch.Tensor,      # [N, V]
    lang_masks:   torch.Tensor,      # [C, V] bool
    top_ks:       torch.Tensor,      # [N] long
    top_ps:       torch.Tensor,      # [N] float
    chunk_size:   int = 128,        # Adjustable chunk size
) -> torch.Tensor:
    N = logits_batch.shape[0]
    
    # List to store outputs
    outputs = []

    for i in range(0, N, chunk_size):
        logits_chunk = logits_batch[i:i+chunk_size]
        top_ks_chunk = top_ks[i:i+chunk_size]
        top_ps_chunk = top_ps[i:i+chunk_size]

        chunk_output = batch_varied_top_k_p_in_logits(
            logits_batch=logits_chunk,
            lang_masks=lang_masks,
            top_ks=top_ks_chunk,
            top_ps=top_ps_chunk,
        )
        
        outputs.append(chunk_output)

    # Concatenate all chunk outputs along batch dimension
    result = torch.cat(outputs, dim=0)  # [N, C]

    return result

@dataclass
class CsLogitsOutputWithPast(ModelOutput):
    code_switch_logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits: Optional[torch.FloatTensor] = None

def last_not_value(lst, value):
    for item in reversed(lst):
        if item != value:
            return item
    return lst[-1]
 
class GatingMixin:
    def __init__(self, config):
        super().__init__(config)
        num_cs = 4
        try:
            middle_size = config.hidden_size
        except:
            middle_size = config.text_config.hidden_size
        self.code_switch_pre = nn.Linear(middle_size, middle_size, bias=False)
        self.code_switch_relu = nn.ReLU()
        self.code_switch_head = nn.Linear(middle_size, num_cs, bias=False)

    def load_lang_masks(self, tokenizer):
        self.lang_masks = get_lang_masks(self.lm_head.out_features, tokenizer)
        assert((self.lang_masks.int().sum(dim=0) == 1).all())
        self.tok_to_lang = self.lang_masks.int().argmax(dim=0)
        self.should_token_norm = False
        self.token_norm = None
        # self.cache_helper = CacheHelper()

    def compute_loss_by_logits(self, token_ids, code_switch_logits, logits):
        loss = None
        logits = logits.reshape((-1, logits.shape[-1]))
        code_switch_logits = code_switch_logits.reshape((-1, code_switch_logits.shape[-1]))
        with torch.no_grad():
            top_k_t = torch.full((logits.shape[0],), 20).to(logits.device)
            top_p_t = torch.full((logits.shape[0],), 0.95).to(logits.device)
            if not self.should_token_norm:
                labels = chunked_batch_varied_top_k_p_in_logits(logits, self.lang_masks.to(logits.device), top_k_t, top_p_t).to(logits.device)
                # labels = torch.full((logits.shape[0], 4), 0).to(logits.device)
            else:
                labels = chunked_batch_varied_top_k_p_in_logits(logits / self.token_norm, self.lang_masks.to(logits.device), top_k_t, top_p_t).to(logits.device)
                # labels = torch.full((logits.shape[0], 4), 0).to(logits.device)
        if labels is not None:
            # Flatten inputs for loss computation
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # print("code_switch_logits shape", code_switch_logits.shape)
            # print("labels shape", labels.shape)
            loss = loss_fct(code_switch_logits, labels.float())
        return loss

    def compute_logits():
        pass

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            if self.token_norm is None:
                self.token_norm = self.lm_head.weight.norm(dim=1)
            output = super().forward(*args, **kwargs)
        last_hidden_state = output.hidden_states[-1]
        code_switch_x = self.code_switch_pre(last_hidden_state)
        code_switch_x = self.code_switch_relu(code_switch_x)
        code_switch_logits = self.code_switch_head(code_switch_x)

        token_ids = args[0] if len(args) > 0 else kwargs['input_ids']
        loss = None
        if self.training:
            loss = self.compute_loss_by_logits(token_ids, code_switch_logits, output.logits)

        # --- 3. Apply Gating Logic Intervention (Ported from vLLM) ---
        # This logic is applied during generation (when labels are typically not provided)
        # We apply it to the final logits returned by the model.
        logits = output.logits
        # if not self.training and token_ids is not None and logits is not None:
        if False:
            # --- Prepare for intervention ---
            B, S, V = logits.shape
            device = logits.device
            
            # Get the language prediction from the code switch head for the *last* token of each sequence
            # Shape: [B, S, 4] -> [B, 4] (prediction for the last token)
            last_token_lang_logits = code_switch_logits[:, -1, :] # [B, 4]
            
            # --- Determine allowed languages per sequence ---
            # Use sigmoid > 0.5 or take the argmax class
            pred_lang_labels = last_token_lang_logits.sigmoid() >= 0.5 # [B, 4]
            max_idx = last_token_lang_logits.sigmoid().argmax(dim=-1, keepdim=True)  # Shape: [B, 1]
            best_is_true = torch.zeros_like(pred_lang_labels).scatter_(-1, max_idx, 1).bool() # [B, 4]
            pred_lang_labels = pred_lang_labels | best_is_true # [B, 4]
            # print(pred_lang_labels)
            
            # --- Get last token language from input_ids ---
            last_10_token_ids = token_ids[:, -10:]
            last_10_tok_langs = self.tok_to_lang[last_10_token_ids].tolist()
            last_tok_langs = [last_not_value(l, 2) for l in last_10_tok_langs]
            last_tok_langs = torch.tensor(last_tok_langs).cuda()
            # last_token_ids = token_ids[:, -1] # [B]
            # last_tok_langs = self.tok_to_lang[last_token_ids] # [B] - Get language ID of last input token
            
            # --- Refine allowed languages based on last token ---
            original_pred_lang_labels = pred_lang_labels.clone()
            # Allow the language of the last token
            pred_lang_labels.scatter_(dim=1, index=last_tok_langs.unsqueeze(-1), value=True) 
            # No intervention if last token is 'special' (lang id 2)
            mask = (last_tok_langs == 2) 
            pred_lang_labels[mask, :] = True # Allow all if last is special
            
            # --- Determine languages to mask out ---
            langs_to_mask = ~pred_lang_labels # [B, 4]
            langs_to_mask[:, [2, 3]] = False  # Never mask 'special' (2) or 'low-res' (3) categories globally for this check
            
            # --- Check if target language is in top candidates (top-k/p sampling) ---
            # Focus on the logits for the *last* token of each sequence for intervention
            selected_logits = logits[:, -1, :].clone() # [B, V] - Clone to avoid modifying original logits prematurely
            # Prepare top_k and top_p tensors for batch processing
            top_ks = torch.full((B,), 20, device=device, dtype=torch.long) # [B]
            top_ps = torch.full((B,), 0.95, device=device, dtype=torch.float) # [B]
            
            # Check which language categories are present in the top-(20, 0.95) logits for the last token
            logits_lang_labels_sampling = batch_varied_top_k_p_in_logits(
                selected_logits, self.lang_masks, top_ks, top_ps # top_k=20, top_p=0.95
            ) # [B, 4]
            
            # Check which language categories are present in the top-5 OR top-(20, 0.95) logits (OR logic from vLLM)
            logits_lang_labels_topk = batch_varied_top_k_p_in_logits(
                selected_logits, self.lang_masks, torch.full_like(top_ks, 5), torch.full_like(top_ps, 0.999) # top_k=5, top_p=0.999
            ) | batch_varied_top_k_p_in_logits(
                selected_logits, self.lang_masks, top_ks, top_ps # top_k=20, top_p=0.95
            ) # [B, 4]

            # --- Determine if intervention should be applied ---
            # Intervention has effect if sampled tokens (top 20, 0.95) contain disallowed languages
            intervention_has_effect = (logits_lang_labels_sampling & langs_to_mask).any(dim=1) # [B]
            # Target language (predicted) is in the top candidates (top 5 or 20/0.95)
            target_in_topk = (logits_lang_labels_topk & original_pred_lang_labels).any(dim=1) # [B]
            
            # Combine conditions for applying intervention
            apply_intervention = intervention_has_effect & target_in_topk # [B]
            # print("intervention_has_effect", intervention_has_effect, "target_in_topk", target_in_topk, 'langs_to_mask', langs_to_mask, 'logits_lang_labels_sampling', logits_lang_labels_sampling)
            
            # --- Handle special case: only symbol allowed ---
            symbol_true_lang_labels = original_pred_lang_labels.clone()
            symbol_true_lang_labels[:, 2] = True # Pretend symbol is allowed
            only_symbol_allowed = symbol_true_lang_labels.sum(dim=-1) == 1 # [B]
            # Potentially disable intervention if only symbols are allowed (commented out in vLLM)
            # apply_intervention = apply_intervention & ~only_symbol_allowed

            # --- Apply the intervention ---
            if apply_intervention.any():
                # Get the logits that will be modified (for the last token of sequences needing intervention)
                indices_to_modify = apply_intervention.nonzero(as_tuple=True)[0] # [num_interventions]
                logits_to_modify = selected_logits[indices_to_modify] # [num_interventions, V]
                
                # Get the specific language masks for the sequences needing intervention
                final_langs_to_mask = langs_to_mask[indices_to_modify] # [num_interventions, 4]
                # print('final_langs_to_mask', final_langs_to_mask)
                
                # Create the final vocabulary mask: [num_interventions, 4] @ [4, V] -> [num_interventions, V]
                vocab_mask = torch.matmul(final_langs_to_mask.float(), self.lang_masks.float()).bool() # [num_interventions, V]
                
                # Apply the mask: set disallowed logits to -inf
                masked_logits = logits_to_modify.masked_fill(vocab_mask, float('-inf')) # [num_interventions, V]
                
                # Write the modified logits back to the main logits tensor
                # Modify the logits for the *last token* of the relevant sequences
                logits[indices_to_modify, -1, :] = masked_logits # Update logits tensor in place

        '''if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output'''
        return CsLogitsOutputWithPast(
            loss=loss,
            logits=output.logits,
            code_switch_logits=code_switch_logits,
            hidden_states=output.hidden_states
        )

class Qwen3Gating(GatingMixin, Qwen3ForCausalLM):
    pass

class Qwen3MoeGating(GatingMixin, Qwen3MoeForCausalLM):
    pass

class LlamaGating(GatingMixin, LlamaForCausalLM):
    pass

class GemmaGating(GatingMixin, Gemma3ForCausalLM):
    pass

class OlmoGating(GatingMixin, Olmo2ForCausalLM):
    pass

# from transformers import GptOssForCausalLM
# class GptOssGating(GatingMixin, GptOssForCausalLM):
#     pass