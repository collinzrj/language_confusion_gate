from collections.abc import Iterable
from typing import Optional, Union

import torch, unicodedata, os
from torch import nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import Pooler, PoolingType, SimplePooler
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.gemma3 import Gemma3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.model_executor.sampling_metadata import SamplingMetadata
# from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler, Sampler, _apply_top_k_top_p
from vllm.model_executor.layers.sampler import *
from vllm.model_executor.layers.sampler import _apply_min_tokens_penalty, _apply_top_k_top_p, _apply_min_p, _sample, _build_sampler_output
import torch.distributed as dist
from typing import List, Tuple
from line_profiler import profile
from vllm.utils import make_tensor_with_pad, is_pin_memory_available
from .tok_contains_partial_cjk import contain_partial_cjk

def last_not_value(lst, value):
    for item in reversed(lst):
        if item != value:
            return item
    return lst[-1]

LAST_TOK_WINDOW = 20
USE_LOG = False

## this expects logits_batch to be processed such that non topk topp tokens are -inf
@torch.no_grad()
def batch_varied_top_k_p_in_logits_opt(
    logits_batch: torch.Tensor,      # [N, V]
    lang_masks:   torch.Tensor,      # [C, V] bool
) -> torch.Tensor:
    # Memory-efficient approach: avoid creating large intermediate tensors
    # Check which logits are not -inf (keep_mask)
    keep_mask = ~torch.isneginf(logits_batch)  # [N, V]
    
    # Use einsum for efficient computation without expanding dimensions
    # This computes: for each (n, c), check if any v where keep_mask[n,v] & lang_masks[c,v]
    # Result: [N, C] where any_lang[n,c] = True if any valid token in batch n belongs to language c
    any_lang = torch.einsum('nv,cv->nc', keep_mask.float(), lang_masks.float()) > 0
    
    return any_lang

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

tok_path = os.environ['TOK_PATH']
tokenizer = AutoTokenizer.from_pretrained(tok_path)

def get_lang_masks(vocab_size):
    cj_labels, latin_labels, special_labels, lowres_labels = [], [], [], []
    for tok in range(vocab_size):
        tok_text = tokenizer.decode(tok)
        cj_label = contains_cj(tok_text) or contain_partial_cjk(tokenizer, tok)
        latin_label = contains_latin(tok_text)
        special_label = not cj_label and contains_only_special(tok_text)
        cj_labels.append(cj_label)
        latin_labels.append(latin_label)
        special_labels.append(special_label)
        lowres_labels.append((cj_label == 0) and (latin_label == 0) and (special_label == 0))
    current_device = torch.cuda.current_device()
    lang_masks = torch.tensor([cj_labels, latin_labels, special_labels, lowres_labels]).to(current_device)
    print('before', lang_masks[0].sum(), lang_masks[1].sum(), lang_masks[2].sum(), lang_masks[3].sum())
    control_toks_mask = get_control_toks(vocab_size)
    lang_masks[:, control_toks_mask] = 0
    lang_masks[2, control_toks_mask] = 1
    print('after', lang_masks[0].sum(), lang_masks[1].sum(), lang_masks[2].sum(), lang_masks[3].sum())
    return lang_masks

def print_logits(logits):
    next_token_logits = logits[-1, :]
    top_k_tokens = torch.topk(next_token_logits, 20, dim=-1).indices
    for i, token_id in enumerate(top_k_tokens):
        token_str = tokenizer.decode(token_id.item())
        print(f"Top {i+1}: {[token_str]}", end = ', ')

def get_control_toks(vocab_size):
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
    special_toks.append(tokenizer.encode('assistant', add_special_tokens=False)[0])
    for c in 'ABCDEFG':
        ctok1 = tokenizer.encode(c, add_special_tokens=False)
        special_toks.append(ctok1[0])
        ctok2 = tokenizer.encode(' ' + c, add_special_tokens=False)
        if len(ctok2) == 1:
            special_toks.append(ctok2[0])
    for tok in special_toks:
        control_toks_mask[tok] = 1
    control_toks_mask = torch.tensor(control_toks_mask).bool()
    return control_toks_mask

def get_tokens_tensor(sampling_metadata, vocab_size, device):
    pin_memory = is_pin_memory_available()
    output_tokens: List[array] = []
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        if (seq_group.is_prompt
                and sampling_params.prompt_logprobs is not None):
            prefill_len = len(seq_group.prompt_logprob_indices)
            output_tokens.extend(
                array(VLLM_TOKEN_ID_ARRAY_TYPE)
                for _ in range(prefill_len))
        if seq_group.do_sample:
            for seq_id in seq_ids:
                seq_data = seq_group.seq_data[seq_id]
                output_tokens.append(seq_data.output_token_ids_array)
    output_t = make_tensor_with_pad(
        output_tokens,
        vocab_size,
        device='cpu',
        dtype=torch.int64,
        pin_memory=pin_memory,
    ).to(device)
    return output_t


class GatingMixin:
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # print("QQQ", GatingMixin.mro())
        # import inspect; print(f"LYH File: {__file__}, Line: {inspect.currentframe().f_lineno}")
        super().__init__(vllm_config=vllm_config)
        print("after init")
        self.packed_modules_mapping = {
            "qkv_proj": [
                "q_proj",
                "k_proj",
                "v_proj",
            ],
            "gate_up_proj": [
                "gate_proj",
                "up_proj",
            ],
        }
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config

        self.code_switch_pre = ColumnParallelLinear(config.hidden_size, config.hidden_size, quant_config=quant_config, return_bias=False)
        self.code_switch_head = RowParallelLinear(config.hidden_size, 4, quant_config=quant_config, return_bias=False)
        self._lang_head = nn.Sequential(
            self.code_switch_pre,
            nn.ReLU(),
            self.code_switch_head,
        )
        lang_masks = get_lang_masks(self.logits_processor.vocab_size)
        print('before', lang_masks[0].sum(), lang_masks[1].sum(), lang_masks[2].sum(), lang_masks[3].sum())
        control_toks_mask = get_control_toks(self.logits_processor.vocab_size)
        lang_masks[:, control_toks_mask] = 0
        lang_masks[2, control_toks_mask] = 1
        print('after', lang_masks[0].sum(), lang_masks[1].sum(), lang_masks[2].sum(), lang_masks[3].sum())
        assert((lang_masks.int().sum(dim=0) == 1).all())
        tok_to_lang = lang_masks.int().argmax(dim=0)
        self.tok_to_lang = tok_to_lang
        self.lang_masks = lang_masks

        self.sampler = GateSampler(self.tok_to_lang, self.lang_masks)

    # @profile
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        # import inspect; print(f"LYH File: {__file__}, Line: {inspect.currentframe().f_lineno}")
        # print(type(self.sampler))

        device = torch.cuda.current_device()
        batch_logits = self.logits_processor(
            self.lm_head, hidden_states, sampling_metadata
        )

        # return batch_logits

        original_hidden_state = hidden_states.clone()
        if sampling_metadata is not None:
            hidden_states = hidden_states.index_select(0, sampling_metadata.selected_token_indices)

        hidden_lang = self.code_switch_pre.quant_method.apply(self.code_switch_pre, hidden_states, bias=self.code_switch_pre.bias)

        hidden_lang = nn.functional.relu(hidden_lang)
        batch_lang_logits = self.code_switch_head.quant_method.apply(self.code_switch_head, hidden_lang, bias=self.code_switch_head.bias)

        return [batch_logits, batch_lang_logits]

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loaded_params = super().load_weights(weights)
        # print("loaded_params", loaded_params)
        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=["lm_head."])
        # TODO: a bit hacky, need to fix this
        params_to_ignore = {'code_switch_head.bias', 'code_switch_pre.bias'}
        return loader.load_weights(weights).union(loaded_params).union(params_to_ignore)

    def sample(
        self,
        logits: Optional[torch.Tensor],
        # batch_lang_logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    @classmethod
    def is_backend_compatible(cls) -> bool:
        return True  # Or implement actual logic if needed

class GateSampler(Sampler):
    def __init__(self, tok_to_lang, lang_masks):
        super().__init__()
        self.tok_to_lang = tok_to_lang
        # tokens come from sampling_tensor are padded with vocab_size, so set it to symbol
        self.tok_to_lang = torch.cat((self.tok_to_lang, torch.tensor([2], device=self.tok_to_lang.device)), dim=0)
        self.lang_masks = lang_masks

    # @profile
    def apply_lang_mask_logprobNone_fast(
        self,
        logits_topk_topp: torch.Tensor,
        logits_k5_p999: torch.Tensor,
        logits_k20_p95: torch.Tensor,
        batch_lang_logits: torch.Tensor,
        output_tokens_tensor: torch.Tensor,
        full_logits = None
    ) -> Optional[SamplerOutput]:
        original_batch_logits = logits_topk_topp
        combine_batch_logits = logits_topk_topp.clone()
        mask_k5_p999 = ~torch.isneginf(logits_k5_p999)
        mask_k20_p95 = ~torch.isneginf(logits_k20_p95)
        combine_batch_logits[mask_k5_p999] = logits_k5_p999[mask_k5_p999]
        combine_batch_logits[mask_k20_p95] = logits_k5_p999[mask_k20_p95]
        # combine_batch_logits = torch.where(mask_k5_p999, logits_k5_p999, combine_batch_logits)
        # combine_batch_logits = torch.where(mask_k20_p95, logits_k20_p95, combine_batch_logits)
        
        if output_tokens_tensor.shape[1] == 0:
            return original_batch_logits
        device = torch.cuda.current_device()
        vocab_size = logits_topk_topp.shape[-1]
        has_output_tokens = (output_tokens_tensor != vocab_size).any(dim=-1)
        # it left-padding, so can't take from the last
        # last_token_ids_tensor = output_tokens_tensor[:, -LAST_TOK_WINDOW:]
        last_token_ids_tensor = output_tokens_tensor
        tok_langs_tensor = self.tok_to_lang[last_token_ids_tensor]

        # # GPU 上处理 tok_langs_tensor 生成 last_tok_lang_arr
        # for i, (seq_group, idx_within_group, logit_index_in_selected) in enumerate(indices_mapping):
        #     tok_langs = tok_langs_tensor[i]
        #     last_tok_lang = last_not_value(tok_langs, 2)
        #     last_tok_lang_arr.append(last_tok_lang)

        seq_len = tok_langs_tensor.shape[1]
        mask = (tok_langs_tensor != 2)
        has_non_value = mask.any(dim=1)

        rev_mask = mask.flip(dims=[1])
        offsets = rev_mask.float().argmax(dim=1)
        last_pos = seq_len - 1 - offsets

        # 修正全 False 行的 last_pos
        last_pos = torch.where(has_non_value, last_pos, torch.full_like(last_pos, seq_len - 1))
        last_pos = torch.clamp(last_pos, 0, seq_len - 1)

        # 安全 gather
        batch_idx = torch.arange(tok_langs_tensor.size(0), device=tok_langs_tensor.device)
        last_tok_lang_tensor = tok_langs_tensor[batch_idx, last_pos]

        last_tok_lang_arr = list(last_tok_lang_tensor)

        # 2. Perform batched calculations
        # Check which language categories are present in the sampling-filtered logits
        # logits_lang_labels_sampling = batch_varied_top_k_p_in_logits(
        #     selected_logits, self.lang_masks, top_ks, top_ps
        # )
        last_tok_langs = torch.stack(last_tok_lang_arr)
        logits_lang_labels_sampling = batch_varied_top_k_p_in_logits_opt(logits_topk_topp, self.lang_masks)
        logits_lang_labels_topk = batch_varied_top_k_p_in_logits_opt(logits_k5_p999, self.lang_masks) | batch_varied_top_k_p_in_logits_opt(logits_k20_p95, self.lang_masks)

        pred_lang_labels = batch_lang_logits.sigmoid() >= 0.5
        max_idx = batch_lang_logits.sigmoid().argmax(dim=-1, keepdim=True)
        best_is_true = torch.zeros_like(pred_lang_labels).scatter_(-1, max_idx, 1).bool()
        pred_lang_labels = pred_lang_labels | best_is_true

        original_pred_lang_labels = pred_lang_labels.clone()
        pred_lang_labels.scatter_(dim=1, index=last_tok_langs.unsqueeze(-1), value=True)
        # no intervention if last all symbol
        mask = (last_tok_langs == 2)
        pred_lang_labels[mask, :] = True

        # Determine which languages to mask out
        langs_to_mask = ~pred_lang_labels
        langs_to_mask[:, 2:4] = False  # Never mask 'special' or 'low-res' categories

        # 3. Determine which sequences need intervention
        intervention_has_effect = (logits_lang_labels_sampling & langs_to_mask).any(dim=1)
        target_in_topk = (logits_lang_labels_topk & original_pred_lang_labels).any(dim=1)
        apply_intervention = intervention_has_effect & target_in_topk & has_output_tokens

        if USE_LOG:
            for i in range(len(logits_lang_labels_sampling)):
                # log when has zh but last is not zh
                if (logits_lang_labels_sampling[i][0] == True and last_tok_langs[i] != 0) or (logits_lang_labels_sampling[i][1] == True and last_tok_langs[i] != 1 and last_tok_langs[i] != 1):
                # if True:
                    print("!" * 30)
                    print("FAST")
                    print("apply intervention", apply_intervention[i])
                    print("last_tok_langs", last_tok_langs[i])
                    # print("tok_langs_tensor", tok_langs_tensor[i])
                    print("last_token_ids_tensor text", tokenizer.decode(last_token_ids_tensor[i]))
                    print("last_pos", last_pos[i])
                    # print([tokenizer.decode(token_ids_arr[i].get_token_ids())])
                    print("intervention_has_effect", intervention_has_effect[i])
                    print("target_in_topk", target_in_topk[i])
                    print("logits_lang_labels_sampling", logits_lang_labels_sampling[i])
                    print("logits_lang_labels_topk", logits_lang_labels_topk[i])
                    # print("only_symbol_allowed", only_symbol_allowed[i])
                    print("original_pred_lang_labels", original_pred_lang_labels[i])
                    print("selected_lang_logits", batch_lang_logits[i].sigmoid())
                    print("langs_to_mask", langs_to_mask[i])
                    scores = torch.softmax(original_batch_logits[i] / 0.7, dim=-1)
                    for tok in original_batch_logits[i].topk(k=10).indices.tolist():
                        print(tok, [tokenizer.decode(tok), scores[tok].item()], end=', ')
                    print()
                    for tok in combine_batch_logits[i].topk(k=10).indices.tolist():
                        print(tok, [tokenizer.decode(tok), combine_batch_logits[i][tok].item()], end=', ')
                    print()
                    if full_logits is not None:
                        for tok in full_logits[i].topk(k=10).indices.tolist():
                            print(tok, [tokenizer.decode(tok), full_logits[i][tok].item()], end=', ')
                        print()

        if apply_intervention.any():
            vocab_mask = torch.matmul(langs_to_mask.float(), self.lang_masks.float()).bool()
            vocab_mask = vocab_mask & apply_intervention.unsqueeze(1)
            combine_batch_logits = combine_batch_logits.masked_fill(vocab_mask, -torch.inf)
            # if no intervention, use original logits, otherwise use more tokens to ensure lowres is included
            original_batch_logits[apply_intervention] = combine_batch_logits[apply_intervention]
            # original_batch_logits = torch.where(apply_intervention.unsqueeze(1), combine_batch_logits, original_batch_logits)

        return original_batch_logits
    
    # @profile
    def forward(
        self,
        logits: torch.Tensor,
        # batch_lang_logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Single-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Pythonize sampling result & logprobs tensor

        Multi-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Defer Pythonization of sampling result & logprobs
          tensor
        * Encapsulate arguments required for deferred Pythonization
          in the :class:`SamplerOutput` structure

        Args:
            logits: (num_tokens, vocab_size).
            sampling_metadata: Metadata for sampling.
        """
        logits, batch_lang_logits = logits
        assert logits is not None
        _, vocab_size = logits.shape
        device = torch.cuda.current_device()
        self._do_penalties = True

        # Prepare sampling tensors with pinned memory to avoid blocking.
        if not sampling_metadata.reuse_sampling_tensors:
            self._init_sampling_tensors(logits, sampling_metadata)
        elif self._do_penalties:
            # In this case, the sampling tensors logic depends on
            # "output_tokens" of a sequence. As a result, we cannot
            # reuse sampling tensors, since "output_tokens" changes
            # between decode runs.
            self._init_sampling_tensors(logits, sampling_metadata)

        assert self._sampling_tensors is not None
        sampling_tensors = self._sampling_tensors
        do_penalties = self._do_penalties
        do_top_p_top_k = self._do_top_p_top_k
        do_min_p = self._do_min_p

        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = apply_penalties(logits, sampling_tensors.prompt_tokens,
                                     sampling_tensors.output_tokens,
                                     sampling_tensors.presence_penalties,
                                     sampling_tensors.frequency_penalties,
                                     sampling_tensors.repetition_penalties)

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k and flashinfer_top_k_top_p_sampling is None:
            # logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
            #                             sampling_tensors.top_ks)
            # logits2 = _apply_top_k_top_p(logits, torch.full(sampling_tensors.top_ps.size(), 0.999, device=logits.device),
            #                             torch.full(sampling_tensors.top_ks.size(), 5, device=logits.device))
            # logits3 = _apply_top_k_top_p(logits, torch.full(sampling_tensors.top_ps.size(), 0.95, device=logits.device),
            #                             torch.full(sampling_tensors.top_ks.size(), 20, device=logits.device))
            p_k_tuple = (
                (sampling_tensors.top_ps, sampling_tensors.top_ks),
                # (torch.full(sampling_tensors.top_ps.size(), 0.999, device=logits.device), torch.full(sampling_tensors.top_ks.size(), 5, device=logits.device)),
                # (torch.full(sampling_tensors.top_ps.size(), 0.95, device=logits.device), torch.full(sampling_tensors.top_ks.size(), 20, device=logits.device)),
                (torch.full(sampling_tensors.top_ps.size(), 0.95, device=logits.device), torch.full(sampling_tensors.top_ks.size(), 20, device=logits.device)),
                (torch.full(sampling_tensors.top_ps.size(), 0.999, device=logits.device), torch.full(sampling_tensors.top_ks.size(), 5, device=logits.device)),
            )
            [logits1, logits2, logits3] = _apply_top_k_top_p_new(logits, p_k_tuple)
            if sampling_tensors.output_tokens.shape[0] == 0:
                output_tokens_t = get_tokens_tensor(sampling_metadata, logits.shape[-1], logits.device)
            
            prompt_logprob_all_None = True
            for seq_group in sampling_metadata.seq_groups:
                prompt_logprob_all_None = prompt_logprob_all_None and (seq_group.sampling_params.prompt_logprobs is None)
            logits = self.apply_lang_mask_logprobNone_fast(logits1, logits2, logits3, batch_lang_logits, output_tokens_t, full_logits=logits)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        maybe_deferred_sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
        )

        if self.include_gpu_probs_tensor:
            # Since we will defer sampler result Pythonization,
            # preserve GPU-side tensors in support of later
            # deferred pythonization of logprobs
            assert maybe_sampled_tokens_tensor is not None
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            # Since Pythonization has already happened, don't preserve
            # GPU-side tensors.
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs = None
        sample_logprobs = None
        if not sampling_metadata.skip_sampler_cpu_output:
            # Pythonize logprobs now (GPU -> CPU); do not defer.
            assert not isinstance(maybe_deferred_sample_results,
                                  SampleResultArgsType)
            prompt_logprobs, sample_logprobs = get_logprobs(
                logprobs, sampling_metadata, maybe_deferred_sample_results)

        return _build_sampler_output(
            maybe_deferred_sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
            on_device_tensors=on_device_tensors,
            skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)

def apply_top_p_k_small(
    logits_sort: torch.Tensor,
    logits_idx: torch.Tensor,
    probs_sum: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
):
    # Apply top-k.
    top_k_mask = logits_sort.size(1) - k.to(torch.long)
    # Get all the top_k values.
    top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
    top_k_mask = logits_sort < top_k_mask
    logits_sort.masked_fill_(top_k_mask, -float("inf"))

    # Apply top-p.
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = torch.empty_like(logits_sort).scatter_(dim=-1,
                                                    index=logits_idx,
                                                    src=logits_sort)
    
    return logits

def _apply_top_k_top_p_new(
    logits: torch.Tensor,
    p_k_lst: Tuple[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    # LYH 合并计算后，复用的部分
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)

    logits_lst = []
    for (p, k) in p_k_lst:
        logits = apply_top_p_k_small(logits_sort.clone(), logits_idx, probs_sum, p, k)
        logits_lst.append(logits)

    return logits_lst

class Qwen3Gating(GatingMixin, Qwen3ForCausalLM):
    pass

class Qwen3MoeGating(GatingMixin, Qwen3MoeForCausalLM):
    pass

class LlamaGating(GatingMixin, LlamaForCausalLM):
    pass

class GemmaGating(GatingMixin, Gemma3ForCausalLM):
    # @profile
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        # import inspect; print(f"LYH File: {__file__}, Line: {inspect.currentframe().f_lineno}")
        # print(type(self.sampler))

        device = torch.cuda.current_device()

        # Notice: this is the only line different from the above
        batch_logits = self.logits_processor(self.model.embed_tokens, hidden_states, sampling_metadata)

        # return batch_logits

        original_hidden_state = hidden_states.clone()
        if sampling_metadata is not None:
            hidden_states = hidden_states.index_select(0, sampling_metadata.selected_token_indices)

        hidden_lang = self.code_switch_pre.quant_method.apply(self.code_switch_pre, hidden_states, bias=self.code_switch_pre.bias)

        hidden_lang = nn.functional.relu(hidden_lang)
        batch_lang_logits = self.code_switch_head.quant_method.apply(self.code_switch_head, hidden_lang, bias=self.code_switch_head.bias)

        return [batch_logits, batch_lang_logits]