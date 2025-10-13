"""
2025.5.11
2025.5.9
4.52.4
0.18.1
__UNSLOTH_VERSIONING__
"""

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import importlib.util
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import List, Dict, Tuple, Optional, Any, Callable
import math


import os
import torch
from unsloth_zoo.loss_utils import fused_linear_cross_entropy

if UNSLOTH_STUDIO_ENABLED:
    from unsloth_zoo.loss_utils import fast_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 24, 'combo_kernels': False, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': False, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': True}

from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.mllama.modeling_mllama import (F, math, Callable, List, Optional, Tuple, Union, torch, nn, ACT2FN, Cache, GenerationMixin, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, ALL_ATTENTION_FUNCTIONS, PreTrainedModel, Unpack, can_return_tuple, MllamaConfig, MllamaTextConfig, MllamaVisionConfig, logger, __name__, MllamaPreTrainedModel, MllamaTextModel, KwargsForCausalLM, MllamaForCausalLM, MllamaModel, MllamaForConditionalGeneration)

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def _prepare_cross_attention_mask(cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min)

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = ((cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None])
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask!=torch.finfo(cross_attention_mask.dtype).min, full_text_row_masked_out_mask

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def _prepare_aspect_ratio_attention_mask(aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(-1, -2) * torch.finfo(dtype).min
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask!=torch.finfo(attention_mask.dtype).min

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaPrecomputedAspectRatioEmbedding_forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
    embeddings = self.embedding(aspect_ratio_ids)
    embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

    if self.is_gated:
        embeddings = embeddings * self.gate.tanh()

    hidden_state = hidden_state + embeddings
    return hidden_state

class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        return MllamaPrecomputedAspectRatioEmbedding_forward(self, hidden_state, aspect_ratio_ids)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaPrecomputedPositionEmbedding_forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
    # position embeddings
    gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
    hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

    # precomputed tile position embeddings
    tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
    batch_size = hidden_state.shape[0]
    tile_position_embedding = tile_position_embedding.reshape(
        batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
    )
    gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
    hidden_state = hidden_state + gated_tile_position_embedding

    return hidden_state

class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size
        )

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        return MllamaPrecomputedPositionEmbedding_forward(self, hidden_state, aspect_ratio_ids)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def MllamaVisionMLP_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = self.fc2(hidden_states)
    return hidden_states

class MllamaVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return MllamaVisionMLP_forward(self, hidden_states)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@torch.compiler.disable(recursive = False)
def MllamaVisionAttention_forward(
    self,
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    query = self.q_proj(hidden_state)
    key = self.k_proj(hidden_state)
    value = self.v_proj(hidden_state)

    batch_size, q_seq_len, _ = query.shape
    _, kv_seq_len, _ = key.shape

    query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    attention_interface: Callable = eager_attention_forward

    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and output_attentions:
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, q_seq_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights

class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = 1

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return MllamaVisionAttention_forward(self, hidden_state, attention_mask, output_attentions, **kwargs)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaTextRMSNorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

class MllamaTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MllamaTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return MllamaTextRMSNorm_forward(self, hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@torch.compiler.disable(recursive = False)
def MllamaTextCrossAttention_forward(
    self,
    hidden_states: torch.Tensor,
    cross_attention_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    query_states = self.q_norm(query_states)

    if cross_attention_states is not None:
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_norm(key_states)
        if past_key_value is not None:
            # if we have a new image + new tokens, we only computed key_states on that new image
            # we still update the cross key states, past_image, new_image. And use it!
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
    elif cache_position[0] != 0:
        key_states, value_states = (
            past_key_value.key_cache[self.layer_idx],
            past_key_value.value_cache[self.layer_idx],
        )
    else:
        raise ValueError(
            "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
        )

    attention_interface: Callable = eager_attention_forward

    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and output_attentions:
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Optional[MllamaTextConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return MllamaTextCrossAttention_forward(self, hidden_states, cross_attention_states, past_key_value, attention_mask, output_attentions, use_cache, cache_position, **kwargs)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def apply_rotary_pos_emb(q, k, cos, sin,  unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compiler.disable(recursive = False)
def MllamaTextSelfAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: torch.Tensor,
    output_attentions: bool = False,
    use_cache: bool = False,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward

    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and output_attentions:
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

class MllamaTextSelfAttention(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        return MllamaTextSelfAttention_forward(self, hidden_states, attention_mask, position_embeddings, output_attentions, use_cache, past_key_value, cache_position, **kwargs)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def MllamaTextMLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class MllamaTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Ignore copy
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return MllamaTextMLP_forward(self, x)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def MllamaRotaryEmbedding_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class MllamaRotaryEmbedding(nn.Module):
    def __init__(self, config: MllamaTextConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling["rope_type"]
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


    def forward(self, x, position_ids):
        return MllamaRotaryEmbedding_forward(self, x, position_ids)


@torch.compiler.disable(recursive = False)
def MllamaForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cross_attention_states: Optional[torch.LongTensor] = None,
    cross_attention_mask: Optional[torch.LongTensor] = None,
    full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    cross_attention_states (`torch.FloatTensor`, *optional*):
        Output of the vision model, used for cross-attention. This tensor contains the processed image features that
        the language model will attend to.
    cross_attention_mask (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*):
        Cross-attention mask to control the interaction between text tokens and image tiles.
        This 4D tensor defines which image tiles each text token should attend to.

        For each text token (in seq_length):
        - 1 indicates the token **should attend** to the corresponding image tile
        - 0 indicates the token **should not attend** to the corresponding image tile
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    full_text_row_masked_out_mask (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
        A tuple containing two tensors that mask out rows in the cross-attention mechanism:
        - The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
          A value of 0 indicates that the corresponding text token's entire row in the cross-attention
          matrix should be masked out (all image tokens ignored).
        - The second tensor has the same shape and is used internally to apply the masking during
          the forward pass of cross-attention layers.
        This mask is derived from the cross_attention_mask and is used to handle cases where a text token
        should not attend to any image token.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, MllamaForCausalLM

    >>> model = MllamaForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
    >>> tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision")

    >>> prompt = "If I had to write a haiku, it would be:"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
    >>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    >>> print(result)
    If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
    I love the idea of snowflakes gently falling, each one
    ```
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        cross_attention_states=cross_attention_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cross_attention_mask=cross_attention_mask,
        full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    n_items = (kwargs).get("num_items_in_batch", None) or (kwargs).get("n_items", None)
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if labels is None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    elif (UNSLOTH_STUDIO_ENABLED and NOT_RETURN_LOGITS and labels is not None) and not requires_grad_:
        loss = fast_linear_cross_entropy(
            hidden_states        = hidden_states[:, slice_indices, :],
            lm_head              = self.lm_head,
            labels               = labels,
            num_items_in_batch   = n_items,
            logit_softcapping    = None if () == () else (),
            logit_scale_multiply = None if () == () else (),
            logit_scale_divide   = None if () == () else (),
        )
    elif (() == () and () == ()) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if () == () else (),
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        def _compiled_loss_function(
            output_logits : torch.Tensor,
            output_labels : torch.Tensor,
            logit_scale_multiply : float = 0,
            logit_scale_divide : float = 0,
            logit_softcapping : float = 0,
            vocab_size : int = 0,
            n_items : int = 0,
        ):
            device = output_logits.device
            if logit_scale_multiply != 0:
                output_logits = output_logits * logit_scale_multiply
            if logit_scale_divide != 0:
                output_logits = output_logits / logit_scale_divide
            if logit_softcapping != 0:
                output_logits = output_logits / logit_softcapping
                output_logits = torch.tanh(output_logits)
                output_logits = output_logits * logit_softcapping
    
            shift_logits = output_logits
            shift_labels = torch.empty_like(output_labels, device = device)
            shift_labels[..., :-1] = output_labels[..., 1:]
            shift_labels[..., -1] = -100
            # shift_logits = output_logits[..., :-1, :].float().contiguous()
            # shift_labels = output_labels[..., 1:].contiguous()
    
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
    
            n_chunks = int(math.ceil((vocab_size / 262144) * 8))
            if requires_grad_: n_chunks += 2
            __shift_logits = torch.chunk(shift_logits, n_chunks, dim = 0)
            __shift_labels = torch.chunk(shift_labels, n_chunks, dim = 0)
            loss = 0.0
            for (_shift_logits, _shift_labels) in zip(__shift_logits, __shift_labels):
                loss += torch.nn.functional.cross_entropy(
                    input  = _shift_logits.float().contiguous(),
                    target = _shift_labels.contiguous(),
                    reduction = 'sum',
                )
            pass
            if n_items != 0:
                loss = loss / n_items
            else:
                loss = loss / (shift_labels != -100).sum()
            return loss
        pass
        _compiled_loss_function = torch.compile(
            _compiled_loss_function,
            fullgraph = False,
            dynamic = True,
            options = torch_compile_options,
        )
        torch._dynamo.mark_dynamic(logits, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        loss = _compiled_loss_function(
            output_logits        = logits,
            output_labels        = labels,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () not in (None, (),) else 0,
            vocab_size           = (self.vocab_size),
            n_items              = n_items if n_items is not None else 0,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), self.vocab_size, **kwargs)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class MllamaForCausalLM(MllamaPreTrainedModel, GenerationMixin):
    config_class = MllamaTextConfig
    _supports_static_cache = True  # only the LLM without cross attn can do compile
    base_model_prefix = "language_model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.model = MllamaTextModel._from_config(self.text_config)
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return MllamaForCausalLM_forward(self, input_ids, attention_mask, position_ids, cross_attention_states, cross_attention_mask, full_text_row_masked_out_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, logits_to_keep, **kwargs)


@torch.compiler.disable(recursive = False)
@can_return_tuple
def MllamaForConditionalGeneration_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    aspect_ratio_mask: Optional[torch.Tensor] = None,
    aspect_ratio_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_mask: Optional[torch.Tensor] = None,
    cross_attention_states: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    aspect_ratio_mask (`torch.Tensor` of shape `(batch_size, max_num_images, max_num_tiles)`, *optional*):
        Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:

        - 1 for tiles that are **not masked**,
        - 0 for tiles that are **masked**.
    aspect_ratio_ids (`torch.Tensor` of shape `(batch_size, max_num_images)`, *optional*):
        Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
        These ids correspond to indices in the model's list of supported aspect ratios, offset by 1.

        For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:
        - An image with aspect ratio [1, 1] would have ID 1
        - An image with aspect ratio [1, 2] would have ID 2
        - An image with aspect ratio [2, 1] would have ID 3

        The id 0 is reserved for padding (i.e., no image).

        If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its `aspect_ratio_id` would be 2.
    cross_attention_mask (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*):
        Cross-attention mask to control the interaction between text tokens and image tiles.
        This 4D tensor defines which image tiles each text token should attend to.

        For each text token (in seq_length):
        - 1 indicates the token **should attend** to the corresponding image tile
        - 0 indicates the token **should not attend** to the corresponding image tile
    cross_attention_states (`torch.FloatTensor`, *optional*):
        Output of the vision model, used for cross-attention. This tensor contains the processed image features that
        the language model will attend to.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

    >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
    >>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
    >>> processor = AutoProcessor.from_pretrained(checkpoint)

    >>> prompt = "<|image|>If I had to write a haiku for this one"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> output = model.generate(**inputs, max_new_tokens=15)

    >>> prompt_len = inputs.input_ids.shape[-1]
    >>> generated_ids = output[:, prompt_len:]
    >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    >>> print(generated_text)
    [', it would be:.\\nA stop sign in Chinatown.\\n']
    ```
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        aspect_ratio_mask=aspect_ratio_mask,
        aspect_ratio_ids=aspect_ratio_ids,
        cross_attention_mask=cross_attention_mask,
        cross_attention_states=cross_attention_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    n_items = (kwargs).get("num_items_in_batch", None) or (kwargs).get("n_items", None)
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if labels is None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    elif (UNSLOTH_STUDIO_ENABLED and NOT_RETURN_LOGITS and labels is not None) and not requires_grad_:
        loss = fast_linear_cross_entropy(
            hidden_states        = hidden_states[:, slice_indices, :],
            lm_head              = self.lm_head,
            labels               = labels,
            num_items_in_batch   = n_items,
            logit_softcapping    = None if () == () else (),
            logit_scale_multiply = None if () == () else (),
            logit_scale_divide   = None if () == () else (),
        )
    elif (() == () and () == ()) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if () == () else (),
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        def _compiled_loss_function(
            output_logits : torch.Tensor,
            output_labels : torch.Tensor,
            logit_scale_multiply : float = 0,
            logit_scale_divide : float = 0,
            logit_softcapping : float = 0,
            vocab_size : int = 0,
            n_items : int = 0,
        ):
            device = output_logits.device
            if logit_scale_multiply != 0:
                output_logits = output_logits * logit_scale_multiply
            if logit_scale_divide != 0:
                output_logits = output_logits / logit_scale_divide
            if logit_softcapping != 0:
                output_logits = output_logits / logit_softcapping
                output_logits = torch.tanh(output_logits)
                output_logits = output_logits * logit_softcapping
    
            shift_logits = output_logits
            shift_labels = torch.empty_like(output_labels, device = device)
            shift_labels[..., :-1] = output_labels[..., 1:]
            shift_labels[..., -1] = -100
            # shift_logits = output_logits[..., :-1, :].float().contiguous()
            # shift_labels = output_labels[..., 1:].contiguous()
    
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
    
            n_chunks = int(math.ceil((vocab_size / 262144) * 8))
            if requires_grad_: n_chunks += 2
            __shift_logits = torch.chunk(shift_logits, n_chunks, dim = 0)
            __shift_labels = torch.chunk(shift_labels, n_chunks, dim = 0)
            loss = 0.0
            for (_shift_logits, _shift_labels) in zip(__shift_logits, __shift_labels):
                loss += torch.nn.functional.cross_entropy(
                    input  = _shift_logits.float().contiguous(),
                    target = _shift_labels.contiguous(),
                    reduction = 'sum',
                )
            pass
            if n_items != 0:
                loss = loss / n_items
            else:
                loss = loss / (shift_labels != -100).sum()
            return loss
        pass
        _compiled_loss_function = torch.compile(
            _compiled_loss_function,
            fullgraph = False,
            dynamic = True,
            options = torch_compile_options,
        )
        torch._dynamo.mark_dynamic(logits, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        loss = _compiled_loss_function(
            output_logits        = logits,
            output_labels        = labels,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () not in (None, (),) else 0,
            vocab_size           = (self.config.text_config.vocab_size),
            n_items              = n_items if n_items is not None else 0,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), self.config.text_config.vocab_size, **kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class MllamaForConditionalGeneration(MllamaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_model": "model.vision_model",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _supports_quantized_cache = False  # quant cache not supported in encoder-decoder setting
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MllamaConfig):
        super().__init__(config)
        self.model = MllamaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_model(self):
        return self.model.vision_model


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return MllamaForConditionalGeneration_forward(self, input_ids, pixel_values, aspect_ratio_mask, aspect_ratio_ids, attention_mask, cross_attention_mask, cross_attention_states, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, logits_to_keep, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["aspect_ratio_ids"] = None
            model_inputs["aspect_ratio_mask"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        return model_kwargs
