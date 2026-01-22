# coding=utf-8
# Copyright 2025 The FlashLabs team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, fields
from typing import Optional, Tuple, Dict, Union, Any
from .configuration_chroma import ChromaConfig, ChromaDecoderConfig, ChromaBackboneConfig
from .generation_chroma import ChromaGenerationMixin
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.mimi.modeling_mimi import MimiModel

logger = logging.get_logger(__name__)

PASSTHROUGH_KEYS = [
    "thinker_input_ids",
    "thinker_attention_mask",
    "thinker_cache_position",
    "thinker_past_key_values",
    "thinker_input_features",
    "thinker_feature_attention_mask",
    "thinker_eos",
    "thinker_hidden_states",
    "thinker_logits",
    "thinker_flag",
    "prefilled",
    "attention_mask",  # we need to control the attention_mask manually, not just increment by 1
]

ONE_TIME_KEYS = [
    "input_values",  # ref audio waveform
    "thinker_input_features",
    "thinker_feature_attention_mask",
]


@dataclass
class ChromaOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits: torch.FloatTensor | None = None
    past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    cache_position: Optional[int] = None
    attention_mask: Optional[torch.LongTensor] = None

    # all thinker inputs should be carried through the forward function to the next step
    thinker_loss: Optional[torch.FloatTensor] = None
    thinker_logits: torch.FloatTensor = None
    thinker_past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    thinker_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    thinker_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    thinker_input_ids: Optional[torch.FloatTensor] = None
    thinker_attention_mask: Optional[torch.FloatTensor] = None
    thinker_input_features: Optional[torch.FloatTensor] = None
    thinker_feature_attention_mask: Optional[torch.FloatTensor] = None
    thinker_cache_position: Optional[torch.FloatTensor] = None
    thinker_flag: Optional[bool] = None
    thinker_eos: Optional[torch.BoolTensor] = None
    # text_streamer: Optional[BaseStreamer] = None # for text output

    backbone_loss: Optional[torch.FloatTensor] = None
    backbone_logits: torch.FloatTensor = None
    backbone_past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    backbone_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    decoder_loss: Optional[torch.FloatTensor] = None
    decoder_logits: torch.FloatTensor = None
    decoder_past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ChromaLlamaModel(LlamaModel):
    """
    Base model for chroma
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Identity()


class ChromaPreTrainedModel(PreTrainedModel):
    config_class = ChromaConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None and not getattr(module.bias, "_is_hf_initialized", False):
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ChromaCodebookHead):
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data.normal_(mean=0.0, std=std)


class ChromaAudioEmbedding(nn.Module):
    def __init__(self, audio_num_codebooks, audio_vocab_size, hidden_size):
        super().__init__()
        self.embed_audio_tokens = nn.Embedding(
            num_embeddings=audio_num_codebooks * audio_vocab_size,
            embedding_dim=hidden_size
        )
        self.audio_vocab_size = audio_vocab_size

    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: [B, num_codebooks]
        Returns: [B, num_codebooks, hidden_size]
        """
        num_codebooks = input_ids.shape[-1]
        audio_frames = input_ids + (
            self.audio_vocab_size * torch.arange(num_codebooks, device=input_ids.device)
        )
        embeddings = self.embed_audio_tokens(audio_frames.view(-1)).reshape(audio_frames.shape + (2048,))
        return embeddings


class ChromaBackboneForCausalLM(ChromaPreTrainedModel):
    config_class = ChromaBackboneConfig
    _supports_flash_attn_2 = True

    def __init__(self, config: ChromaBackboneConfig):
        super().__init__(config)
        self.model = ChromaLlamaModel(LlamaConfig(**config.to_dict()))
        self.codebook0_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # During inference, pass parameters of this embedding to decoder
        self.audio_embedding = ChromaAudioEmbedding(
            config.audio_num_codebooks,
            config.vocab_size,
            config.hidden_size,
        )

        self.post_init()

    def emb_audio_frames(self, audio_frames: torch.Tensor, add_frame: bool = True) -> torch.Tensor:
        assert audio_frames.dim() > 1, "audio_frames must be a tensor with shape [..., codebook_num]"
        audio_frames = audio_frames.contiguous()
        codebook_num = audio_frames.shape[-1]
        audio_frames = audio_frames.masked_fill(audio_frames == -100, 0)
        audio_embeddings = self.audio_embedding(audio_frames)

        if add_frame:
            audio_embeddings = audio_embeddings.sum(dim=-2)
        return audio_embeddings

    def loss_fn(self, logits, labels, ignore_index=-100):
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()

        # Shift so that tokens < n predict n
        labels = F.pad(labels, (0, 1), value=ignore_index)

        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)

        logits = logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.to(logits.device)

        loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index)

        return loss

    def forward(
        self,
        input_embeddings: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        args:
            input_embeddings: [B, seq_len, 2048]  every [2048] is a hidden from qwen
            labels: [B, seq_len]  every element is codebook0 id
        return:
            output: BaseModelOutputWithPast
                loss: [B, seq_len]
                logits: [B, seq_len, 2051]
                hidden_states: [B, seq_len, 2048]
        """
        if input_embeddings is None:
            raise ValueError("input_embeddings is required")

        assert input_embeddings.shape[-1] == self.config.hidden_size, \
            f"input_embeddings must have {self.config.hidden_size} dimensions"

        # Forward
        output: BaseModelOutputWithPast = self.model(
            inputs_embeds=input_embeddings,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs
        )
        logits = self.codebook0_head(output.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.clone().detach())

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class ChromaCodebookHead(nn.Module):

    def __init__(
        self,
        audio_num_codebooks,
        audio_vocab_size,
        hidden_size,
    ):
        super().__init__()
        self.num_codebooks = audio_num_codebooks
        self.vocab_size = audio_vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_codebooks, self.hidden_size, self.vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: [B, num_codebook, input_dim]
        return:
            output: [B, num_codebook, output_dim]
        """
        codebook_num = x.shape[1]
        output = torch.bmm(
            x.transpose(0, 1),  # [num_codebook, B, input_dim]
            self.weight[:codebook_num, :, :]  # [num_codebook, input_dim, output_dim]
        )
        return output.transpose(0, 1)  # [B, num_codebook, output_dim]

    def get_logits(self, x: torch.Tensor, codebook_id: int):
        """
        args:
            x: [B, input_dim]
            codebook_id: int
        return:
            logits: [B, ]
        """
        # codebook 0 is in backbone, so the weight is from 1 to num_codebooks
        if codebook_id == 0 or codebook_id > self.num_codebooks:
            raise ValueError(f"codebook_id must be between 1 and {self.num_codebooks}, but got {codebook_id}")
        return torch.mm(x, self.weight[codebook_id - 1, :, :])


class ChromaDecoderForCausalLM(ChromaPreTrainedModel, GenerationMixin):
    config_class = ChromaDecoderConfig
    _supports_flash_attn_2 = True

    def __init__(self, config: ChromaDecoderConfig):
        super().__init__(config)

        self.projection = nn.Linear(self.config.audio_embedding_dim, self.config.hidden_size, bias=False)

        self.model = ChromaLlamaModel(LlamaConfig(**config.to_dict()))

        self.codebook_head = ChromaCodebookHead(
            audio_num_codebooks=self.config.audio_num_codebooks - 1,
            audio_vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
        )

        self.audio_embedding = ChromaAudioEmbedding(
            config.audio_num_codebooks,
            config.vocab_size,
            config.audio_embedding_dim,
        )

        self.post_init()

    def loss_fn(self, logits, labels, ignore_index=-100):
        """
        logits: [B, num_codebooks-1, 2051]
        labels: [B, num_codebooks-1]
        """

        # flatten logits and labels
        vocab_size = logits.size(-1)
        logits_flat = logits.contiguous().view(-1, vocab_size)
        labels_flat = labels.contiguous().view(-1)

        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)

        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
            # During training, just pass inputs_embeds which are embedded by backbone.embedding, so the internal embedding is not used.
            # During inference, pass input_ids and backbone_last_hidden_state, which is aligned with the standard HuggingFace model.
        args:
            input_ids: [B, codebook_num]  every sequence is an audio frame
            backbone_last_hidden_state: [B, 2048]  every [2048] is a hidden from qwen
            inputs_embeds: [B, seq_len, codebook_num, 2048]  every [2048] is a hidden from qwen
            labels: [B, seq_len, num_codebook]  every [n] element is [-100, codebook0-num_codebook-1 id]
        return:
            output: Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]
                loss: [B, seq_len]
                logits: [B, seq_len, codebook_num, 2051]
        """

        if inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds or input_ids is required")

        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("inputs_embeds and input_ids cannot be used at the same time")

        loss = None

        if inputs_embeds is None:

            past_codebook_num = past_key_values.get_seq_length() - 1 if past_key_values is not None else 0

            if past_codebook_num > self.config.audio_num_codebooks - 1:
                raise ValueError(
                    f"past_codebook_num is greater than audio_num_codebooks - 1, {past_codebook_num} > {self.config.audio_num_codebooks - 1}")
            offset = (torch.arange(input_ids.shape[-1],
                                   device=input_ids.device) + past_codebook_num) * self.config.vocab_size
            audio_ids_embed = self.audio_embedding.embed_audio_tokens(input_ids + offset)
            inputs_embeds = torch.cat([backbone_last_hidden_state.unsqueeze(1), audio_ids_embed],
                                      dim=1) if backbone_last_hidden_state is not None else audio_ids_embed

        orig_shape = inputs_embeds.shape

        # if input_embeddings is 4D, it means that the input_embeddings is a batch of sequences
        if inputs_embeds.dim() == 4:
            # [B, seq_len, codebook_num, 2048] -> [B*seq_len, codebook_num, 2048]
            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            # [B, seq_len, codebook_num] -> [B*seq_len, codebook_num]
            labels = labels.reshape(-1, labels.shape[-1])

        # cut off the eos any way (delete -1)
        has_eos = inputs_embeds.shape[1] == self.config.audio_num_codebooks + 1
        inputs_embeds = inputs_embeds[:, :self.config.audio_num_codebooks, :]

        # Forward
        inputs_embeds = self.projection(inputs_embeds)
        output: BaseModelOutputWithPast = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs
        )

        if past_key_values is not None:
            logits = self.codebook_head.get_logits(
                output.last_hidden_state.squeeze(1),
                past_codebook_num + 1
            ).unsqueeze(1)
        else:
            logits = self.codebook_head(output.last_hidden_state[:, 1:, :])  # (delet 0)

        if labels is not None:
            # the sequence must be full, calculate loss at codebook 1-31
            assert labels.shape[
                       1] == self.config.audio_num_codebooks - 1, f"labels must have {self.config.audio_num_codebooks - 1} tokens, but got {labels.shape[1]}"
            assert logits.shape[
                       1] == self.config.audio_num_codebooks - 1, f"logits must have {self.config.audio_num_codebooks - 1} tokens, but got {logits.shape[1]}"
            loss = self.loss_fn(logits, labels.clone().detach())

        # Ensure that the output logits sequence length matches the input sequence length
        pad_left = 1 if backbone_last_hidden_state is not None or has_eos or input_ids is None else 0
        pad_right = 1 if has_eos else 0
        # if see 0 in the first position, it's just a padding
        logits = F.pad(logits, (0, 0, pad_left, pad_right), value=0)
        logits = logits.reshape(*orig_shape[:-1], logits.shape[-1])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        for decoder generate
        Args:
            input_ids:
            past_key_values:
            attention_mask:
            inputs_embeds:
            cache_position:
            **kwargs:
        Returns:
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **kwargs
        )

        is_first_generation_step = past_key_values is None
        if not is_first_generation_step:
            model_inputs.pop("backbone_last_hidden_state")
        return model_inputs


class ChromaForConditionalGeneration(ChromaPreTrainedModel, ChromaGenerationMixin):
    base_model_prefix = "chroma"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    _tied_weights_keys = {
        "backbone.audio_embedding.embed_audio_tokens.weight": "decoder.audio_embedding.embed_audio_tokens.weight",
    }

    def __init__(self, config: ChromaConfig):
        super().__init__(config)
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.backbone = ChromaBackboneForCausalLM._from_config(config.backbone_config)
        self.decoder = ChromaDecoderForCausalLM._from_config(config.decoder_config)
        self.codec_model = MimiModel._from_config(config.codec_config)

        assert self.backbone.config.audio_num_codebooks == config.audio_num_codebooks, f"backbone.config.audio_num_codebooks {self.backbone.config.audio_num_codebooks} != config.audio_num_codebooks {config.audio_num_codebooks}"
        assert self.decoder.config.audio_num_codebooks == config.audio_num_codebooks, f"decoder.config.audio_num_codebooks {self.decoder.config.audio_num_codebooks} != config.audio_num_codebooks {config.audio_num_codebooks}"

        self.post_init()

        # initialize prompt embedding
        self._prompt_embeddings_initialized = False

    def _tie_weights(self):
        self._tie_or_clone_weights(
            self.backbone.audio_embedding.embed_audio_tokens,
            self.decoder.audio_embedding.embed_audio_tokens,
        )

    def _embed_text_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "thinker"):
            return self.thinker.model.embed_tokens(ids.to(self.device))
        else:
            return self.embed_tokens(ids.to(self.device))

    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_values_cutoffs: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        thinker_input_ids: Optional[torch.LongTensor] = None,
        thinker_attention_mask: Optional[torch.LongTensor] = None,
        thinker_cache_position: Optional[torch.LongTensor] = None,
        thinker_past_key_values: Optional[Cache] = None,
        thinker_hidden_states: Optional[torch.FloatTensor] = None,
        thinker_input_features: Optional[torch.FloatTensor] = None,
        thinker_feature_attention_mask: Optional[torch.LongTensor] = None,
        thinker_logits: Optional[torch.FloatTensor] = None,
        prompt_audio: Optional[torch.FloatTensor] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        thinker_flag: bool = True,
        thinker_eos: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """
        args:
            input_ids: [B, seq_len]
            input_values: [B, channels, audio_seq_len]
            input_values_cutoffs: [B, max_num_audio]
            past_key_values: [B, num_layers, num_heads, seq_len, hidden_size]
            attention_mask: [B, seq_len]
            inputs_embeds: [B, seq_len, hidden_size]
            cache_position: [B, seq_len]
            thinker_input_ids: [B, seq_len]  None means thinker had generate eos
            thinker_attention_mask: [B, seq_len]
            thinker_cache_position: [B, seq_len]
            thinker_past_key_values:
            thinker_hidden_states: [B, seq_len, hidden_size]
            thinker_input_features: [B, seq_len, hidden_size]
            thinker_feature_attention_mask: [B, seq_len]
            thinker_logits: [B, seq_len, hidden_size]
            prompt_audio: [B, channels, audio_seq_len]
            prompt_ids: [B, seq_len]
            thinker_flag: bool whether thinker need to generate next token and inject into inputs_embeds
        Returns:
            inputs_embeds: [B, seq_len, hidden_size]
        use input_ids to build prompt_embeds, if it is the step that need thinker to generate next token, then inject its hidden states and next token embedding into inputs_embeds
        Generate input concatenation (1:2 ratio):
            First step (with input_values): build the prompt, then forcefully inject one pair of thinker tokens;
            Subsequent steps: first concatenate the previous frame's audio, then inject thinker tokens only if thinker_flag is True;
            After injection, set thinker_flag = False;
            If no injection occurs, set thinker_flag = True;
        """

        if input_values is not None:
            # first step: build inputs_embeds from input_values
            inputs_embeds, attention_mask = self._build_prompt_embeds(input_ids, attention_mask, input_values,
                                                                      input_values_cutoffs)
        else:
            # subsequent steps: build inputs_embeds from input_ids
            inputs_embeds = self.backbone.emb_audio_frames(
                input_ids.to(self.device)
            )
            # attention_mask is already updated by parent class _update_model_kwargs_for_generation
            # It should already have the correct length for past_key_values + new tokens

        # Initialize thinker_eos if it's None
        if thinker_eos is None:
            if thinker_input_ids is not None:
                thinker_eos = torch.zeros(thinker_input_ids.shape[0], dtype=torch.bool, device=thinker_input_ids.device)
            else:
                thinker_eos = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)

        if thinker_input_ids is not None and thinker_flag:
            # Incrementally update the new token(s) for thinker
            thinker_input_ids, thinker_attention_mask, thinker_cache_position, thinker_past_key_values = self._update_thinker_model_kwargs(
                thinker_input_ids, thinker_attention_mask, thinker_cache_position, thinker_past_key_values
            )

            # thinker forward
            thinker_outputs = self.thinker(
                input_ids=thinker_input_ids,
                input_features=thinker_input_features,
                attention_mask=thinker_attention_mask,
                feature_attention_mask=thinker_feature_attention_mask,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                past_key_values=thinker_past_key_values,
                cache_position=thinker_cache_position,
                use_audio_in_video=False,
            )

            thinker_hidden_states = thinker_outputs.hidden_states[-1]
            thinker_past_key_values = thinker_outputs.past_key_values
            thinker_logits = thinker_outputs.logits

            # get next token
            thinker_next_ids = thinker_logits[:, -1:, :].argmax(dim=-1)
            next_token_emb = self._embed_text_tokens(thinker_next_ids)

            # Update thinker_eos: once True, always True
            next_token_eos = thinker_next_ids.squeeze(-1) == self.config.im_end_token_id
            new_thinker_eos = thinker_eos | next_token_eos

            # Incrementally extend inputs_embeds for thinker generation
            thinker_input_embeddings = torch.cat([thinker_hidden_states[:, -1:, :], next_token_emb], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, thinker_input_embeddings], dim=1)

            # Incrementally extend attention_mask for thinker generation
            # The two thinker tokens (hidden state + next token) should have attention_mask = 1
            # even if the next token is EOS, because they need to be processed in this step
            # Only tokens added AFTER thinker reached EOS should have attention_mask = 0
            thinker_attention_values = (~thinker_eos).long().unsqueeze(1)
            attention_mask = torch.cat([attention_mask, thinker_attention_values, thinker_attention_values], dim=1)

            # Update thinker_eos for next iteration
            thinker_eos = new_thinker_eos
            thinker_input_ids = thinker_next_ids if not thinker_eos.all() else None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device
        )

        # Ensure attention_mask has the correct length: past_seen_tokens + current_tokens
        expected_attention_mask_length = past_seen_tokens + inputs_embeds.shape[1]
        assert attention_mask.shape[
                   1] == expected_attention_mask_length, f"attention_mask.shape[1] {attention_mask.shape[1]} != expected_attention_mask_length {expected_attention_mask_length}"

        model_inputs = {
            "input_ids": None,
            "input_embeddings": inputs_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": True,
            "output_hidden_states": True,
            # thinker related states
            "thinker_past_key_values": thinker_past_key_values,
            "thinker_hidden_states": thinker_hidden_states,
            "thinker_logits": thinker_logits,
            "thinker_input_ids": thinker_input_ids,
            "thinker_attention_mask": thinker_attention_mask,
            "thinker_input_features": thinker_input_features,
            "thinker_feature_attention_mask": thinker_feature_attention_mask,
            "thinker_cache_position": thinker_cache_position,
            "thinker_flag": not thinker_flag if thinker_input_ids is not None else False,
            "thinker_eos": thinker_eos,
        }
        return model_inputs

    @torch.no_grad()
    def _register_prompt_embeddings(self):

        # text_start_emb
        text_start_ids = torch.tensor([self.config.text_start_token_id], dtype=torch.long, device=self.device)
        text_start_emb = self.thinker.model.embed_tokens(text_start_ids).unsqueeze(0)
        self.register_buffer("text_start_emb", text_start_emb, persistent=False)

        # text_end_emb
        text_end_ids = torch.tensor([self.config.text_end_token_id], dtype=torch.long, device=self.device)
        text_end_emb = self.thinker.model.embed_tokens(text_end_ids).unsqueeze(0)
        self.register_buffer("text_end_emb", text_end_emb, persistent=False)

        # eos_token_audio
        eos_token_audio = torch.zeros((1, 1, self.config.backbone_config.hidden_size), dtype=text_start_emb.dtype,
                                      device=self.device)
        self.register_buffer("eos_token_audio", eos_token_audio, persistent=False)

        # attention_mask
        attention_mask = torch.ones(1, 1, dtype=torch.long, device=self.device)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

        # arrange for audio frame cutoff
        arr = torch.arange(self.config.backbone_config.max_position_embeddings, device=self.device)
        self.register_buffer("arr", arr, persistent=False)

        self._prompt_embeddings_initialized = True

    def _build_prompt_embeds(
        self,
        input_ids,
        attention_mask=None,
        input_values=None,
        input_values_cutoffs=None,
    ):
        """
        Build QSM input embeddings according to the specified layout for generation
        Args:
            input_ids [B, seq_len]: prompt text ids
            attention_mask [B, seq_len]: attention mask
            input_value Tensor[B, channels(1), audio_seq_len]: prompt audio waveform
            input_values_cutoffs Tensor[B, max_num_audio]: prompt audio waveform cutoffs
        Returns:
            input_embeddings [B, seq_len, hidden_size]: input embeddings
            attention_mask [B, seq_len]: attention mask
        """

        if not self._prompt_embeddings_initialized:
            self._register_prompt_embeddings()

        N = input_ids.shape[0]
        assert N == input_values.shape[0], f"input_values.shape[0] {input_values.shape[0]} != input_ids.shape[0] {N}"
        assert N == input_values_cutoffs.shape[
            0], f"input_values_cutoffs.shape[0] {input_values_cutoffs.shape[0]} != input_ids.shape[0] {input_ids.shape[0]}"
        assert N == attention_mask.shape[
            0], f"attention_mask.shape[0] {attention_mask.shape[0]} != input_ids.shape[0] {N}"

        audio_codes = self.codec_model.encode(input_values).audio_codes  # add channel dimension
        audio_codes = audio_codes[:, :self.config.audio_num_codebooks, :]  # HACK: may not necessary

        prompt_audio_emb = self.backbone.emb_audio_frames(
            audio_codes.permute(0, 2, 1).to(self.device)
        )
        prompt_audio_attention_mask = torch.ones((N, prompt_audio_emb.shape[1]), device=self.device)

        audio_codes_cutoffs = torch.ceil(input_values_cutoffs / self.config.audio_frame_freq).long().unsqueeze(1)
        arr = self.arr[:prompt_audio_emb.shape[1]].unsqueeze(0).expand(N, -1)
        prompt_audio_attention_mask[arr >= audio_codes_cutoffs] = 0

        prompt_text_emb = self._embed_text_tokens(input_ids.to(self.device))
        prompt_text_attention_mask = attention_mask.clone()

        # Concatenate all embeddings using torch.cat for better performance
        input_embeddings = torch.cat([
            self.text_start_emb.expand(N, 1, -1),
            prompt_text_emb,
            self.text_end_emb.expand(N, 1, -1),
            prompt_audio_emb,
            self.eos_token_audio.expand(N, 1, -1),
        ], dim=1)

        attention_mask = torch.cat([
            self.attention_mask.expand(N, 1),
            prompt_text_attention_mask,
            self.attention_mask.expand(N, 1),
            prompt_audio_attention_mask,
            self.attention_mask.expand(N, 1),
        ], dim=1)

        return input_embeddings, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # [seq_len, num_codebooks]
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        input_embeddings: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ChromaOutputWithPast:
        """
        main model forward
        Args:
            input_ids: never used
            position_ids:
            attention_mask:
            feature_attention_mask:
            past_key_values:
            inputs_embeds:
            labels:
            use_cache:
            output_attentions:
            output_hidden_states:
            input_embeddings: backbone input embeddings
            cache_position: backbone position
            **kwargs:
        Returns:
        """
        backbone_outputs: CausalLMOutputWithPast = self.backbone(
            input_embeddings=input_embeddings,
            labels=labels,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return self._build_outputs(
            loss=backbone_outputs.loss,
            logits=backbone_outputs.logits,
            hidden_states=backbone_outputs.hidden_states,
            past_key_values=backbone_outputs.past_key_values,
            attention_mask=attention_mask,
            **kwargs
        )

    def _build_outputs(self, **kwargs) -> ChromaOutputWithPast:
        fields_names = [f.name for f in fields(ChromaOutputWithPast)]
        outputs = ChromaOutputWithPast(**{k: v for k, v in kwargs.items() if k in fields_names})
        return outputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ChromaOutputWithPast,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1
    ) -> Dict[str, Any]:
        """
        Update model_kwargs during the generation process
        """

        # Update thinker-related keys
        for key in PASSTHROUGH_KEYS:
            model_kwargs[key] = getattr(outputs, key, None)

        # Clear one-time keys
        for key in ONE_TIME_KEYS:
            model_kwargs[key] = None

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder,
            1  # Always use 1 to let parent add 1 token for next step
        )

        return model_kwargs

    def _update_thinker_model_kwargs(
        self,
        thinker_input_ids: torch.Tensor,
        thinker_attention_mask: Optional[torch.Tensor] = None,
        thinker_cache_position: Optional[torch.Tensor] = None,
        thinker_past_key_values: Optional[Cache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Cache]]:

        """
        update thinker model kwargs during the generation process
        Args:
            thinker_input_ids:
            thinker_attention_mask:
            thinker_cache_position:
            thinker_past_key_values:
        Returns:
        """

        past_seen_tokens = thinker_past_key_values.get_seq_length() if thinker_past_key_values is not None else 0
        num_new_tokens = thinker_input_ids.shape[1]

        if thinker_cache_position is None:
            thinker_cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + thinker_input_ids.shape[1], device=thinker_input_ids.device
            )
        else:
            thinker_cache_position = thinker_cache_position[-num_new_tokens:] + num_new_tokens

        if thinker_attention_mask is None:
            thinker_attention_mask = torch.ones((thinker_input_ids.shape[0], num_new_tokens),
                                                device=thinker_input_ids.device)
        else:
            if thinker_past_key_values is not None:
                thinker_attention_mask = torch.cat(
                    [thinker_attention_mask,
                     thinker_attention_mask.new_ones((thinker_attention_mask.shape[0], num_new_tokens))],
                    dim=-1,
                )

        return thinker_input_ids, thinker_attention_mask, thinker_cache_position, thinker_past_key_values


__all__ = [
    "ChromaPreTrainedModel",
    "ChromaLlamaModel",
    "ChromaBackboneForCausalLM",
    "ChromaDecoderForCausalLM",
    "ChromaForConditionalGeneration",
]
