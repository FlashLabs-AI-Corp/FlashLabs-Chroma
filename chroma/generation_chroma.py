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
from typing import List, Tuple, Optional, Dict, Union, TYPE_CHECKING
from dataclasses import dataclass
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    GenerationMode,
)
from transformers.utils import logging

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


def multinomial_sample_one_no_sync(probs):
    """Does multinomial sampling without a cuda synchronization."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """Sample from logits using top-k sampling with temperature."""
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ChromaGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of ChromaForConditionalGeneration.generate.
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
        audio (`list(torch.FloatTensor)` of length `batch_size`):
            The generated audio.
    """
    audio: Optional[List[torch.Tensor]] = None


class ChromaGenerationMixin(GenerationMixin):
    def _get_stopping_criteria(self, *args, **kwargs) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(*args, **kwargs)

        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if not isinstance(criterion, MaxLengthCriteria):
                logger.warning(
                    f"Chroma does not support {criterion.__class__.__name__} stopping criteria, it will be ignored."
                )
            else:
                kept_criteria.append(criterion)
        return kept_criteria

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        """
        This method overrides [~generation.utils.GenerationMixin._prepare_generation_config].
        It ensures that the decoder generation config is initialized and that passed args as depth_decoder_* are properly handled.
        """
        # extract depth decoder kwargs and remove them from the main kwargs
        depth_decoder_kwargs = {
            k[len("decoder_"):]: v for k, v in kwargs.items() if k.startswith("decoder_")
        }

        # remove the decoder keys from the original kwargs
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("decoder_")}

        # initialize the generation config
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self.decoder.generation_config.update(**depth_decoder_kwargs)

        # ensure the depth decoder generation config is valid
        decoder_min_new_tokens = getattr(self.decoder.generation_config, "min_new_tokens") or (
            self.decoder.config.audio_num_codebooks - 1
        )
        decoder_max_new_tokens = getattr(self.decoder.generation_config, "max_new_tokens") or (
            self.decoder.config.audio_num_codebooks - 1
        )

        if {decoder_min_new_tokens, decoder_max_new_tokens} != {self.decoder.config.audio_num_codebooks - 1}:
            raise ValueError(
                f"decoder_generation_config's min_new_tokens ({decoder_min_new_tokens}) and max_new_tokens ({decoder_max_new_tokens}) must be equal to self.config.num_codebooks - 1 ({self.decoder.config.audio_num_codebooks - 1})"
            )
        elif self.decoder.generation_config.return_dict_in_generate:
            self.decoder.generation_config.return_dict_in_generate = False

        # Monkey patch the get_generation_mode method to support Chroma model
        original_get_generation_mode = generation_config.get_generation_mode

        def patched_get_generation_mode(assistant_model=None):
            generation_mode = original_get_generation_mode(assistant_model)
            if generation_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]:
                raise ValueError(
                    f"Generation mode {generation_mode} is not supported for Chroma model. Please set generation parameters to use greedy or sampling generation."
                )

            return generation_mode

        generation_config.get_generation_mode = patched_get_generation_mode

        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        只支持sample和greedy, 拿到backbone的hidden_states, logits
        Args:
            input_ids:
            generation_config:
            logits_processor:
            stopping_criteria:
            synced_gpus:
            streamer:
            **model_kwargs:
        Returns:
        """
        pad_token_id = self.config.codebook_pad_token_id
        has_eos_stopping_criteria = generation_config._eos_token_tensor is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        top_k = generation_config.top_k
        temperature = generation_config.temperature

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        if input_ids.ndim == 2 and model_kwargs.get("inputs_embeds") is None:
            for criterion in stopping_criteria:
                if isinstance(criterion, MaxLengthCriteria):
                    criterion.max_length -= cur_len

        # 累积每一帧，最终返回完整的 [B, T, num_codebooks]
        generated_frames = []

        # 获取ChromaForConditionalGeneration的forward方法
        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
        ):
            # 调用ChromaForConditionalGeneration的prepare_inputs_for_generation方法，组装输入用于forward
            # 处理thinker以及prompt_audio  prompt_text
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            # *************** Chroma specific ***************
            model_inputs.update({"output_hidden_states": True})
            # ============================================

            # 调用ChromaForConditionalGeneration的forward方法
            if is_prefill:
                backbone_outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                backbone_outputs = model_forward(**model_inputs, return_dict=True)

            # 获取backbone的hidden_states
            next_token_logits = backbone_outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            backbone_last_hidden_state = backbone_outputs.hidden_states[-1][:, -1, :]

            # 更新decoder部分进行generate的config
            model_kwargs = self._update_model_kwargs_for_generation(
                backbone_outputs,
                model_kwargs,
            )

            if synced_gpus and this_peer_finished:
                continue

            # Acquire backbone scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (backbone_outputs.attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (backbone_outputs.hidden_states,)

            # Do sample
            if do_sample:
                next_tokens = sample_topk(next_token_logits, top_k, temperature)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                next_tokens = next_tokens.unsqueeze(1)  # [B, 1]

            # decoder generate
            frame_codes = self.decoder.generate(
                input_ids=next_tokens,
                backbone_last_hidden_state=backbone_last_hidden_state.clone(),
                max_new_tokens=self.config.decoder_config.audio_num_codebooks - 1,
                min_new_tokens=self.config.decoder_config.audio_num_codebooks - 1,
                do_sample=do_sample,
                use_cache=True,
                temperature=temperature,
                top_k=top_k,
            )

            # 确保形状正确
            if frame_codes.shape[-1] != self.config.decoder_config.audio_num_codebooks:
                raise ValueError(
                    f"Generated codebooks shape {frame_codes.shape[-1]} does not match expected "
                    f"audio_num_codebooks {self.config.decoder_config.audio_num_codebooks}"
                )

            next_tokens = frame_codes

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + pad_token_id * (
                    1 - unfinished_sequences.unsqueeze(-1)
                )

            # 累积当前帧，形状为 [B, 1, num_codebooks]
            if next_tokens.sum() != 0:
                generated_frames.append(next_tokens.unsqueeze(1))

            # update generated ids, model inputs, and length for next step
            input_ids = next_tokens[:, None, :]

            # ============================================

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # *************** Chroma specific ***************
            # for the eos stopping criteria, is it expected that the eos token is the same for each codebook !!!!
            unfinished_sequences = unfinished_sequences & ~(
                input_ids[:, -1, :-1] == self.config.codebook_eos_token_id
            ).all(-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(torch.cat(generated_frames, dim=1), scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            del backbone_outputs

            del frame_codes

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            sequences = torch.cat(generated_frames, dim=1) if len(generated_frames) > 0 else input_ids
            return GenerateDecoderOnlyOutput(
                sequences=sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            sequences = torch.cat(generated_frames, dim=1) if len(generated_frames) > 0 else input_ids
            return sequences

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        output_audio: Optional[bool] = False,
        bos_token_id: Optional[int] = 0,
        **kwargs: dict
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        This method overrides [`~generation.utils.GenerationMixin.generate`] to match the specifics of the Chroma model.
        Indeed, Chroma model requires a custom generation sampling step:
        1. Infer the backbone model to sample the first codebook token
        2. Call generate on the depth decoder with the first codebook token as `input_ids` to sample the next codebook tokens
        3. Use these generated codebook tokens as `input_ids` to sample the next first codebook token using the backbone model
        4. Repeat until stopping criteria is met
        <Tip warning={true}>
        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, do_sample=True)`.
        </Tip>
        Parameters:
            inputs_ids (`torch.Tensor` of shape (batch_size, seq_length), *optional*):
                The sequence used as a prompt for the backbone model.
            input_values (`torch.Tensor` of shape (batch_size, channels, max_concatenated_audio_length), *optional*):
                The batched audio input values, where each batch entry contains the concatenation of all audio segments for that entry.
                These values will be encoded into codebook tokens using the codec model and merged with the text input ids provided in `input_ids`.
            input_values_cutoffs (`torch.Tensor` of shape (batch_size, max_num_audio), *optional*):
                Specify the end positions of audio segments within each batch entry, relative to the concatenated audio input.
                If a batch entry has fewer segments than the maximum, it is padded with -1. For example, in a batch of 2 sequences
                where the first contains 2 audio segments of length l1, and the second contains 1 audio segment of length l2,
                the input_values_cutoffs would be: [[l1, 2 * l1], [l2, -1]].
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            output_audio (`bool`, *optional*):
                Whether to return the generated audio.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. Depth decoder specific kwargs should be prefixed with *depth_decoder_*.
        Return:
            [`ChromaGenerateOutput`] or `torch.LongTensor` or `list[torch.FloatTensor]`: A [`ChromaGenerateOutput`]
            (if `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a `torch.LongTensor` when `output_audio=False`
            or a `list[torch.FloatTensor]` otherwise.
        """
        generate_output = super().generate(
            input_ids=input_ids,
            input_values=input_values,
            input_values_cutoffs=input_values_cutoffs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            synced_gpus=synced_gpus,
            streamer=streamer,
            bos_token_id=bos_token_id,
            **kwargs,
        )
        generate_returned_dict = not isinstance(generate_output, torch.Tensor)
        audio = None
        if output_audio:
            generated_audio_codes = generate_output.sequences if generate_returned_dict else generate_output

            # infer the codec model
            audio = []
            with torch.no_grad():
                for audio_codes_batch in generated_audio_codes:
                    eos_idxs = (audio_codes_batch == self.config.codebook_eos_token_id).all(dim=-1).nonzero()
                    if eos_idxs.numel() != 0:
                        cutoff_idx = eos_idxs.min()
                    else:
                        cutoff_idx = audio_codes_batch.shape[0]

                    audio_codes_batch = audio_codes_batch[:cutoff_idx]
                    codec_decode_output = self.codec_model.decode(audio_codes_batch.transpose(0, 1).unsqueeze(0))
                    audio.append(codec_decode_output.audio_values)

        if generate_returned_dict:
            return ChromaGenerateOutput(audio=audio, **generate_output)
        elif output_audio:
            return audio
        else:
            return generate_output
