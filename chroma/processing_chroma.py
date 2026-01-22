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


import base64
import logging
from io import BytesIO

import av
import torch
import librosa
import audioread
import torchaudio
import numpy as np
from typing import Union, Optional, Tuple, Unpack, List
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.processing_utils import AudioKwargs, ProcessingKwargs
from transformers.feature_extraction_utils import BatchFeature


def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele:
                        path = ele["audio"]
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(path)
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = base64.b64decode(base64_data)
                            audios.append(librosa.load(BytesIO(data), sr=16000)[0])
                        elif path.startswith("http://") or path.startswith("https://"):
                            audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                        elif path.startswith("file://"):
                            audios.append(librosa.load(path[len("file://"):], sr=16000)[0])
                        else:
                            audios.append(librosa.load(path, sr=16000)[0])
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                if use_audio_in_video and ele["type"] == "video":
                    if "video" in ele:
                        path = ele["video"]
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                        elif path.startswith("file://"):
                            audios.append(librosa.load(path[len("file://"):], sr=16000)[0])
                        else:
                            audios.append(librosa.load(path, sr=16000)[0])
                    else:
                        raise ValueError("Unknown video {}".format(ele))
    if len(audios) == 0:
        audios = None
    return audios


class ChromaAudioKwargs(AudioKwargs, total=False):
    target_sample_rate: Optional[int]


class ChromaProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: ChromaAudioKwargs
    prompt_text: Optional[str]
    prompt_audio: Optional[Union[str, torch.Tensor]]
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "target_sample_rate": 24000
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ChromaProcessor(Qwen2_5OmniProcessor):
    r"""
        Constructs a Chroma processor inherit from Qwen2.5Omni processor.
        [`ChromaProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`], [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
        [`~ChromaProcessor.__call__`] and [`~ChromaProcessor.decode`] for more information.
        Args:
            image_processor ([`Qwen2VLImageProcessor`], *optional*):
                The image processor.
            video_processor ([`Qwen2VLVideoProcessor`], *optional*):
                The video processor.
            feature_extractor ([`WhisperFeatureExtractor`], *optional*):
                The audio feature extractor.
            tokenizer ([`Qwen2TokenizerFast`], *optional*):
                The text tokenizer.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
        """

    def __init__(self, image_processor=None, video_processor=None, feature_extractor=None, tokenizer=None,
                 chat_template=None):
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template)

    def __call__(
        self,
        conversations: List[List[dict]],
        prompt_audio: List[str],
        prompt_text: List[str],
        **kwargs: Unpack[ChromaProcessorKwargs]
    ) -> BatchFeature:

        assert prompt_audio is not None, "prompt_audio can not be empty"
        assert prompt_text is not None, "prompt_text can not be empty"

        N = len(conversations)
        assert len(prompt_audio) == N, f"prompt_audio length {len(prompt_audio)} != conversations length {N}"
        assert len(prompt_text) == N, f"prompt_text length {len(prompt_text)} != conversations length {N}"

        # thinker processor
        text, audios = self.apply_chat_template(conversations, **kwargs)
        thinker_inputs = super().__call__(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        thinker_inputs = {f"thinker_{k}": v for k, v in thinker_inputs.items()}

        inputs = super().__call__(text=prompt_text, return_tensors="pt", padding=True)
        prompt_audio_wavs = [self.load_audio(audio, kwargs.get("target_sample_rate", 24000)) for audio in prompt_audio]
        prompt_audio_cutoffs = torch.tensor([len(audio) for audio in prompt_audio_wavs], dtype=torch.long)
        prompt_audio_tensor = torch.nn.utils.rnn.pad_sequence(
            prompt_audio_wavs, batch_first=True
        ).unsqueeze(1)  # add channel dimension

        return BatchFeature(
            data={
                **thinker_inputs,
                **inputs,
                'input_values': prompt_audio_tensor,
                'input_values_cutoffs': prompt_audio_cutoffs
            },
            tensor_type=kwargs.get("return_tensors"),
        )

    def load_audio(self, audio_path: str | None, target_sample_rate: int = 24000) -> torch.Tensor:
        """
        load audio wav and resample it to target sample rate
        Args:
            audio_path:
            target_sample_rate:
        Returns:
        """
        try:
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0),
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )

            return audio_tensor
        except Exception as e:
            logging.error(f"load audio file erro: {e}")
            raise

    def apply_chat_template(
        self,
        conversations,
        chat_template=None,
        **kwargs
    ) -> Tuple[str, list]:
        """
        apply chat_template.jinja template to format conversations
        Args:
            conversations:
            chat_template:
            **kwargs:
        Returns:
        """
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        audios = process_audio_info(conversations, use_audio_in_video=False)
        return self.tokenizer.apply_chat_template(conversations, chat_template, **kwargs), audios


__all__ = ["ChromaProcessor"]
