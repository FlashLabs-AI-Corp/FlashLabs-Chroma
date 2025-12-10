import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor
from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.processing_utils import AudioKwargs, ProcessingKwargs
from qwen_omni_utils import process_audio_info
from typing import Union, Optional, Tuple
from transformers.feature_extraction_utils import BatchFeature

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class ChromaAudioKwargs(AudioKwargs, total=False):
    target_sample_rate: Optional[int]  # 目标采样率


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs: Unpack[ChromaProcessorKwargs]) -> BatchFeature:
        # thinker processor
        text, audios = self.apply_chat_template(*args, **kwargs)
        thinker_inputs = super().__call__(text=text, audio=audios, return_tensors="pt", padding=True,
                                          use_audio_in_video=False)

        inputs = {f"thinker_{k}": v for k, v in thinker_inputs.items()}

        prompt_audio = kwargs.get("prompt_audio")
        prompt_text = kwargs.get("prompt_text")
        assert prompt_audio is not None, "prompt_audio can not be empty"
        assert prompt_text is not None, "prompt_text can not be empty"

        prompt_ids = super().__call__(text=prompt_text, return_tensors="pt")

        if isinstance(prompt_audio, str):
            prompt_audio_tensor = self.load_audio(prompt_audio, kwargs.get("target_sample_rate", 24000))
        elif isinstance(prompt_audio, torch.Tensor):
            prompt_audio_tensor = prompt_audio
        elif isinstance(prompt_audio, np.ndarray):
            prompt_audio_tensor = torch.from_numpy(prompt_audio)
            if prompt_audio_tensor.dim() > 1:
                prompt_audio_tensor = prompt_audio_tensor.squeeze()
        else:
            raise ValueError(f"prompt audio must be str, tensor or numpy, but got  {type(prompt_audio)}")

        return BatchFeature(
            data={
                **inputs,
                **prompt_ids,
                'input_values': prompt_audio_tensor
            },
            tensor_type=kwargs.get("return_tensors"),
        )

    def load_audio(self, audio_path: str, target_sample_rate: int = 24000) -> torch.Tensor:
        """加载音频文件并重采样，支持本地路径和 HTTP/HTTPS URL"""
        try:
            # 本地文件路径
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
            print(f"加载音频文件失败: {audio_path}, 错误: {e}")
            raise

    def apply_chat_template(
        self,
        conversations,
        chat_template=None,
        **kwargs
    ) -> Tuple[str, list]:
        """应用聊天模板"""
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        audios = process_audio_info(conversations, use_audio_in_video=False)
        return super().apply_chat_template(conversations, chat_template, **kwargs), audios


AutoProcessor.register("ChromaProcessor", ChromaProcessor)
