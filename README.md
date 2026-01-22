<div align="center">

# FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning

</div>

<br>
<div align="center">
<img src="figures/logo.svg" alt="FlashLabs Chroma Logo" width="400px"/>

<h3>🚀 Get Started with <a href="https://www.flashlabs.ai/flashai-voice-agents">FlashAI Voice Agents</a>!</h3>

<p><strong>Production-ready voice AI solutions</strong> powered by Chroma | <strong>Open-source model</strong> for developers & researchers</p>

[![Deploy AI Voice Agents](https://img.shields.io/badge/🎯%20Voice%20Agents-blue?style=for-the-badge)](https://www.flashlabs.ai/flashai-voice-agents)
[![Download Model](https://img.shields.io/badge/🤗%20Download%20Model-orange?style=for-the-badge)](https://huggingface.co/FlashLabs/Chroma-4B)
[![Technical Report](https://img.shields.io/badge/📄%20Technical%20Report-red?style=for-the-badge)](https://arxiv.org/abs/2601.11141)

</div>

## Model Description

**Chroma 1.0** is an advanced multimodal model developed by **[FlashLabs](https://flashlabs.ai)**. It is designed to understand and generate content across multiple modalities, including text and audio. As a virtual human model, Chroma possesses the ability to process auditory inputs and respond with both text and synthesized speech, enabling natural voice interactions.

- **Model Type:** Multimodal Causal Language Model
- **Developed by:** FlashLabs
- **Language(s):** English
- **License:** Apache-2.0
- **Model Architecture:** 
  - **Reasoner:** Based on Qwen2.5-Omni-3B
  - **Backbone:** Based on Llama3 (16 layers, 2048 hidden size)
  - **Decoder:** Based on Llama3 (4 layers, 1024 hidden size)
  - **Codec:** Mimi (24kHz sampling rate)

## Model Architecture

<img src="figures/architecture.png" alt="Model Architecture" width="800" />


## Capabilities

Chroma 1.0 is capable of:
- **Speech Understanding:** Processing user audio input directly.
- **Multimodal Generation:** Generating coherent text and speech responses simultaneously.
- **Voice Cloning:** utilizing reference audio prompts to guide speech generation style.

## Usage

### Installation

Ensure you have the necessary dependencies installed. You may need the latest versions of `transformers` and `torch`.

```bash
pip install transformers torch
```

### Loading the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "FlashLabs/Chroma-4B" # Or local path

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    device_map="auto"
)

# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

### Inference Example

Here is how to perform a simple conversation with audio input and audio output:

```python
import torch
from IPython.display import Audio



# Construct conversation history
system_prompt = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)
conversation = [[
    {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt}
        ],
    },
    {
        "role": "user",
        "content": [
            # Input audio file path
            {"type": "audio", "audio": "assets/make_taco.wav"}, 
        ],
    },
]]

# Provide reference audio/text for style or context
prompt_text = ["War and bloodshed throughout the world."]
prompt_audio = ["assets/reference_audio.wav"]

# Process inputs
inputs = processor(
    conversation,
    add_generation_prompt=True, 
    tokenize=False,
    prompt_audio=prompt_audio,
    prompt_text=prompt_text
)

# Move inputs to device
device = model.device
inputs = {k: v.to(device) for k, v in inputs.items()}

# 2. Generate
output = model.generate(
    **inputs, 
    max_new_tokens=100, 
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    use_cache=True
)

# 3. Decode Audio
# The model outputs raw tokens; we decode the audio part using the codec
audio_values = model.codec_model.decode(output.permute(0, 2, 1)).audio_values

# Save or play audio (e.g., in Jupyter)
Audio(audio_values[0].cpu().detach().numpy(), rate=24_000)
```

## Citation

If you use Chroma in your research, please cite:

```bibtex
@misc{chen2026flashlabschroma10realtime,
      title={FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning}, 
      author={Tanyu Chen and Tairan Chen and Kai Shen and Zhenghua Bao and Zhihui Zhang and Man Yuan and Yi Shi},
      year={2026},
      eprint={2601.11141},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.11141}, 
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 FlashLabs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Contact

For questions or issues, please contact: chroma@flashlabs.ai
