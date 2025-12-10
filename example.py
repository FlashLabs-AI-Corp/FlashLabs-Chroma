from transformers import AutoProcessor, AutoModel
from .chroma.modeling_chroma import ChromaForConditionalGeneration
from .chroma.processing_chroma import ChromaProcessor

# 自动下载模型
model_id = "FlashLabs/Chroma-4B"

# load model and processor
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant, your name is Chroma developed by FlashLabs."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "text": "./example/make_tacco.wav"
            }
        ]
    }
]

# ref audio and text
prompt_audio = "./example/prompt_audio/make_tacco.wav"
with open("./example/prompt_text/scarlett_johansson.txt") as f:
    prompt_text = f.read()

# prepare input ids for generation
inputs = processor(
    conversation,
    add_generation_prompt=True,
    tokenize=False,
    prompt_audio=prompt_audio,
    prompt_text=prompt_text
)

# inference
audio = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=True,
    output_audio=True
)
