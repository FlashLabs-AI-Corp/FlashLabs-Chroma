# examples/local_voice_chat.py
# -----------------------------------------------------------------------------
# Local Voice Chat with Chroma-4B (Light Edition)
# -----------------------------------------------------------------------------
# Minimalist script to run Chroma-4B on consumer hardware (4GB VRAM).
# Features:
# - 4-bit Quantization (BitsAndBytes)
# - Direct Memory Playback (No WAV file overhead)
# - Simplified Paths (Requires make_taco.wav in same folder)
#
# Requirements:
#   pip install torch transformers bitsandbytes pyaudio numpy
# -----------------------------------------------------------------------------

import torch
import numpy as np
import pyaudio
import wave
import os
import sys
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# --- Configuration ---
MODEL_ID = "FlashLabs/Chroma-4B"
MAX_MEMORY = {0: "4GB", "cpu": "8GB"} # Critical for 3050/4050 cards
SAMPLE_RATE = 24000 # Chroma native rate

print("🚀 Initializing Local Chroma (Light)...")

# --- 1. Load Model (4-Bit) ---
print("⏳ Loading model...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        max_memory=MAX_MEMORY
    )
    if hasattr(model, "codec_model"):
        model.codec_model.to(dtype=torch.float32)
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# --- 2. Audio IO (Memory Optimized) ---
p = pyaudio.PyAudio()

def record_audio(duration=5):
    """Records directly to memory buffer."""
    chunk = 1024
    rate = 16000
    print(f"🎤 Recording ({duration}s)...")
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
        frames = [stream.read(chunk) for _ in range(0, int(rate / chunk * duration))]
        stream.stop_stream()
        stream.close()
        
        # Convert to temp wav in memory (or file if processor needs path)
        # Processor supports raw audio? Usually yes, but path is safer for this specific processor version.
        # We will keep 'input.wav' just for input stability, but output will be memory.
        with wave.open("input.wav", 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
        return "input.wav"
    except Exception as e:
        print(f"❌ Mic Error: {e}")
        return None

def play_audio_memory(audio_numpy):
    """Plays float32 numpy array directly without saving to disk."""
    # Convert float32 [-1, 1] to int16
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    stream.write(audio_int16.tobytes())
    stream.stop_stream()
    stream.close()

# --- 3. Main Loop ---
def main():
    # Setup Voice Style from local file
    if not os.path.exists("make_taco.wav"):
        print("❌ Error: 'make_taco.wav' not found in current directory.")
        sys.exit(1)
        
    prompt_audio = ["make_taco.wav"]
    prompt_text = ["Tell me a story."] # Default text prompt match
    
    print("\n✨ Ready (Light Mode)! Ctrl+C to exit.")
    
    try:
        while True:
            input("\n🔵 Press ENTER...")
            
            user_audio = record_audio()
            if not user_audio: continue
            
            print("🧠 Thinking...")
            
            conversation = [[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "audio", "audio": user_audio}]}
            ]]
            
            inputs = processor(
                conversation, add_generation_prompt=True, tokenize=False,
                prompt_audio=prompt_audio, prompt_text=prompt_text
            )
            
            # GPU Move
            device = model.device
            model_inputs = {k: v.to(device) for k, v in inputs.items() if k != "prompt_text"} # filter args if needed
            # Explicit cast
            # NOTE: We keep 'input_values' (audio) in float32 because we cast codec_model to float32 above.
            # reducing it to float16 causes "Input type (Half) and bias type (float) should be the same" error.
            for k, v in model_inputs.items():
               if v.dtype.is_floating_point:
                   if k == "input_values":
                       model_inputs[k] = v.to(dtype=torch.float32)
                   else:
                       model_inputs[k] = v.to(dtype=torch.float16)

            # Generate
            output = model.generate(**model_inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
            
            # Decode to Memory
            audio_values = model.codec_model.decode(output.permute(0, 2, 1)).audio_values
            audio_numpy = audio_values[0].cpu().detach().numpy()
            
            # Transpose Fix
            if audio_numpy.ndim > 1 and audio_numpy.shape[0] < audio_numpy.shape[1]:
                audio_numpy = audio_numpy.T
            
            # Flatten to 1D if stereo
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.mean(axis=1)

            print("🔊 Speaking...")
            play_audio_memory(audio_numpy)

    except KeyboardInterrupt:
        print("\n👋 Bye!")
        p.terminate()

if __name__ == "__main__":
    main()
