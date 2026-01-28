# examples/local_voice_chat_with_telemetry.py
# -----------------------------------------------------------------------------
# Local Voice Chat with Chroma-4B (Telemetry Edition)
# -----------------------------------------------------------------------------
# Same as the standard version but adds detailed performance metrics.
# Useful for benchmarking GPU performance and latency.
#
# Metrics tracked:
# - Load Time (Model initialization)
# - Input Duration
# - TTFT (Time To First Token) - approximated by Generation Time
# - Audio Generation Duration
# - RTF (Real-Time Factor)
# -----------------------------------------------------------------------------

import torch
import numpy as np
import pyaudio
import wave
import os
import sys
import time
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# --- Configuration ---
MODEL_ID = "FlashLabs/Chroma-4B"
MAX_MEMORY = {0: "4GB", "cpu": "8GB"} 
SAMPLE_RATE = 24000 

print("🚀 Initializing Local Chroma (Telemetry Mode)...")
t0 = time.time()

# --- 1. Load Model ---
# ... (Standard Loading) ...
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
    print(f"✅ Model loaded! (Took {time.time() - t0:.2f}s)")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# --- 2. Audio & Telemetry Utils ---
p = pyaudio.PyAudio()

def record_audio_memory(duration=5):
    chunk = 1024
    rate = 16000
    print(f"🎤 Recording ({duration}s)...")
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
        frames = [stream.read(chunk) for _ in range(0, int(rate / chunk * duration))]
        stream.stop_stream()
        stream.close()
        
        # Save temp file for stability
        with wave.open("telemetry_input.wav", 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
        return "telemetry_input.wav"
    except Exception as e:
        print(f"❌ Mic Error: {e}")
        return None

def play_audio_memory(audio_numpy):
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    
    t_start_play = time.time()
    stream.write(audio_int16.tobytes())
    t_end_play = time.time()
    
    stream.stop_stream()
    stream.close()
    return t_end_play - t_start_play

# --- 3. Main Loop ---
def main():
    if not os.path.exists("make_taco.wav"):
         print("❌ Error: 'make_taco.wav' missing.")
         sys.exit(1)
         
    prompt_audio = ["make_taco.wav"]
    prompt_text = ["Tell me a story."]
    
    print("\n✨ Ready (Telemetry Mode)! Ctrl+C to exit.")
    
    try:
        while True:
            input("\n🔵 Press ENTER...")
            
            # --- METRIC: Input ---
            t_in_start = time.time()
            user_audio = record_audio_memory()
            if not user_audio: continue
            t_in_end = time.time()
            input_dur = t_in_end - t_in_start
            
            print(f"   [Metric] Input Process Time: {input_dur:.2f}s")
            print("🧠 Thinking...")
            
            # Prepare Inputs
            conversation = [[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "audio", "audio": user_audio}]}
            ]]
            inputs = processor(
                conversation, add_generation_prompt=True, tokenize=False,
                prompt_audio=prompt_audio, prompt_text=prompt_text
            )
            device = model.device
            model_inputs = {k: v.to(device) for k, v in inputs.items() if k != "prompt_text"}
            # Fix: Keep input_values in float32 for Mimi Codec, cast others to float16
            for k, v in model_inputs.items():
               if v.dtype.is_floating_point:
                   if k == "input_values":
                       model_inputs[k] = v.to(dtype=torch.float32)
                   else:
                       model_inputs[k] = v.to(dtype=torch.float16)
            
            # --- METRIC: Generation ---
            t_gen_start = time.time()
            output = model.generate(**model_inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
            t_gen_end = time.time()
            gen_dur = t_gen_end - t_gen_start
            
            # --- METRIC: Decoding ---
            t_dec_start = time.time()
            audio_values = model.codec_model.decode(output.permute(0, 2, 1)).audio_values
            audio_numpy = audio_values[0].cpu().detach().numpy()
            
            if audio_numpy.ndim > 1 and audio_numpy.shape[0] < audio_numpy.shape[1]:
                audio_numpy = audio_numpy.T
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.mean(axis=1)
            t_dec_end = time.time()
            dec_dur = t_dec_end - t_dec_start
            
            audio_length_seconds = len(audio_numpy) / SAMPLE_RATE
            
            # --- REPORT ---
            print(f"   [Metric] Generation Time: {gen_dur:.2f}s")
            print(f"   [Metric] Decoding Time:   {dec_dur:.2f}s")
            print(f"   [Metric] Audio Duration:  {audio_length_seconds:.2f}s")
            # Real Time Factor = Time to Generate / Audio Duration
            # RTF < 1 means faster than real time.
            rtf = gen_dur / audio_length_seconds if audio_length_seconds > 0 else 0
            print(f"   [Metric] RTF:             {rtf:.2f}x (Lower is better)")
            print(f"   [Info] Output Shape:      {audio_numpy.shape}")

            print("🔊 Speaking...")
            play_audio_memory(audio_numpy)

    except KeyboardInterrupt:
        print("\n👋 Bye!")
        p.terminate()

if __name__ == "__main__":
    main()
