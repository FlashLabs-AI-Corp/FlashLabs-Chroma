# Proposal: Low-VRAM Inference Script (4GB GPU Support)

Hi everyone! 👋

I managed to run **Chroma-4B** successfully on a consumer laptop GPU (**RTX 3050 Ti 4GB**) using 4-bit quantization (`bitsandbytes`) and careful memory offloading.

**Performance:**
- **RTF (Real-Time Factor):** ~2.60x (on 4GB VRAM + 8GB RAM Offload)
- **Stability:** Rock solid (no OOM crashes).

I realized many developers might be struggling with the 14GB VRAM requirement, so I created a clean, minimal "Walkie-Talkie" script to demonstrate how to run this locally.

### Included Files:
1.  **`local_voice_chat.py`**: A clean, light script for "Talk & Listen" interaction.
    - Uses direct memory playback (no disk I/O latency).
    - Robust fallback if no custom voice prompt is found.
2.  **`local_voice_chat_with_telemetry.py`**: Adds performance metrics (TF, RTF, Input/Output Latency).

I would love to contribute these as examples under `examples/local_inference/` to help the community access this amazing model on lower-end hardware.

Best regards,
A Fan of Chroma! 🚀
