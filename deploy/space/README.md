---
title: MasriSwitch-TTS
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.25.2
python_version: 3.10.13
app_file: app.py
license: apache-2.0
models:
  - Tarek737/MasriSwitch-TTS
short_description: Experimental Egyptian Arabic and English speech
---

# MasriSwitch-TTS

AI-generated Egyptian Arabic and English speech using a fixed, consented voice.
No voice uploads. This experimental model missed its accuracy target; see the
[measured model card](https://huggingface.co/Tarek737/MasriSwitch-TTS).

Runs on free ZeroGPU, with one request at a time and a 200-character limit.
Enter a short sentence, click **Generate speech**, wait for the GPU queue,
and play or download the resulting WAV. Signing in may increase free quota.

The approved reference is supplied through Space secrets, never repository files.
The model and vocoder are downloaded at immutable revisions and hash-checked.
The Space uses PyTorch/torchaudio 2.8.0 for ZeroGPU compatibility; published
E0/E1 evaluation used 2.6.0. Its Gradio SDK is 5.25.2 to avoid the platform's
Gradio 5.35 MCP extra/Pydantic conflict with F5-TTS 1.1.7. The demo does not
provide MCP tools. No training occurs in this application.
