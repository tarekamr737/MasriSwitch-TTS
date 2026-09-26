"""Public fixed-voice demo on free ZeroGPU; no uploads or paid fallbacks."""

import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HOME", "/tmp/masriswitch/hf-cache")
os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/masriswitch/generated")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402
import spaces  # noqa: E402

from masriswitch.infer.engine import F5Engine  # noqa: E402
from masriswitch.infer.hosted import hosted_engine_files  # noqa: E402

root = Path("/tmp/masriswitch/private")
engine = F5Engine(hosted_engine_files(root, os.environ), device="cuda")


@spaces.GPU(duration=30)
def synthesize(text: str):
    if not text.strip() or len(text) > 200:
        raise gr.Error("Enter between 1 and 200 characters.")
    waveform, rate = engine.synthesize(text)
    return rate, waveform


with gr.Blocks(title="MasriSwitch-TTS", delete_cache=(600, 600)) as demo:
    gr.Markdown(
        "# MasriSwitch-TTS\n"
        "Egyptian Arabic + English speech with a fixed, consented voice. "
        "**All audio is AI-generated.**\n\n"
        "Experimental: pronunciation and numbers may be wrong. "
        "Free GPU access has a queue and daily limits. Start with a short sentence."
    )
    text = gr.Textbox(label="Text to speak", lines=3, max_length=200)
    gr.Examples(
        ["أهلا بيك في خدمة العملاء.", "ال order جاهز للتوصيل.", "ممكن تعمل reset لل password؟"],
        inputs=text,
    )
    generate = gr.Button("Generate speech", variant="primary")
    audio = gr.Audio(label="AI-generated speech", interactive=False)
    generate.click(
        synthesize, inputs=text, outputs=audio, api_name="synthesize", concurrency_limit=1
    )
    gr.Markdown(
        "[Model and measured limitations](https://huggingface.co/Tarek737/MasriSwitch-TTS) · "
        "[Source code](https://github.com/tarekamr737/MasriSwitch-TTS)"
    )

demo.queue(max_size=8, default_concurrency_limit=1).launch(
    server_name="0.0.0.0", server_port=7860, blocked_paths=[str(root)], show_error=False
)
