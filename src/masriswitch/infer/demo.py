"""Minimal public demo; no reference-audio upload control."""

from __future__ import annotations

from masriswitch.infer.engine import SynthesisEngine


def create_demo(engine: SynthesisEngine | None = None):  # type: ignore[no-untyped-def]
    import gradio as gr

    with gr.Blocks(title="MasriSwitch-TTS") as demo:
        gr.Markdown(
            "# MasriSwitch-TTS\nAudio is AI-generated. "
            "Synthesis uses a fixed, approved reference voice when available."
        )
        if engine is None:
            gr.Markdown(
                "Synthesis is unavailable until an approved voice and checkpoint are installed."
            )
        text = gr.Textbox(label="Arabic / English text", max_length=500)
        audio = gr.Audio(label="Generated audio", interactive=False)
        button = gr.Button("Generate", interactive=engine is not None)
        if engine is not None:

            def generate(value: str):  # type: ignore[no-untyped-def]
                waveform, sample_rate = engine.synthesize(value)
                return sample_rate, waveform

            button.click(generate, inputs=text, outputs=audio)
    return demo
