"""Web demo:  upload / record speech -> verdict (bona fide vs synthetic) + mel-spectrogram + Grad-CAM + band importance.
   python -m spoofdet.app.demo --ckpt checkpoints/<exp_id>/best.pt            (opens http://127.0.0.1:7860)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..eval.gradcam import GradCAM, band_occlusion, band_edges_hz
from ..infer import Detector


def build_app(ckpt, device="cpu"):
    import gradio as gr
    det = Detector(ckpt, device)
    cam_fn = GradCAM(det.model)
    cfg = det.cfg

    def analyse(audio_path, threshold):
        if audio_path is None:
            return None, None, "Please upload or record an audio clip first."
        mel = det.mel(audio_path)
        x = det.tensor(mel)
        cam, p = cam_fn(x)
        occ = band_occlusion(det.model, x, 8)[0]
        p = float(p[0])
        verdict = "SYNTHETIC (spoof)" if p >= threshold else "REAL (bona fide)"
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [3, 3, 2]})
        im = ax[0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        ax[0].set(title="Log-mel spectrogram (4 s)", xlabel="frame (10 ms)", ylabel="mel band")
        ax[1].imshow(mel, origin="lower", aspect="auto", cmap="gray")
        ax[1].imshow(cam[0], origin="lower", aspect="auto", cmap="jet", alpha=0.45)
        ax[1].set(title="Grad-CAM: regions that push towards 'spoof'", xlabel="frame")
        edges = band_edges_hz(cfg, 8)
        ax[2].barh([f"{lo/1000:.1f}-{hi/1000:.1f} kHz" for lo, hi in edges], occ, color="tab:red")
        ax[2].set(title="Band occlusion\n(drop in P(spoof))")
        fig.tight_layout()
        label = {"Synthetic": p, "Real": 1 - p}
        return label, fig, f"Verdict: {verdict}   |   P(spoof) = {p:.3f}   |   threshold = {threshold:.2f}"

    with gr.Blocks(title="Synthetic Speech Detector") as demo:
        gr.Markdown("# Synthetic speech detector\nCNN on log-mel spectrograms. Research prototype - "
                    "not reliable against attacks it has not seen, and not for legal/forensic use.")
        with gr.Row():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Speech clip")
            with gr.Column():
                thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Decision threshold on P(spoof)")
                btn = gr.Button("Analyse", variant="primary")
        lab = gr.Label(label="Probability")
        txt = gr.Textbox(label="Result", interactive=False)
        plot = gr.Plot(label="Explanation")
        btn.click(analyse, [audio, thr], [lab, plot, txt])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    build_app(a.ckpt, a.device).launch(share=a.share)
