"""Gradio demo for trained CNN checkpoints.

Launch with::

    python -m spoofdet.app.demo --ckpt checkpoints/<exp_id>/best.pt

The interface can also be opened without a checkpoint to show a clear setup
message; it will not produce predictions until a valid trained checkpoint exists.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..eval.gradcam import GradCAM, band_occlusion, band_edges_hz
from ..infer import Detector


def build_app(ckpt=None, device="cpu"):
    import gradio as gr

    checkpoint = Path(ckpt).expanduser() if ckpt is not None else None
    unavailable_message = None
    det = cam_fn = None
    if checkpoint is None or not checkpoint.is_file():
        shown = checkpoint if checkpoint is not None else "<not specified>"
        unavailable_message = (
            f"Trained checkpoint not found: {shown}. Provide a Member B checkpoint, "
            "for example checkpoints/<exp_id>/best.pt. No prediction has been made."
        )
    else:
        try:
            det = Detector(checkpoint, device=device)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            unavailable_message = f"Checkpoint is unavailable or incompatible: {exc}. No prediction has been made."
        if det is not None and det.gradcam_supported:
            try:
                cam_fn = GradCAM(det.model)
            except (ValueError, RuntimeError) as exc:
                cam_fn = None
                xai_error = f"Grad-CAM unavailable for this checkpoint: {exc}"
            else:
                xai_error = None
        elif det is not None:
            xai_error = "Grad-CAM is not supported by this checkpoint architecture."

    def analyse(audio_path, threshold):
        if unavailable_message:
            return None, None, unavailable_message
        if audio_path is None:
            return None, None, "Please upload or record an audio clip first."
        try:
            mel = det.mel(audio_path)
            x = det.tensor(mel)
            probability = det.score_tensor(x)
            verdict = "SPOOF (synthetic)" if probability >= threshold else "BONA FIDE (real)"
            fig, ax = plt.subplots(1, 3 if cam_fn is not None else 2,
                                   figsize=(14, 4), gridspec_kw={"width_ratios": [3, 3, 2]}
                                   if cam_fn is not None else {"width_ratios": [3, 2]})
            ax[0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
            ax[0].set(title="Log-mel spectrogram", xlabel="Time frame", ylabel="Mel band")
            if cam_fn is not None:
                cam, _ = cam_fn(x, target_class=1)
                ax[1].imshow(mel, origin="lower", aspect="auto", cmap="gray")
                ax[1].imshow(cam[0], origin="lower", aspect="auto", cmap="jet", alpha=0.45)
                ax[1].set(title="Grad-CAM for spoof score", xlabel="Time frame", ylabel="Mel band")
                n_bands = min(8, int(det.cfg["n_mels"]))
                occlusion = band_occlusion(det.model, x, n_bands)[0]
                edges = band_edges_hz(det.cfg, n_bands)
                ax[2].barh([f"{lo / 1000:.1f}-{hi / 1000:.1f} kHz" for lo, hi in edges],
                           occlusion, color="tab:red")
                ax[2].set(title="Mel-band occlusion", xlabel="Decrease in P(spoof)")
            else:
                ax[1].axis("off")
                ax[1].text(0, 0.7, xai_error or "Explainability unavailable.", wrap=True)
            fig.tight_layout()
            label = {"Bona fide": 1 - probability, "Spoof": probability}
            result = (f"Verdict: {verdict} | P(spoof) = {probability:.3f} | "
                      f"threshold = {threshold:.2f}")
            if cam_fn is None:
                result += f" | {xai_error}"
            return label, fig, result
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
            plt.close("all")
            return None, None, f"Could not analyze audio: {exc}"

    with gr.Blocks(title="Synthetic Speech Detector") as demo:
        gr.Markdown(
            "# Synthetic speech detector\nCNN on the project's log-mel features. "
            "Research prototype; results may not generalize to unseen attacks."
        )
        if unavailable_message:
            gr.Markdown(f"**Model unavailable:** `{unavailable_message}`")
        with gr.Row():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                             label="Speech clip (WAV, FLAC, or MP3)")
            with gr.Column():
                threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                      label="Decision threshold on P(spoof)")
                button = gr.Button("Analyze", variant="primary")
        label = gr.Label(label="Class probabilities")
        result = gr.Textbox(label="Result", interactive=False)
        plot = gr.Plot(label="Spectrogram and explanation")
        button.click(analyse, [audio, threshold], [label, plot, result])
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None,
                        help="Member B checkpoint (default: none; the UI will explain the missing checkpoint)")
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    build_app(args.ckpt, args.device).launch(share=args.share)


if __name__ == "__main__":
    main()
