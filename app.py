"""Entry point for Hugging Face Spaces (SDK: Gradio). Put a small best.pt next to this file as model/best.pt."""
from spoofdet.app.demo import build_app
demo = build_app("model/best.pt")
if __name__ == "__main__":
    demo.launch()
