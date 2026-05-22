"""Hugging Face Spaces launcher for the DecodeShare demo."""

from demo.app import build_app


app = build_app()


if __name__ == "__main__":
    app.launch()
