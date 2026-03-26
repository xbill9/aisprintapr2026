import argparse

import kagglehub


def download_gemma(model_slug: str = "google/gemma/transformers/2b-it/1"):
    """Downloads Gemma weights using kagglehub."""
    print(f"🚀 Downloading {model_slug} using kagglehub...")
    path = kagglehub.model_download(model_slug)
    print(f"✅ Model downloaded to: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma weights via kagglehub")
    parser.add_argument(
        "--slug",
        type=str,
        default="google/gemma/transformers/2b-it/1",
        help="Kaggle model slug",
    )
    args = parser.parse_args()

    download_gemma(args.slug)
