from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metadata_csv(records, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df


def create_image_grid(records, output_path, ncols=5, figsize=(16, 8)):
    """
    Crea una grilla de imágenes con título simple:
    prompt_id y seed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(records)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = [axes]

    axes = list(axes) if isinstance(axes, list) else axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, rec in enumerate(records):
        img = Image.open(rec["image_path"])
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f'{rec["prompt_id"]}\nseed={rec["seed"]}',
            fontsize=9
        )

    fig.suptitle("Parte A - Exploración inicial", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_txt(records, txt_path):
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    total_time = sum(r["time_seconds"] for r in records)
    avg_time = total_time / len(records) if records else 0.0

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Resumen Parte A\n")
        f.write("================\n")
        f.write(f"Número de imágenes: {len(records)}\n")
        f.write(f"Tiempo total de inferencia: {total_time:.2f} s\n")
        f.write(f"Tiempo promedio por imagen: {avg_time:.2f} s\n")