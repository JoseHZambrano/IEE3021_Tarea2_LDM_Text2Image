from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from src.generation import load_pipeline, generate_image


def save_metadata_csv(records, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df


def create_adversarial_grid(records, output_path, ncols=3):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(records)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, rec in enumerate(records):
        img = Image.open(rec["image_path"])
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f'{rec["case_id"]}\nseed={rec["seed"]}',
            fontsize=9
        )

    fig.suptitle("Parte B - Casos adversariales", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_txt(records, txt_path):
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    total_time = sum(r["time_seconds"] for r in records)
    avg_time = total_time / len(records) if records else 0.0

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Resumen Parte B - Casos adversariales\n")
        f.write("=====================================\n")
        f.write(f"Número de imágenes: {len(records)}\n")
        f.write(f"Tiempo total de inferencia: {total_time:.2f} s\n")
        f.write(f"Tiempo promedio por imagen: {avg_time:.2f} s\n")


def main():
    print("Torch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_id = "stabilityai/sdxl-turbo"

    output_dir = Path("outputs") / "part_b_adversarial"
    images_dir = output_dir / "images"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    images_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adversarial_prompts = [
        {
            "case_id": "C01",
            "case_type": "contradictory_prompt",
            "prompt": "a completely empty white room filled with hundreds of colorful objects, photorealistic",
        },
        {
            "case_id": "C02",
            "case_type": "spatial_counting_prompt",
            "prompt": "a red cube to the left of a blue sphere, a green pyramid behind them, exactly three objects, all reflected in a mirror, realistic studio lighting",
        },
        {
            "case_id": "C03",
            "case_type": "abstract_spanish_prompt",
            "prompt": "una representación visual de la incertidumbre en un teorema matemático, estilo científico, composición clara, iluminación suave",
        },
    ]

    seeds = [3001, 3002, 3003]

    pipe, device = load_pipeline(model_id=model_id)

    results = []

    for item in adversarial_prompts:
        case_id = item["case_id"]
        case_type = item["case_type"]
        prompt = item["prompt"]

        for seed in seeds:
            print(f"\nGenerando {case_id} | seed={seed}")

            image_path = images_dir / f"{case_id}_seed{seed}.png"

            record = generate_image(
                pipe=pipe,
                prompt=prompt,
                seed=seed,
                output_path=image_path,
                num_inference_steps=4,
                guidance_scale=0.0,
                height=512,
                width=512,
            )

            record["case_id"] = case_id
            record["case_type"] = case_type
            record["model_id"] = model_id
            record["device"] = str(device)

            results.append(record)

            print(f"Imagen guardada en: {image_path}")
            print(f"Tiempo: {record['time_seconds']:.2f} s")

    csv_path = tables_dir / "part_b_adversarial_metadata.csv"
    grid_path = figures_dir / "part_b_adversarial_grid.png"
    summary_path = tables_dir / "part_b_adversarial_summary.txt"

    save_metadata_csv(results, csv_path)
    create_adversarial_grid(results, grid_path, ncols=3)
    save_summary_txt(results, summary_path)

    print("\nCasos adversariales finalizados.")
    print(f"Metadata CSV: {csv_path}")
    print(f"Grilla: {grid_path}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()