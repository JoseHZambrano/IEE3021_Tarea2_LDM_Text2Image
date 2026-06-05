from pathlib import Path
import torch

from src.generation import load_pipeline, generate_image
from src.utils import save_metadata_csv, create_image_grid, save_summary_txt


def main():
    print("Torch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_id = "stabilityai/sdxl-turbo"

    part_a_dir = Path("outputs") / "part_a"
    images_dir = part_a_dir / "images"
    figures_dir = part_a_dir / "figures"
    tables_dir = part_a_dir / "tables"

    images_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    prompt_records = [
        {
            "prompt_id": "A01",
            "prompt": "a futuristic city at night with neon lights and flying vehicles, highly detailed",
            "seed": 1001,
        },
        {
            "prompt_id": "A02",
            "prompt": "a realistic portrait of an elderly scientist in a laboratory, soft light, detailed face",
            "seed": 1002,
        },
        {
            "prompt_id": "A03",
            "prompt": "a fantasy dragon resting on a mountain, cinematic lighting, highly detailed",
            "seed": 1003,
        },
        {
            "prompt_id": "A04",
            "prompt": "An ant in the foreground, centered on a little hill, hyper-realistic, looking up and to the right side, like an inspired ant. Behind it, there is a colony or  army of ants fighting. The scene si a set at night, with a clear sky and a full moon.",
            "seed": 1004,
        },
        {
            "prompt_id": "A05",
            "prompt": "A autumn landscape of a park filled with trees and some people walking. The people are wearing a dark-to-brown color palette, admid falling leaves. An orange setting sun casts an afternoon glow, with a light mist hanging in the air",
            "seed": 1005,
        },
        {
            "prompt_id": "A06",
            "prompt": "a bowl of ramen on a wooden table, steam rising, food photography, realistic",
            "seed": 1006,
        },
        {
            "prompt_id": "A07",
            "prompt": "a medieval castle in the rain, dramatic clouds, cinematic atmosphere",
            "seed": 1007,
        },
        {
            "prompt_id": "A08",
            "prompt": "a cute robot reading a book in a library, warm light, illustration style",
            "seed": 1008,
        },
        {
            "prompt_id": "A09",
            "prompt": "an astronaut walking through a colorful alien forest, vibrant, detailed",
            "seed": 1009,
        },
        {
            "prompt_id": "A10",
            "prompt": "An aerial view of Monaco's beachfront and nearby treeline, with a Formula 1 race in the background. Midday, clear sunny sky, a few residents watching from windows, no crowds, only two or three cars on the track.",
            "seed": 1010,
        },
    ]

    pipe, device = load_pipeline(model_id=model_id)

    results = []

    for item in prompt_records:
        prompt_id = item["prompt_id"]
        prompt = item["prompt"]
        seed = item["seed"]

        print(f"\nGenerando {prompt_id} | seed={seed}")

        image_path = images_dir / f"{prompt_id}_seed{seed}.png"

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

        record["prompt_id"] = prompt_id
        record["model_id"] = model_id
        record["device"] = str(device)

        results.append(record)

        print(f"Imagen guardada en: {image_path}")
        print(f"Tiempo: {record['time_seconds']:.2f} s")

    csv_path = tables_dir / "part_a_metadata.csv"
    grid_path = figures_dir / "part_a_grid.png"
    summary_path = tables_dir / "part_a_summary.txt"

    save_metadata_csv(results, csv_path)
    create_image_grid(results, grid_path, ncols=5, figsize=(16, 8))
    save_summary_txt(results, summary_path)

    print("\nParte A completada.")
    print(f"Metadata CSV: {csv_path}")
    print(f"Grilla: {grid_path}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()