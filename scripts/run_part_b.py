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

    part_b_dir = Path("outputs") / "part_b"
    images_dir = part_b_dir / "images"
    figures_dir = part_b_dir / "figures"
    tables_dir = part_b_dir / "tables"

    images_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    prompt_records = [
    {
        "prompt_id": "B01",
        "theme": "biomedical_laboratory",
        "detail_level": "short",
        "prompt": "a futuristic biomedical laboratory",
    },
    {
        "prompt_id": "B02",
        "theme": "biomedical_laboratory",
        "detail_level": "medium",
        "prompt": "a futuristic biomedical laboratory with glass walls, blue lighting, medical devices and a clean workspace",
    },
    {
        "prompt_id": "B03",
        "theme": "biomedical_laboratory",
        "detail_level": "detailed",
        "prompt": "a futuristic biomedical research laboratory with glass walls, blue neon lighting, advanced medical devices, microscopes, monitors, a clean central workspace, cinematic composition, high detail",
    },
    {
        "prompt_id": "B04",
        "theme": "scientist_portrait",
        "detail_level": "short",
        "prompt": "a portrait of an elderly scientist",
    },
    {
        "prompt_id": "B05",
        "theme": "scientist_portrait",
        "detail_level": "medium",
        "prompt": "a realistic portrait of an elderly scientist in a laboratory, soft lighting, detailed face",
    },
    {
        "prompt_id": "B06",
        "theme": "scientist_portrait",
        "detail_level": "detailed",
        "prompt": "a realistic close-up portrait of an elderly scientist wearing a white lab coat, standing inside a modern laboratory, soft lighting, detailed facial expression, shallow depth of field",
    },
    {
        "prompt_id": "B07",
        "theme": "autumn_park",
        "detail_level": "short",
        "prompt": "an autumn park landscape",
    },
    {
        "prompt_id": "B08",
        "theme": "autumn_park",
        "detail_level": "medium",
        "prompt": "an autumn park landscape with trees, falling leaves, people walking and warm afternoon light",
    },
    {
        "prompt_id": "B09",
        "theme": "autumn_park",
        "detail_level": "detailed",
        "prompt": "an autumn park landscape filled with trees and a few people walking, dark-to-brown clothing palette, falling leaves, orange setting sun, warm afternoon glow, light mist in the air",
    },
    {
        "prompt_id": "B10",
        "theme": "ant_scene",
        "detail_level": "short",
        "prompt": "a hyper-realistic ant on a small hill",
    },
    {
        "prompt_id": "B11",
        "theme": "ant_scene",
        "detail_level": "medium",
        "prompt": "a hyper-realistic ant in the foreground on a small hill, looking upward under a full moon",
    },
    {
        "prompt_id": "B12",
        "theme": "ant_scene",
        "detail_level": "detailed",
        "prompt": "a hyper-realistic ant in the foreground, centered on a small hill, looking upward and to the right as if inspired. In the background, a colony of ants is engaged in battle. Night scene, clear sky, full moon",
    },
    {
        "prompt_id": "B13",
        "theme": "monaco_race",
        "detail_level": "short",
        "prompt": "an aerial view of Monaco beachfront",
    },
    {
        "prompt_id": "B14",
        "theme": "monaco_race",
        "detail_level": "medium",
        "prompt": "an aerial view of Monaco beachfront with a Formula 1 race in the background, sunny midday",
    },
    {
        "prompt_id": "B15",
        "theme": "monaco_race",
        "detail_level": "detailed",
        "prompt": "an aerial view of Monaco beachfront and nearby treeline, with a Formula 1 race in the background. Midday, clear sunny sky, a few residents watching from windows, no crowds, only two or three cars on the track",
    },
    ]

    seeds = [2001, 2002, 2003]

    pipe, device = load_pipeline(model_id=model_id)

    results = []

    for item in prompt_records:
        prompt_id = item["prompt_id"]
        theme = item["theme"]
        detail_level = item["detail_level"]
        prompt = item["prompt"]

        for seed in seeds:
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
            record["theme"] = theme
            record["detail_level"] = detail_level
            record["model_id"] = model_id
            record["device"] = str(device)

            results.append(record)

            print(f"Imagen guardada en: {image_path}")
            print(f"Tiempo: {record['time_seconds']:.2f} s")

    csv_path = tables_dir / "part_b_metadata.csv"
    grid_path = figures_dir / "part_b_grid.png"
    summary_path = tables_dir / "part_b_summary.txt"

    save_metadata_csv(results, csv_path)
    create_image_grid(results, grid_path, ncols=5, figsize=(16, 8))
    save_summary_txt(results, summary_path)

    print("\nParte B, primera parte, completada.")
    print(f"Metadata CSV: {csv_path}")
    print(f"Grilla: {grid_path}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()