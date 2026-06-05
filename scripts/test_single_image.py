from pathlib import Path
import pandas as pd
import torch

from src.generation import load_pipeline, generate_image


def main():
    print("Torch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_id = "stabilityai/sdxl-turbo"

    prompt = (
        "a small futuristic research laboratory, clean composition, "
        "soft lighting, scientific instruments, high detail"
    )
    seed = 1234

    output_dir = Path("outputs") / "part_a"
    image_path = output_dir / "images" / "test_single_image_seed1234.png"
    table_path = output_dir / "tables" / "test_single_image_metadata.csv"

    pipe, device = load_pipeline(model_id=model_id)

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

    record["model_id"] = model_id
    record["device"] = str(device)

    pd.DataFrame([record]).to_csv(table_path, index=False)

    print("\nImagen generada correctamente.")
    print(f"Ruta imagen: {image_path}")
    print(f"Ruta metadata: {table_path}")
    print(f"Tiempo de inferencia: {record['time_seconds']:.2f} s")


if __name__ == "__main__":
    main()