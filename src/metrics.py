from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip(model_id="openai/clip-vit-base-patch32"):
    """
    Carga CLIP para calcular alineación texto-imagen y embeddings visuales.
    """
    device = get_device()

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    return model, processor, device


def load_image_rgb(image_path):
    image_path = Path(image_path)
    return Image.open(image_path).convert("RGB")


@torch.no_grad()
def compute_clip_score(model, processor, device, image, prompt):
    """
    Calcula similitud coseno entre embedding de imagen y embedding de texto.
    Se reporta también como CLIP score escalado por 100.
    """
    inputs = processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    image_emb = F.normalize(outputs.image_embeds, p=2, dim=-1)
    text_emb = F.normalize(outputs.text_embeds, p=2, dim=-1)

    cosine = torch.sum(image_emb * text_emb, dim=-1).item()
    clip_score = 100.0 * cosine

    return cosine, clip_score, image_emb.cpu().numpy()[0]


def compute_laplacian_sharpness(image):
    """
    Métrica no-reference de nitidez.
    Calcula la varianza del Laplaciano sobre la imagen en escala de grises.

    Valores mayores suelen indicar más bordes y mayor nitidez.
    """
    gray = image.convert("L")
    arr = np.asarray(gray).astype(np.float32) / 255.0

    padded = np.pad(arr, pad_width=1, mode="edge")

    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )

    return float(np.var(lap))


def evaluate_images_from_metadata(metadata_csv, output_csv):
    """
    Evalúa imágenes usando:
    - CLIP cosine similarity
    - CLIP score = 100*cosine
    - sharpness por varianza del Laplaciano

    Retorna:
    - dataframe con métricas por imagen
    - diccionario con embeddings de imagen por fila
    """
    metadata_csv = Path(metadata_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)

    clip_model, clip_processor, device = load_clip()

    rows = []
    image_embeddings = {}

    for idx, row in df.iterrows():
        image_path = Path(row["image_path"])
        prompt = row["prompt"]

        if not image_path.exists():
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

        image = load_image_rgb(image_path)

        clip_cosine, clip_score, image_emb = compute_clip_score(
            model=clip_model,
            processor=clip_processor,
            device=device,
            image=image,
            prompt=prompt,
        )

        sharpness = compute_laplacian_sharpness(image)

        record = row.to_dict()
        record["clip_cosine"] = clip_cosine
        record["clip_score"] = clip_score
        record["sharpness_laplacian_var"] = sharpness

        rows.append(record)
        image_embeddings[idx] = image_emb

        print(
            f"{idx+1:03d}/{len(df)} | "
            f"{record.get('prompt_id', record.get('case_id', 'NA'))} | "
            f"seed={record.get('seed', 'NA')} | "
            f"CLIP={clip_score:.3f} | "
            f"sharpness={sharpness:.6f}"
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_csv, index=False)

    return metrics_df, image_embeddings


def compute_group_diversity(metrics_df, image_embeddings, group_col):
    """
    Calcula diversidad visual promedio dentro de cada grupo.
    Usa distancia coseno promedio entre embeddings CLIP de imagen.

    diversity = mean(1 - cosine_similarity)
    """
    diversity_rows = []

    for group_value, group_df in metrics_df.groupby(group_col):
        indices = list(group_df.index)

        distances = []

        for i, j in combinations(indices, 2):
            emb_i = image_embeddings[i]
            emb_j = image_embeddings[j]

            cos = float(np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)))
            dist = 1.0 - cos
            distances.append(dist)

        diversity = float(np.mean(distances)) if distances else 0.0

        diversity_rows.append(
            {
                group_col: group_value,
                "diversity_clip_image_distance": diversity,
                "num_images": len(indices),
            }
        )

    return pd.DataFrame(diversity_rows)


def summarize_part_b(metrics_df, diversity_df, output_dir):
    """
    Genera resúmenes por prompt, nivel de detalle y tema.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_summary = (
        metrics_df
        .groupby(["prompt_id", "theme", "detail_level"], as_index=False)
        .agg(
            clip_score_mean=("clip_score", "mean"),
            clip_score_std=("clip_score", "std"),
            sharpness_mean=("sharpness_laplacian_var", "mean"),
            sharpness_std=("sharpness_laplacian_var", "std"),
        )
    )

    prompt_summary = prompt_summary.merge(
        diversity_df,
        on="prompt_id",
        how="left",
    )

    detail_summary = (
        prompt_summary
        .groupby("detail_level", as_index=False)
        .agg(
            clip_score_mean=("clip_score_mean", "mean"),
            sharpness_mean=("sharpness_mean", "mean"),
            diversity_mean=("diversity_clip_image_distance", "mean"),
        )
    )

    theme_summary = (
        prompt_summary
        .groupby("theme", as_index=False)
        .agg(
            clip_score_mean=("clip_score_mean", "mean"),
            sharpness_mean=("sharpness_mean", "mean"),
            diversity_mean=("diversity_clip_image_distance", "mean"),
        )
    )

    prompt_summary.to_csv(output_dir / "part_b_prompt_summary.csv", index=False)
    detail_summary.to_csv(output_dir / "part_b_detail_summary.csv", index=False)
    theme_summary.to_csv(output_dir / "part_b_theme_summary.csv", index=False)

    return prompt_summary, detail_summary, theme_summary


def summarize_adversarial(metrics_df, diversity_df, output_dir):
    """
    Genera resumen por caso adversarial.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_summary = (
        metrics_df
        .groupby(["case_id", "case_type"], as_index=False)
        .agg(
            clip_score_mean=("clip_score", "mean"),
            clip_score_std=("clip_score", "std"),
            sharpness_mean=("sharpness_laplacian_var", "mean"),
            sharpness_std=("sharpness_laplacian_var", "std"),
        )
    )

    case_summary = case_summary.merge(
        diversity_df,
        on="case_id",
        how="left",
    )

    case_summary.to_csv(output_dir / "part_b_adversarial_summary_metrics.csv", index=False)

    return case_summary