from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.metrics import (
    evaluate_images_from_metadata,
    compute_group_diversity,
    summarize_part_b,
    summarize_adversarial,
)


def plot_bar(df, x_col, y_col, title, ylabel, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Torch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    output_dir = Path("outputs") / "part_c"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Parte B: 45 imágenes sistemáticas
    # ------------------------------------------------------------
    part_b_metadata = Path("outputs") / "part_b" / "tables" / "part_b_metadata.csv"

    if not part_b_metadata.exists():
        raise FileNotFoundError(f"No se encontró: {part_b_metadata}")

    print("\nEvaluando imágenes de la Parte B...")
    part_b_metrics_csv = tables_dir / "part_b_image_metrics.csv"

    part_b_metrics, part_b_embeddings = evaluate_images_from_metadata(
        metadata_csv=part_b_metadata,
        output_csv=part_b_metrics_csv,
    )

    part_b_diversity = compute_group_diversity(
        metrics_df=part_b_metrics,
        image_embeddings=part_b_embeddings,
        group_col="prompt_id",
    )

    part_b_diversity.to_csv(
        tables_dir / "part_b_prompt_diversity.csv",
        index=False,
    )

    prompt_summary, detail_summary, theme_summary = summarize_part_b(
        metrics_df=part_b_metrics,
        diversity_df=part_b_diversity,
        output_dir=tables_dir,
    )

    print("\nResumen por nivel de detalle:")
    print(detail_summary)

    print("\nResumen por tema:")
    print(theme_summary)

    plot_bar(
        df=detail_summary,
        x_col="detail_level",
        y_col="clip_score_mean",
        title="CLIP Score promedio por nivel de detalle",
        ylabel="CLIP Score",
        output_path=figures_dir / "clip_score_by_detail_level.png",
    )

    plot_bar(
        df=detail_summary,
        x_col="detail_level",
        y_col="sharpness_mean",
        title="Nitidez promedio por nivel de detalle",
        ylabel="Varianza del Laplaciano",
        output_path=figures_dir / "sharpness_by_detail_level.png",
    )

    plot_bar(
        df=detail_summary,
        x_col="detail_level",
        y_col="diversity_mean",
        title="Diversidad visual promedio por nivel de detalle",
        ylabel="Distancia coseno promedio",
        output_path=figures_dir / "diversity_by_detail_level.png",
    )

    # ------------------------------------------------------------
    # Casos adversariales
    # ------------------------------------------------------------
    adversarial_metadata = (
        Path("outputs")
        / "part_b_adversarial"
        / "tables"
        / "part_b_adversarial_metadata.csv"
    )

    if adversarial_metadata.exists():
        print("\nEvaluando casos adversariales...")

        adversarial_metrics_csv = tables_dir / "part_b_adversarial_image_metrics.csv"

        adv_metrics, adv_embeddings = evaluate_images_from_metadata(
            metadata_csv=adversarial_metadata,
            output_csv=adversarial_metrics_csv,
        )

        adv_diversity = compute_group_diversity(
            metrics_df=adv_metrics,
            image_embeddings=adv_embeddings,
            group_col="case_id",
        )

        adv_diversity.to_csv(
            tables_dir / "part_b_adversarial_diversity.csv",
            index=False,
        )

        case_summary = summarize_adversarial(
            metrics_df=adv_metrics,
            diversity_df=adv_diversity,
            output_dir=tables_dir,
        )

        print("\nResumen adversarial:")
        print(case_summary)
    else:
        print("\nNo se encontró metadata de casos adversariales. Se omite esta parte.")

    # ------------------------------------------------------------
    # Parte A: métricas opcionales para las 10 imágenes iniciales
    # ------------------------------------------------------------
    part_a_metadata = Path("outputs") / "part_a" / "tables" / "part_a_metadata.csv"

    if part_a_metadata.exists():
        print("\nEvaluando imágenes de la Parte A...")

        part_a_metrics_csv = tables_dir / "part_a_image_metrics.csv"

        part_a_metrics, _ = evaluate_images_from_metadata(
            metadata_csv=part_a_metadata,
            output_csv=part_a_metrics_csv,
        )

        representative_ids = ["A02", "A05", "A10"]
        representatives = part_a_metrics[
            part_a_metrics["prompt_id"].isin(representative_ids)
        ].copy()

        representatives.to_csv(
            tables_dir / "part_a_representative_metrics.csv",
            index=False,
        )

        print("\nMétricas de imágenes representativas Parte A:")
        print(
            representatives[
                ["prompt_id", "seed", "clip_score", "sharpness_laplacian_var"]
            ]
        )
    else:
        print("\nNo se encontró metadata de Parte A. Se omite esta parte.")

    print("\nEvaluación cuantitativa finalizada.")
    print(f"Tablas guardadas en: {tables_dir}")
    print(f"Figuras guardadas en: {figures_dir}")


if __name__ == "__main__":
    main()