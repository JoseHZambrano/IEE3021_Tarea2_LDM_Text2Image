### Tarea 2 - Evaluación experimental de un modelo de difusión latente texto-imagen


Este repositorio contiene el código utilizado para la Tarea 2 del curso IEE3021 - Modelos Generativos.

El objetivo fue evaluar experimentalmente el comportamiento de un modelo de difusión latente texto-imagen frente a distintos prompts, semillas de ruido y métricas cuantitativas.



#### Modelo utilizado

Se utilizó `stabilityai/sdxl-turbo` mediante la biblioteca `diffusers`.



#### Estructura

* `src/`: funciones auxiliares de generación y métricas.
* `scripts/`: scripts ejecutables para cada parte del experimento.
* `outputs/`: tablas, figuras y resultados generados.
* `requirements.txt`: dependencias del entorno.



#### Ejecución

Activar entorno:

```powershell
.\\.venv\\Scripts\\Activate.ps1

Ejecutar Parte A:

```powershell
py -m scripts.run_part_a

Ejecutar Parte B:

```powershell
py -m scripts.run_part_b

Ejecutar casos adversariales:

```powershell
py -m scripts.run_part_b_adversarial

Ejecutar métricas:

```powershell
py -m scripts.run_metrics

FIN

Jose H. Zambrano P.
IEE3021 - Modelos Generativos
Pontificia Universidad Católica de Chile
2026