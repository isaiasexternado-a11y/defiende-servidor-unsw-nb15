# Defiende el servidor · UNSW-NB15

Proyecto del curso de MLOps (Universidad Externado de Colombia, 2026-II): clasificar conexiones de red como **normales o ataque** con UNSW-NB15, de forma trazable y reproducible.

Guía del reto: [`guia-conceptual-defiende-servidor-2026II.md`](guia-conceptual-defiende-servidor-2026II.md).
Datos, versión y atribución: [`docs/DATASET.md`](docs/DATASET.md).

| Herramienta | Uso |
|---|---|
| Git | Código y decisiones |
| DVC | Versiones de datos y modelos |
| MLflow | Registro de experimentos y modelos |
| Prefect | Orquestación del pipeline |

## Puesta en marcha

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # versiones exactas (o requirements.txt)

# Obtener los datos (dos opciones)
python -m src.data.download --full   # los 9 CSV desde la fuente oficial de UNSW, verifica SHA-256
dvc pull                             # desde el remoto DVC (cuando esté configurado)
```

## Estructura

```
data/raw/          datos originales (versionados con DVC, no con Git)
docs/              documentación del dataset y decisiones
models/            modelos entrenados (DVC)
notebooks/         exploración
src/data/          descarga y preparación
```
