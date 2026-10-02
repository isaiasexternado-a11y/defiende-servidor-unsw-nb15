# Dataset: UNSW-NB15

## Origen

- **Creador:** Cyber Range Lab, UNSW Canberra (Australian Centre for Cyber Security), herramienta IXIA PerfectStorm.
- **Página oficial:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Carpeta oficial (SharePoint UNSW):** enlace en `src/data/download.py` (`SHARE_URL`).
- **Fecha de descarga:** 2026-10-02.

## Versión utilizada

Partición oficial *Training and Testing Sets*. Los SHA-256 están fijados en
`src/data/download.py` y registrados en `data/raw/MANIFEST.json`; los archivos
se versionan con DVC (`data/raw.dvc`).

| Archivo | Filas | SHA-256 |
|---|---:|---|
| `UNSW_NB15_training-set.csv` | 175.341 | `bec7dd5e…0648fa` |
| `UNSW_NB15_testing-set.csv` | 82.332 | `734fe664…c0a559` |
| `NUSW-NB15_features.csv` | 49 variables | `c55f19cc…9cafce` |

Columnas: 45 (`id`, 42 variables de flujo, `attack_cat`, `label`).

- `label`: 0 = normal, 1 = ataque → **objetivo principal** (binario).
- `attack_cat`: familia del ataque (Normal + 9 familias) → **ampliación** (multiclase). No se debe usar como predictor del binario: es la respuesta.
- `id`: consecutivo del archivo, no una propiedad de la conexión. Revisar si filtra el orden de las clases antes de usarlo.

Observaciones iniciales:

- El encabezado del training CSV trae un BOM UTF-8 (`﻿id`); leer con `encoding="utf-8-sig"`.
- Los nombres "training" y "testing" son los de UNSW. Según la guía, el set de test debe usarse una sola vez al final; las particiones de validación se sacan del training.
- Los CSV completos (`UNSW-NB15_1..4.csv`, ~2,5 M registros) se pueden bajar con `python -m src.data.download --full`.

## Condiciones de uso

> "Free use of the UNSW-NB15 dataset for academic research purposes is hereby granted in perpetuity." El uso comercial requiere autorización de los autores (Dr. Nour Moustafa).

Citas obligatorias para uso académico:

1. Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *MilCIS 2015*.
2. Moustafa, N., & Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset. *Information Security Journal: A Global Perspective*.
3. Moustafa, N., Slay, J., & Creech, G. (2017). Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks. *IEEE Transactions on Big Data*.
4. Moustafa, N., Creech, G., & Slay, J. (2017). Big data analytics for intrusion detection system: statistical decision-making using finite Dirichlet mixture models. *Data Analytics and Decision Support for Cybersecurity*, Springer.
5. Sarhan, M., Layeghy, S., Moustafa, N., & Portmann, M. (2020). NetFlow datasets for machine learning-based network intrusion detection systems. *Big Data Technologies and Applications*, Springer.

## Limitaciones

Tráfico generado en laboratorio en 2015. Sirve para practicar el ciclo de ML, no prueba eficacia sobre una red actual.
