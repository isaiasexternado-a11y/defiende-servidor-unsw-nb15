# Comandos ejecutados (2026-10-27, WSL2, Python 3.14.4)

```bash
# DagsHub privado: credenciales por variables de entorno (nunca en el log ni en Git)
export DAGSHUB_USER=isaiasexternado-a11y DAGSHUB_TOKEN=<token>
bash scripts/recuperacion.sh "$HOME/recuperacion-defiende-servidor" af36f31
```

El script clona el repo en una carpeta nueva, crea un venv nuevo, instala `requirements-lock.txt`
sin caché de pip, hace `dvc pull` con caché vacía, verifica SHA-256 de datos originales y
preparados, descarga v1 y v2 de Hugging Face por SHA con `HF_HOME` nuevo, predice las 25 filas
fijas de `evidencias/recuperacion/entrada.csv` y compara contra `antes_v1.csv` / `antes_v2.csv`
(tolerancia 1e-9 en `score_attack`). Resultado: `comparacion.json` (diferencia máxima 0.0).
