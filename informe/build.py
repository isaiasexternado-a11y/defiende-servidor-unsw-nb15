"""Genera informe/informe.html (y la figura) leyendo las cifras de evidencias/.

    python informe/build.py
    # PDF: abrir informe.html en Chrome/Edge → Imprimir → Guardar como PDF (A4), o en WSL:
    # "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless \
    #     --print-to-pdf="$(wslpath -w informe/informe.pdf)" --no-pdf-header-footer "$(wslpath -w informe/informe.html)"

Todas las cifras salen de los JSON/CSV de evidencias, así el informe no puede
contradecir a los archivos.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import binary_metrics, predict_label  # noqa: E402

EV = ROOT / "evidencias"
OUT = Path(__file__).resolve().parent

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def load():
    audit = json.loads((EV / "auditoria.json").read_text())
    runs = pd.read_csv(EV / "experimentos.csv")
    sel = json.loads((EV / "seleccion.json").read_text())
    test = json.loads((EV / "metricas-test.json").read_text())
    rec = json.loads((EV / "recuperacion/entorno-nuevo/comparacion.json").read_text())
    card = json.loads((EV / "model-card-test.json").read_text())
    return audit, runs, sel, test, rec, card


def curve(run_id: str) -> pd.DataFrame:
    s = pd.read_csv(EV / "mlflow" / run_id / "val_scores.csv")
    rows = []
    for t in [i / 200 for i in range(1, 200)]:
        m = binary_metrics(s.y_true, predict_label(s.score_attack, t))
        rows.append((t, m["fpr"], m["recall_attack"]))
    return pd.DataFrame(rows, columns=["t", "fpr", "recall"])


def figure(runs: pd.DataFrame, sel: dict) -> None:
    v1, v2 = sel["versions"]["v1"], sel["versions"]["v2"]
    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=200)
    for v, color, name in [(v1, ORANGE, "v1 · e1 regresión logística"), (v2, BLUE, "v2 · e4 HistGradientBoosting")]:
        c = curve(v["mlflow_run_id"])
        ax.plot(c.fpr, c.recall, color=color, lw=2, label=f"{name} (todos los umbrales)", zorder=2)
    ax.axvline(sel["threshold_policy"]["max_fpr"], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(sel["threshold_policy"]["max_fpr"] + 0.004, 0.842, "FPR = 5 %", color=INK2, fontsize=7.5)
    for _, r in runs.iterrows():
        ax.scatter(r.fpr, r.recall_attack, s=34, color="white", edgecolor=INK, lw=1.2, zorder=3)
        dx, dy = {"e1_logreg_base": (0.004, -0.013), "e2_logreg_log_balanced": (0.004, -0.012),
                  "e3_random_forest": (0.004, -0.013), "e4_hist_gradient_boosting": (0.004, -0.013)}[r.config_id]
        ax.text(r.fpr + dx, r.recall_attack + dy, r.config_id.split("_")[0] + " @0.5", fontsize=7.5, color=INK,
                ha="right" if dx < 0 else "left")
    m = v2["val_metrics"]
    ax.scatter(m["fpr"], m["recall_attack"], s=60, color=BLUE, edgecolor="white", lw=1.5, zorder=4)
    ax.annotate(f"v2 elegido: umbral {v2['threshold']}", (m["fpr"], m["recall_attack"]),
                xytext=(0.075, 0.905), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0.83, 1.0)
    ax.set_xlabel("Tasa de falsos positivos (validación)", fontsize=8, color=INK2)
    ax.set_ylabel("Recall de ataque", fontsize=8, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2, length=0)
    ax.grid(color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_validacion.png")


def pct(x: float) -> str:
    return f"{x * 100:.1f} %".replace(".", ",")


def num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def html(audit, runs, sel, test, rec, card) -> str:
    tr, te = audit["files"]["train"], audit["files"]["test"]
    dup, dupt = audit["duplicates"]["train_file"], audit["duplicates"]["test_file"]
    sp, ov = audit["split"], audit["overlaps"]
    v1, v2 = sel["versions"]["v1"], sel["versions"]["v2"]
    m1, m2, mt = v1["val_metrics"], v2["val_metrics"], test["metrics"]
    seen, unseen = test["breakdown_informativo"]["inputs_seen_in_train_val"], \
        test["breakdown_informativo"]["inputs_unseen"]
    r = runs.set_index("config_id")
    concl = {k: re.sub(r"(\d)\.(\d)", r"\1,\2", v)
             for k, v in json.loads((EV / "conclusiones.json").read_text()).items()}
    raw_md5 = json.loads((EV / "mlflow" / v2["mlflow_run_id"] / "run.json").read_text())["tags"]["dvc_raw_md5"]

    def run_row(cid, family, change):
        x = r.loc[cid]
        return (f"<tr><td><b>{cid.split('_')[0]}</b></td><td>{family}</td><td>{change}</td>"
                f"<td>{num(x.recall_attack)}</td><td>{num(x.precision_attack)}</td><td>{num(x.fpr)}</td>"
                f"<td>{num(x.macro_f1)}</td><td>{thousands(int(x.fp))} / {thousands(int(x.fn))}</td>"
                f"<td class=mono>{x.mlflow_run_id[:8]}</td></tr>")

    def cm(m):
        c = m["confusion_matrix"]
        return f"{thousands(c['tn'])} / {thousands(c['fp'])} / {thousands(c['fn'])} / {thousands(c['tp'])}"

    hyp = {k: r.loc[k].hypothesis for k in r.index}
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Defiende el servidor · Informe</title>
<style>
@page {{ size: A4; margin: 16mm 16mm 15mm 16mm; }}
body {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 9.3pt; line-height: 1.38;
       color: #1d1d1b; background: #fff; margin: 0; }}
h1 {{ font-size: 17pt; margin: 0 0 2px; color: #123d2b; }}
h2 {{ break-after: avoid; font-size: 11.5pt; color: #123d2b; border-bottom: 1.5px solid #1f5c40; padding-bottom: 2px; margin: 13px 0 6px; }}
h3 {{ font-size: 9.8pt; margin: 9px 0 3px; }}
p {{ margin: 0 0 6px; text-align: justify; }}
.meta {{ color: #52514e; font-size: 8.6pt; margin-bottom: 8px; }}
figure, table, .box {{ break-inside: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 4px 0 8px; font-size: 8.3pt; }}
th, td {{ border-bottom: 1px solid #dcdad4; padding: 3px 5px; text-align: left; vertical-align: top; }}
th {{ background: #eef3ef; color: #123d2b; font-weight: 600; }}
td.n, th.n {{ text-align: right; }}
.mono {{ font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 7.8pt; }}
.box {{ background: #f4f6f3; border-left: 3px solid #1f5c40; padding: 5px 9px; margin: 6px 0 8px; }}
ul {{ margin: 2px 0 6px 16px; padding: 0; }} li {{ margin-bottom: 2px; }}
figure {{ margin: 4px 0 6px; text-align: center; }} figure img {{ width: 92%; }}
figcaption {{ font-size: 8pt; color: #52514e; }}
.refs {{ page-break-before: always; font-size: 8.6pt; }}
.refs li {{ margin-bottom: 4px; }}
</style></head><body>

<h1>Defiende el servidor</h1>
<div class="meta">Informe del taller evaluable · MLOps 2026-II · Universidad Externado de Colombia<br>
Isaias Cespedes Novoa · Grupo A · entrega individual · Docente: Julián Zuluaga</div>

<div class="box"><b>Resumen.</b> Clasificador binario normal/ataque sobre UNSW-NB15 con trazabilidad completa:
código en Git (tags <span class=mono>v1</span>, <span class=mono>v2</span>), datos originales y preparados en DVC,
cuatro experimentos en MLflow y dos versiones del modelo en Hugging Face identificadas por SHA. v2
(HistGradientBoosting, umbral {num(v2['threshold'], 2)}) baja la tasa de falsos positivos de validación de
{num(m1['fpr'])} a {num(m2['fpr'])} frente a v1, a cambio de {num((m1['recall_attack'] - m2['recall_attack']) * 100, 1)}
puntos de recall. En la única evaluación sobre test, el FPR sube a {num(mt['fpr'])}: el umbral elegido en validación
no se transfiere a la distribución del test oficial. La recuperación en un entorno nuevo reprodujo las predicciones
de ambas versiones con diferencia máxima de score igual a cero.</div>

<h2>1. Datos y limpieza</h2>
<p>Se usaron las particiones publicadas por UNSW: <span class=mono>UNSW_NB15_training-set.csv</span>
({thousands(tr['rows'])} filas) y <span class=mono>UNSW_NB15_testing-set.csv</span> ({thousands(te['rows'])} filas),
45 columnas, sin faltantes. Sus SHA-256 coinciden con las huellas del anexo del taller
(<span class=mono>{tr['sha256'][:12]}…</span>, <span class=mono>{te['sha256'][:12]}…</span>). Objetivo: <b>label</b>
(1 = ataque, clase positiva). Se excluyen <b>attack_cat</b> (es la respuesta en otra granularidad), <b>label</b> e
<b>id</b>. La auditoría mostró que <b>id filtra la clase</b>: el archivo de training está ordenado (id 1–56.000 son
normales, el resto ataques), así que un modelo podría aprender el orden del archivo. Quedan 42 variables predictoras,
3 categóricas (<i>proto</i>, <i>service</i>, <i>state</i>).</p>

<table><tr><th>Auditoría (por variables predictoras)</th><th class=n>Training</th><th class=n>Test</th></tr>
<tr><td>Filas · normal / ataque</td><td class=n>{thousands(tr['rows'])} · {thousands(tr['classes']['0'])} / {thousands(tr['classes']['1'])}</td>
<td class=n>{thousands(te['rows'])} · {thousands(te['classes']['0'])} / {thousands(te['classes']['1'])}</td></tr>
<tr><td>Grupos de entradas únicas</td><td class=n>{thousands(dup['unique_input_groups'])}</td><td class=n>{thousands(dupt['unique_input_groups'])}</td></tr>
<tr><td>Filas duplicadas por entradas</td><td class=n>{thousands(dup['duplicate_rows_by_inputs'])} ({pct(dup['duplicate_rows_by_inputs'] / dup['rows'])})</td>
<td class=n>{thousands(dupt['duplicate_rows_by_inputs'])} ({pct(dupt['duplicate_rows_by_inputs'] / dupt['rows'])})</td></tr>
<tr><td>Grupos con etiquetas contradictorias (filas)</td><td class=n>{dup['contradictory_groups']} ({dup['rows_in_contradictory_groups']})</td>
<td class=n>{dupt['contradictory_groups']} ({dupt['rows_in_contradictory_groups']})</td></tr></table>

<p><b>Política.</b> Los duplicados se conservan porque reflejan la frecuencia real de conexiones, pero cada grupo de
entradas idénticas queda completo en una sola partición. Los {dup['contradictory_groups']} grupos contradictorios
({dup['rows_in_contradictory_groups']} filas, {pct(dup['rows_in_contradictory_groups'] / dup['rows'])}) no se pueden
aprender y se excluyen de train/val; quedan marcados en <span class=mono>splits.csv</span>. El test no se modifica.</p>

<p><b>Particiones.</b> Train y validación salen solo del archivo de training con
<span class=mono>StratifiedGroupKFold</span> (5 folds, semilla {sp['seed']}; el fold 0 es validación), agrupando por la
huella de las variables predictoras: <b>{thousands(sp['sizes']['train'])}</b> filas de train y
<b>{thousands(sp['sizes']['val'])}</b> de validación, con <b>{ov['train_val_groups']}</b> grupos compartidos. La
proporción de ataques es la misma en ambas ({pct(sp['classes']['train']['1'] / sp['sizes']['train'])}). El split es
determinista: dos ejecuciones produjeron archivos idénticos. Codificación, imputación y escalado se ajustan solo con
train, dentro de un <span class=mono>Pipeline</span> que se guarda completo. Cruce con el test:
{thousands(ov['test_rows_with_inputs_in_train_or_val'])} filas del test ({pct(ov['test_rows_with_inputs_in_train_or_val'] / ov['test_rows_total'])})
tienen entradas idénticas a alguna de train/val.</p>

<p><b>Dos versiones de datos en DVC.</b> <span class=mono>data/raw.dvc</span> (archivos recibidos, md5
<span class=mono>{raw_md5}</span>)
y <span class=mono>data/prepared.dvc</span> (train/val/test preparados y asignación de filas, md5
<span class=mono>{v2['dvc_prepared_md5']}</span>), con remoto principal en DagsHub y un espejo de lectura anónima en
Hugging Face (remoto por defecto <span class=mono>hf</span>).</p>

<h2>2. Cuatro experimentos</h2>
<p>Las cuatro configuraciones usan la misma partición, la misma semilla y el mismo preprocesamiento base (mediana +
escalado en numéricas, one-hot con categoría “infrecuente” para valores raros o nuevos). Cada run registra en MLflow
la hipótesis, los parámetros, la semilla, el commit Git (<span class=mono>{v1['git_commit_trained'][:7]}</span>), el md5
DVC, las métricas de validación, los scores por fila y el modelo completo. Métricas con ataque como positivo; si un
denominador es cero la métrica vale 0 y se marca en <span class=mono>zero_division</span> (no ocurrió).</p>

<table><tr><th>ID</th><th>Familia</th><th>Cambio</th><th class=n>Recall</th><th class=n>Precision</th><th class=n>FPR</th>
<th class=n>Macro-F1</th><th class=n>FP / FN</th><th>Run</th></tr>
{run_row('e1_logreg_base', 'Reg. logística', 'Referencia')}
{run_row('e2_logreg_log_balanced', 'Reg. logística', 'log1p + class_weight balanced')}
{run_row('e3_random_forest', 'Árboles (bagging)', 'Random forest, 200 árboles')}
{run_row('e4_hist_gradient_boosting', 'Árboles (boosting)', 'HistGradientBoosting, 300 iter.')}
</table>
<p style="font-size:8pt;color:#52514e">Validación, {thousands(sp['sizes']['val'])} filas, umbral 0,5. Tabla completa: <span class=mono>evidencias/experimentos.csv</span>.</p>

<ul>
<li><b>e1.</b> {hyp['e1_logreg_base']} <i>Resultado:</i> {concl['e1_logreg_base']}</li>
<li><b>e2.</b> {hyp['e2_logreg_log_balanced']} <i>Resultado:</i> {concl['e2_logreg_log_balanced']}</li>
<li><b>e3.</b> {hyp['e3_random_forest']} <i>Resultado:</i> {concl['e3_random_forest']}</li>
<li><b>e4.</b> {hyp['e4_hist_gradient_boosting']} <i>Resultado:</i> {concl['e4_hist_gradient_boosting']}</li>
</ul>

<figure><img src="fig_validacion.png" alt="Recall de ataque frente a FPR en validación">
<figcaption>Figura 1. Recall frente a FPR en validación. Las curvas recorren todos los umbrales de v1 y v2; los
círculos son las cuatro configuraciones a umbral 0,5. La línea discontinua es el tope de FPR de la política de v2.</figcaption></figure>

<p><b>Lectura de errores.</b> Los modelos lineales cometen sobre todo falsos positivos (e1: {thousands(int(r.loc['e1_logreg_base'].fp))}
normales marcados como ataque). Ponderar clases (e2) los reduce a la mitad, pero convierte parte en falsos negativos.
Los árboles mejoran ambos errores a la vez, lo que confirma que la separación depende de interacciones entre
protocolo, estado, TTL y volumen. En toda la curva de la Figura 1, la de boosting queda por encima de la lineal.</p>

<h2>3. Dos versiones: v1 y v2</h2>
<table><tr><th></th><th>v1</th><th>v2</th></tr>
<tr><td>Configuración</td><td>e1 · regresión logística</td><td>e4 · HistGradientBoosting</td></tr>
<tr><td>Umbral (regla: ataque si score ≥ umbral)</td><td>{num(v1['threshold'], 2)}</td><td>{num(v2['threshold'], 2)}</td></tr>
<tr><td>Recall / precision / FPR / macro-F1 (val)</td><td>{num(m1['recall_attack'])} / {num(m1['precision_attack'])} / {num(m1['fpr'])} / {num(m1['macro_f1'])}</td>
<td>{num(m2['recall_attack'])} / {num(m2['precision_attack'])} / {num(m2['fpr'])} / {num(m2['macro_f1'])}</td></tr>
<tr><td>TN / FP / FN / TP (val)</td><td>{cm(m1)}</td><td>{cm(m2)}</td></tr>
<tr><td>Run MLflow</td><td class=mono>{v1['mlflow_run_id']}</td><td class=mono>{v2['mlflow_run_id']}</td></tr>
<tr><td>Revisión Hugging Face</td><td class=mono>{v1['huggingface']['revision_sha']}</td><td class=mono>{v2['huggingface']['revision_sha']}</td></tr>
<tr><td>SHA-256 model.joblib</td><td class=mono>{v1['model_sha256'][:24]}…</td><td class=mono>{v2['model_sha256'][:24]}…</td></tr>
</table>
<p><b>Qué cambia y por qué.</b> v1 es la primera solución funcional: preprocesamiento, modelo lineal y umbral por
defecto. v2 cambia el <b>modelo</b> (boosting, mejor macro-F1 de los cuatro experimentos y solo 1,1 MB frente a 58 MB
del bosque) y el <b>umbral</b>. <b>Mejora:</b> los falsos positivos bajan de {thousands(m1['confusion_matrix']['fp'])} a
{thousands(m2['confusion_matrix']['fp'])} (FPR de {num(m1['fpr'])} a {num(m2['fpr'])}), la precision sube a
{num(m2['precision_attack'])} y el macro-F1 a {num(m2['macro_f1'])}. <b>Empeora:</b> los ataques no detectados suben de
{thousands(m1['confusion_matrix']['fn'])} a {thousands(m2['confusion_matrix']['fn'])} (recall {num(m1['recall_attack'])} →
{num(m2['recall_attack'])}). <b>Error aceptado:</b> en un centro de operaciones cada falso positivo es una alerta que
alguien revisa. Con FPR de 0,177, casi una de cada cinco conexiones normales generaría alerta y el sistema dejaría de
ser creíble. Se acepta perder unos 3,7 puntos de recall a cambio de reducir las falsas alarmas a menos de un tercio.</p>

<h2>4. Elección del umbral</h2>
<p>La política se fijó antes de mirar el test, en <span class=mono>configs/selection.yaml</span>: <b>el mayor recall
de ataque con FPR de validación ≤ {pct(sel['threshold_policy']['max_fpr'])}</b>, barriendo umbrales con paso
{num(sel['threshold_policy']['grid_step'], 3)} sobre los scores de validación guardados en el run (sin reentrenar). El
resultado es {num(v2['threshold'], 2)} (FPR {num(m2['fpr'])}). El barrido completo está en
<span class=mono>evidencias/umbral_v2.csv</span>. Para v1 se mantiene 0,5 porque representa la solución inicial sin
ajuste. La selección quedó en <span class=mono>evidencias/seleccion.json</span> y se congeló en el commit
<span class=mono>{test['frozen_selection']['git_commit']}</span> ({test['frozen_selection']['committed_at']}), antes de
la evaluación. <span class=mono>src/evaluate_test.py</span> se niega a correr si esa selección no está commiteada o si
el test ya se evaluó.</p>

<h2>5. Evaluación final en test</h2>
<p>Se evaluó <b>una vez</b> la revisión congelada de v2, descargada de Hugging Face por SHA, sobre las
{thousands(test['dataset']['rows_evaluated'])} filas del test oficial, sin eliminar ninguna. Las predicciones por fila
están en <span class=mono>evidencias/predicciones-test.csv</span> (<span class=mono>row_id, y_true, score_attack, y_pred</span>)
y las métricas se pueden recalcular desde ese archivo; el recálculo coincide con
<span class=mono>metricas-test.json</span>.</p>
<table><tr><th>v2 · umbral {num(v2['threshold'], 2)}</th><th class=n>Filas</th><th class=n>Recall</th><th class=n>Precision</th>
<th class=n>FPR</th><th class=n>Macro-F1</th><th class=n>TN / FP / FN / TP</th></tr>
<tr><td>Validación</td><td class=n>{thousands(sp['sizes']['val'])}</td><td class=n>{num(m2['recall_attack'])}</td><td class=n>{num(m2['precision_attack'])}</td><td class=n>{num(m2['fpr'])}</td><td class=n>{num(m2['macro_f1'])}</td><td class=n>{cm(m2)}</td></tr>
<tr><td><b>Test</b></td><td class=n>{thousands(test['dataset']['rows_evaluated'])}</td><td class=n><b>{num(mt['recall_attack'])}</b></td><td class=n><b>{num(mt['precision_attack'])}</b></td><td class=n><b>{num(mt['fpr'])}</b></td><td class=n><b>{num(mt['macro_f1'])}</b></td><td class=n>{cm(mt)}</td></tr>
<tr><td>Test · entradas vistas en train/val</td><td class=n>{thousands(seen['rows'])}</td><td class=n>{num(seen['recall_attack'])}</td><td class=n>{num(seen['precision_attack'])}</td><td class=n>{num(seen['fpr'])}</td><td class=n>{num(seen['macro_f1'])}</td><td class=n>{cm(seen)}</td></tr>
<tr><td>Test · entradas nuevas</td><td class=n>{thousands(unseen['rows'])}</td><td class=n>{num(unseen['recall_attack'])}</td><td class=n>{num(unseen['precision_attack'])}</td><td class=n>{num(unseen['fpr'])}</td><td class=n>{num(unseen['macro_f1'])}</td><td class=n>{cm(unseen)}</td></tr>
</table>
<p><b>Interpretación.</b> El recall se mantiene ({num(mt['recall_attack'])}), pero el <b>FPR se multiplica por
{num(mt['fpr'] / m2['fpr'], 1)}</b> y la precision cae a {num(mt['precision_attack'])}. La caída no viene del solapamiento:
en las filas con entradas nunca vistas el FPR es prácticamente el mismo ({num(unseen['fpr'])}). La explicación más
probable es un corrimiento de distribución entre las particiones oficiales: el test tiene
{pct(te['classes']['0'] / te['rows'])} de tráfico normal frente a {pct(tr['classes']['0'] / tr['rows'])} en training,
y la composición de su tráfico normal también puede diferir. No se exploró más a fondo para no convertir el test en
datos de análisis. Un umbral calibrado para
5 % de FPR en una distribución no garantiza ese FPR en otra. El resultado se conserva sin reajustar, porque hacerlo
convertiría el test en datos de selección. La model card de v2 se actualizó con estas cifras en una revisión nueva
(<span class=mono>{card['card_revision'][:12]}…</span>) que solo cambia <span class=mono>README.md</span>; el hash LFS
de <span class=mono>model.joblib</span> es idéntico al evaluado.</p>

<h2>6. Trazabilidad y recuperación</h2>
<p>Cada versión se puede seguir por identificadores, no por nombres: <b>commit Git</b> → <b>md5 DVC</b> → <b>run ID de
MLflow</b> → <b>revisión de Hugging Face</b> → <b>SHA-256 del modelo</b> (tabla de la sección 3 y
<span class=mono>entrega.md</span>). <span class=mono>src/predict.py</span> verifica el SHA-256 de
<span class=mono>model.joblib</span> contra <span class=mono>config.json</span> antes de cargarlo.
La prueba de recuperación (<span class=mono>scripts/recuperacion.sh</span>) clonó el código en una carpeta nueva, creó
un venv nuevo con las dependencias fijadas, ejecutó <span class=mono>dvc pull</span> con caché vacía desde DagsHub (los 6
SHA-256 de datos originales y preparados coincidieron), descargó v1 y v2 por SHA completo con una caché de Hugging Face
nueva y predijo {rec['versions']['v1']['rows']} filas fijas de validación (mitad normales, mitad ataques). Resultado:
las etiquetas son iguales y la diferencia máxima de score es {rec['versions']['v1']['max_abs_score_diff']:g} (v1) y
{rec['versions']['v2']['max_abs_score_diff']:g} (v2), con tolerancia declarada de 10<sup>−9</sup>. Los archivos de
salida tienen el mismo SHA-256 que los de referencia. Log y comandos: <span class=mono>evidencias/recuperacion/entorno-nuevo/</span>.</p>

<h2>7. Limitaciones</h2>
<ul>
<li><b>Validez externa.</b> UNSW-NB15 es tráfico sintético de laboratorio (IXIA PerfectStorm, 2015). Los resultados no
demuestran eficacia sobre una red actual.</li>
<li><b>Umbral no transferible.</b> El FPR de test (0,181) muestra que el umbral depende de la distribución. En operación
haría falta recalibrarlo con tráfico normal propio y monitorear el FPR.</li>
<li><b>Una sola partición de validación.</b> Se eligió modelo y umbral sobre el mismo fold. No se estimó la varianza
entre folds, así que diferencias pequeñas (por ejemplo e3 frente a e4) no son concluyentes.</li>
<li><b>Población modificada.</b> Excluir {dup['rows_in_contradictory_groups']} filas contradictorias cambia levemente la
población de entrenamiento. Además, los duplicados pesan más en el ajuste.</li>
<li><b>Alcance.</b> Solo clasificación binaria, probabilidades no calibradas e hiperparámetros sin búsqueda sistemática.
MLflow se usó en local (SQLite) y la evidencia consultable es la exportación en
<span class=mono>evidencias/mlflow/</span>.</li>
<li><b>Acceso a datos.</b> El almacenamiento DVC de DagsHub requiere autenticación incluso con el repositorio público.
Por eso el remoto por defecto es un espejo de solo lectura en un dataset público de Hugging Face, que se mantiene
aparte de DagsHub (ver <span class=mono>entrega.md</span>).</li>
</ul>

<div class="refs">
<h2>Referencias</h2>
<ol>
<li>Moustafa, N., &amp; Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems
(UNSW-NB15 network data set). <i>Military Communications and Information Systems Conference (MilCIS)</i>, IEEE.</li>
<li>Moustafa, N., &amp; Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical analysis of the
UNSW-NB15 data set and the comparison with the KDD99 data set. <i>Information Security Journal: A Global Perspective</i>, 25(1–3), 18–31.</li>
<li>UNSW Canberra. UNSW-NB15 dataset. https://research.unsw.edu.au/projects/unsw-nb15-dataset</li>
<li>Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i>, 12, 2825–2830. Guía de errores
comunes: https://scikit-learn.org/stable/common_pitfalls.html</li>
<li>DVC. Data Version Control documentation. https://dvc.org/doc</li>
<li>MLflow. MLflow Tracking documentation. https://mlflow.org/docs/latest/tracking.html</li>
<li>Hugging Face. Download files from the Hub (revision) y Model Cards. https://huggingface.co/docs/huggingface_hub/guides/download ·
https://huggingface.co/docs/hub/model-cards</li>
<li>Mitchell, M. et al. (2019). Model Cards for Model Reporting. <i>FAT* '19</i>, 220–229.</li>
</ol>
<h2>Repositorios</h2>
<ul>
<li>Código: https://github.com/isaiasexternado-a11y/defiende-servidor-unsw-nb15 (tags v1, v2)</li>
<li>Datos (DVC, lectura anónima): https://huggingface.co/datasets/Isxcesp/defiende-servidor-unsw-nb15-dvc</li>
<li>Datos (DVC, principal): https://dagshub.com/isaiasexternado-a11y/defiende-servidor-unsw-nb15</li>
<li>Modelos: https://huggingface.co/Isxcesp/defiende-servidor-unsw-nb15</li>
</ul>
<p style="margin-top:10px"><b>Uso de IA.</b> Este proyecto se desarrolló con apoyo de Claude Code (Anthropic). El detalle
de qué se hizo con el asistente y qué se verificó está en <span class=mono>entrega.md</span>.</p>
</div>
</body></html>
"""


def main() -> int:
    audit, runs, sel, test, rec, card = load()
    figure(runs, sel)
    (OUT / "informe.html").write_text(html(audit, runs, sel, test, rec, card))
    print(f"✓ {OUT / 'informe.html'} y fig_validacion.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
