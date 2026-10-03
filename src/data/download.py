"""Descarga UNSW-NB15 desde la fuente oficial de UNSW (SharePoint público).

Fuente: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Uso académico; ver docs/DATASET.md para atribución y citas obligatorias.

    python -m src.data.download            # train/test oficiales + metadatos
    python -m src.data.download --full     # además, los 4 CSV completos (~670 MB)
"""

import argparse
import hashlib
import http.cookiejar
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SHARE_URL = (
    "https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
    "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS"
)
SITE = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au"
ROOT = "/personal/z5025758_ad_unsw_edu_au/Documents/UNSW-NB15 dataset"

# (ruta remota relativa a ROOT, sha256 esperado o None)
FILES = [
    ("CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
     "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa"),
    ("CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
     "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"),
    ("CSV Files/NUSW-NB15_features.csv",
     "c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce"),
    ("CSV Files/UNSW-NB15_LIST_EVENTS.csv",
     "5b40f8128e2c87e691157c76debde04c328b0eb66bab97f1e7e17625f84497f2"),
    ("CSV Files/The UNSW-NB15 description.pdf",
     "20784d87509db0185b5d816f01a4ea04e60be180798e2d38cabb500f6dd13999"),
    ("ReadMe.pdf",
     "d32464d62346c12a7caa7b140d3c26cb9fd3f97898c3bddb49a8db5375ee6608"),
]
FULL_FILES = [
    ("CSV Files/UNSW-NB15_1.csv",
     "7d851bbeabd27894ce39c8e78835c73341fc946652fb7743b9eff193b55eb511"),
    ("CSV Files/UNSW-NB15_2.csv",
     "6130ad02873cc6069ae695cf2844f2e8c2e9a9a1b7532dd82ab8f202757cacf8"),
    ("CSV Files/UNSW-NB15_3.csv",
     "ae990a96c3dfcd425ce2801aadb1727a34d5e0ae6d8215dbcdae60dedfaef640"),
    ("CSV Files/UNSW-NB15_4.csv",
     "cdf563692d51d405541dd659ddcdad9fa01f001f05fe9fc4b67f00ca12fbc96a"),
    ("CSV Files/NUSW-NB15_GT.csv",
     "6d27542cb6457db599e0a78274ac141fec0296531f00ac5569551a276c5ab1a8"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_opener() -> urllib.request.OpenerDirector:
    # Abrir el enlace compartido otorga una cookie de invitado para la API REST.
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    opener.open(SHARE_URL).read()
    return opener


def download(opener, remote: str, dest: Path) -> None:
    server_path = urllib.parse.quote(f"{ROOT}/{remote}")
    url = f"{SITE}/_api/web/GetFileByServerRelativeUrl('{server_path}')/$value"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with opener.open(url) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.rename(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--full", action="store_true", help="incluye los 4 CSV completos")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    opener = build_opener()
    files = FILES + (FULL_FILES if args.full else [])

    # SharePoint limita la velocidad por conexión: se descargan los archivos en paralelo.
    pending = [r for r, _ in files if not (args.out / Path(r).name).exists()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(download, opener, r, args.out / Path(r).name): r for r in pending}
        print(f"↓ {len(pending)} archivos en paralelo")
        for future in as_completed(futures):
            future.result()
            print(f"✓ {futures[future]}", flush=True)

    manifest = {}
    for remote, expected in files:
        dest = args.out / Path(remote).name
        digest = sha256(dest)
        if expected and digest != expected:
            print(f"✗ checksum distinto en {dest}: {digest}", file=sys.stderr)
            return 1
        manifest[dest.name] = {"source": f"{ROOT}/{remote}", "bytes": dest.stat().st_size, "sha256": digest}

    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"✓ {len(manifest)} archivos en {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
