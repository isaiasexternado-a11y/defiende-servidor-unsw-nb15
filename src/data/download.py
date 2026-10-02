"""Descarga UNSW-NB15 desde la fuente oficial de UNSW (SharePoint público).

Fuente: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Uso académico; ver docs/DATASET.md para atribución y citas obligatorias.

    python -m src.data.download            # train/test oficiales + metadatos
    python -m src.data.download --full     # además, los 4 CSV completos (~590 MB)
"""

import argparse
import hashlib
import http.cookiejar
import json
import sys
import urllib.parse
import urllib.request
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
    ("CSV Files/UNSW-NB15_1.csv", None),
    ("CSV Files/UNSW-NB15_2.csv", None),
    ("CSV Files/UNSW-NB15_3.csv", None),
    ("CSV Files/UNSW-NB15_4.csv", None),
    ("CSV Files/NUSW-NB15_GT.csv", None),
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
    manifest = {}
    for remote, expected in FILES + (FULL_FILES if args.full else []):
        dest = args.out / Path(remote).name
        if dest.exists():
            print(f"= {dest} (ya existe)")
        else:
            print(f"↓ {remote}")
            download(opener, remote, dest)
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
