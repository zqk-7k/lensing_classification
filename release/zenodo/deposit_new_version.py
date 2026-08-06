#!/usr/bin/env python3
"""Create a new Zenodo version of the existing record and upload the built bundle.

It stops before publishing. Publishing is irreversible -- a published Zenodo version
cannot be deleted or have its files changed -- so the last step is left to a human who
can look at the draft first. The script prints the draft URL to open.

    export ZENODO_TOKEN=...            # deposit:actions + deposit:write
    python release/zenodo/deposit_new_version.py --dry-run
    python release/zenodo/deposit_new_version.py

Run it on the machine that holds `release/zenodo/dist/`, so the 2.65 GB archive is
uploaded from where it already is rather than round-tripping through a browser.

Getting a token: https://zenodo.org/account/settings/applications/tokens/new/
Tick "deposit:write" and "deposit:actions". The token is not stored by this script.

What it does:
  1. resolves the latest version of the concept record;
  2. opens a new version draft;
  3. removes the files inherited from the previous version, since this is a full
     replacement rather than an addition;
  4. uploads the four files in dist/;
  5. applies metadata.example.json, with the version and the publication date;
  6. verifies the uploaded archive checksum against the local one;
  7. prints the draft URL and stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("needs requests:  pip install requests")

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "release" / "zenodo" / "dist"
METADATA = ROOT / "release" / "zenodo" / "metadata.example.json"
CONCEPT_RECID = "21311077"          # concept record; always resolves to the latest version
BASE = "https://zenodo.org/api"


def md5(path: Path) -> str:
    value = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve_latest(concept: str, auth: dict):
    """Find the newest published deposition of a concept record.

    /api/records/<conceptrecid> answers 410 rather than redirecting, so resolve through
    the account's own depositions, which carry conceptrecid, and fall back to the record
    endpoint only if that finds nothing.
    """
    listed = requests.get(f"{BASE}/deposit/depositions", **auth,
                          params={"size": 100}, timeout=60)
    if listed.ok:
        mine = [d for d in listed.json()
                if str(d.get("conceptrecid")) == str(concept) and d.get("submitted")]
        if mine:
            return max(int(d["id"]) for d in mine), True
    direct = requests.get(f"{BASE}/records/{concept}", timeout=60)
    if direct.ok:
        return int(direct.json()["id"]), False
    return None, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="check the token, the files and the metadata, then stop")
    parser.add_argument("--concept", default=CONCEPT_RECID)
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("set ZENODO_TOKEN first")
    # header auth, not the access_token query parameter: a failing request would
    # otherwise put the token into the exception message and into any log capturing it
    auth = {"headers": {"Authorization": f"Bearer {token}"}}

    files = sorted(p for p in DIST.iterdir() if p.is_file())
    if not files:
        sys.exit(f"nothing to upload in {DIST}")
    archive = next((p for p in files if p.suffix == ".gz"), None)
    if archive is None:
        sys.exit("no .tar.gz in dist/")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    print(f"concept record : {args.concept}")
    print(f"metadata       : version {metadata.get('version')}, "
          f"{len(metadata['creators'])} creators, license {metadata.get('license')}")
    print("files to upload:")
    for path in files:
        print(f"  {path.name:52} {path.stat().st_size:>13,} bytes")
    print(f"archive md5    : {md5(archive)}")

    if args.dry_run:
        response = requests.get(f"{BASE}/deposit/depositions", **auth,
                                params={"size": 100}, timeout=60)
        print(f"token check    : HTTP {response.status_code}"
              f"{' (valid)' if response.ok else ' -- token rejected'}")
        if not response.ok:
            return 1
        print(f"depositions on this account: {len(response.json())}")
        latest_id, owned = resolve_latest(args.concept, auth)
        print(f"latest version of the concept record: {latest_id}")
        if latest_id and owned:
            print("ownership      : this account owns the record -- ready to deposit")
        elif latest_id:
            print("ownership      : this account does NOT own the record.")
            print("                 A token from the owning account is required.")
            return 1
        else:
            print("ownership      : could not resolve the concept record")
            return 1
        print("\ndry run only, nothing was created")
        return 0

    latest_id, owned = resolve_latest(args.concept, auth)
    if not latest_id:
        sys.exit("could not resolve the concept record")
    if not owned:
        sys.exit("this account does not own the record; use the owning account's token")
    print(f"latest version : {latest_id}")

    new = requests.post(f"{BASE}/deposit/depositions/{latest_id}/actions/newversion",
                        **auth, timeout=120)
    new.raise_for_status()
    draft_url = new.json()["links"]["latest_draft"]
    draft = requests.get(draft_url, **auth, timeout=60).json()
    draft_id, bucket = draft["id"], draft["links"]["bucket"]
    print(f"draft          : {draft_id}")

    for existing in draft.get("files", []):
        requests.delete(f"{BASE}/deposit/depositions/{draft_id}/files/{existing['id']}",
                        **auth, timeout=120).raise_for_status()
        print(f"  removed inherited {existing['filename']}")

    for path in files:
        with path.open("rb") as handle:
            up = requests.put(f"{bucket}/{path.name}", data=handle, **auth, timeout=None)
        up.raise_for_status()
        remote = up.json().get("checksum", "").removeprefix("md5:")
        local = md5(path)
        state = "ok" if remote == local else f"MISMATCH remote={remote} local={local}"
        print(f"  uploaded {path.name:52} {state}")
        if remote != local:
            return 1

    payload = {k: v for k, v in metadata.items() if k != "$schema"}
    payload.setdefault("publication_date", __import__("datetime").date.today().isoformat())
    put = requests.put(f"{BASE}/deposit/depositions/{draft_id}",
                       **auth, json={"metadata": payload}, timeout=120)
    put.raise_for_status()
    print("metadata applied")

    print(f"\nDraft ready, NOT published:  https://zenodo.org/uploads/{draft_id}")
    print("Open it, check the file list and the metadata, then click Publish.")
    print("Publishing is irreversible: files cannot be changed afterwards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
