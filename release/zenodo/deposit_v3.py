#!/usr/bin/env python3
"""Upload a directory of files into a Zenodo draft and apply metadata. Never publishes.

    # new version of an existing concept record (the software release)
    python deposit_v3.py --new-version-of 21311077 --dir $OUT/software --metadata metadata_v3_software.json
    # a new, separate record (the strain catalogues)
    python deposit_v3.py --new-record --dir $OUT/dataset --metadata metadata_v3_dataset.json
    # resume an interrupted upload into an existing draft
    python deposit_v3.py --draft <id> --dir ... --metadata ...

Token: ZENODO_TOKEN, or the file ~/.zenodo_token (sent as a header, never printed).
Files already in the draft with the same md5 are skipped; inherited files whose names are not in
--dir are removed unless --keep lists them. Prints the draft URL and the reserved DOI, then stops:
publishing is irreversible and is left to a person who has looked at the draft.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
import urllib3.util.connection as _u3c

# The GPU server reaches zenodo.org over IPv6 at ~0.2 MB/s and over IPv4 at ~2.5 MB/s (measured
# 2026-11-08 with a 20 MB PUT); a 3 GB upload over IPv6 stalled for two hours. Force IPv4.
_u3c.HAS_IPV6 = False

BASE = "https://zenodo.org/api"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--new-version-of", help="concept record id")
    g.add_argument("--new-record", action="store_true")
    g.add_argument("--draft", help="existing unpublished deposition id")
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--keep", nargs="*", default=[], help="inherited file names to keep")
    ap.add_argument("--attempts", type=int, default=8)
    a = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN") or (Path.home() / ".zenodo_token").read_text().strip()
    auth = {"headers": {"Authorization": f"Bearer {token}"}}
    meta = json.loads(a.metadata.read_text())
    files = sorted(p for p in a.dir.iterdir() if p.is_file())

    if a.draft:
        d = requests.get(f"{BASE}/deposit/depositions/{a.draft}", **auth, timeout=60)
        d.raise_for_status(); d = d.json()
        if d.get("submitted"):
            sys.exit(f"{a.draft} is published; refusing")
    elif a.new_record:
        # idempotent: reuse an unpublished draft that already carries this title
        mine = requests.get(f"{BASE}/deposit/depositions", **auth, params={"size": 100}, timeout=60).json()
        same = [x for x in mine if not x.get("submitted") and x.get("metadata", {}).get("title") == meta["title"]]
        if same:
            d = requests.get(f"{BASE}/deposit/depositions/{same[0]['id']}", **auth, timeout=60).json()
            print(f"reusing unpublished draft with the same title", flush=True)
        else:
            r = requests.post(f"{BASE}/deposit/depositions", **auth, json={}, timeout=120)
            r.raise_for_status(); d = r.json()
    else:
        mine = requests.get(f"{BASE}/deposit/depositions", **auth, params={"size": 100}, timeout=60).json()
        pub = [x for x in mine if str(x.get("conceptrecid")) == a.new_version_of and x.get("submitted")]
        if not pub:
            sys.exit("concept record not owned by this token's account")
        latest = max(int(x["id"]) for x in pub)
        r = requests.post(f"{BASE}/deposit/depositions/{latest}/actions/newversion", **auth, timeout=120)
        r.raise_for_status()
        d = requests.get(r.json()["links"]["latest_draft"], **auth, timeout=60).json()
    did, bucket = d["id"], d["links"]["bucket"]
    print(f"draft {did}", flush=True)

    # metadata first, so the DOI is reserved and visible even if an upload is interrupted
    payload = dict(meta)
    payload.setdefault("publication_date", datetime.date.today().isoformat())
    payload["prereserve_doi"] = True
    r = requests.put(f"{BASE}/deposit/depositions/{did}", **auth, json={"metadata": payload}, timeout=120)
    if not r.ok:
        print(r.status_code, r.text[:2000]); r.raise_for_status()
    doi = (r.json().get("metadata", {}).get("prereserve_doi") or {}).get("doi")
    print(f"reserved DOI {doi}", flush=True)

    wanted = {p.name for p in files} | set(a.keep)
    cur = requests.get(f"{BASE}/deposit/depositions/{did}", **auth, timeout=60).json().get("files", [])
    for f in cur:
        if f["filename"] not in wanted:
            requests.delete(f"{BASE}/deposit/depositions/{did}/files/{f['id']}", **auth, timeout=120).raise_for_status()
            print(f"removed inherited {f['filename']}", flush=True)
    present = {f["filename"]: f.get("checksum", "").removeprefix("md5:") for f in cur if f["filename"] in wanted}

    failed = []
    for p in files:
        local = md5(p)
        if present.get(p.name) == local:
            print(f"present  {p.name} (skipped)", flush=True); continue
        for k in range(1, a.attempts + 1):
            t0 = time.time()
            try:
                with p.open("rb") as fh:
                    up = requests.put(f"{bucket}/{p.name}", data=fh, **auth, timeout=None)
                if up.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {up.status_code}")
                up.raise_for_status()
                remote = up.json().get("checksum", "").removeprefix("md5:")
                if remote != local:
                    raise requests.HTTPError(f"checksum mismatch {remote} != {local}")
                mb = p.stat().st_size / 1e6
                print(f"uploaded {p.name} {mb:.0f} MB in {time.time() - t0:.0f} s"
                      f" ({mb / max(time.time() - t0, 1):.1f} MB/s){'' if k == 1 else f' attempt {k}'}", flush=True)
                break
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                if k == a.attempts:
                    print(f"FAILED   {p.name}: {e}", flush=True); failed.append(p.name); break
                w = min(120, 5 * 2 ** (k - 1))
                print(f"retry    {p.name}: {e}; waiting {w} s", flush=True); time.sleep(w)

    print(f"\nDraft (NOT published): https://zenodo.org/uploads/{did}\nReserved DOI: {doi}")
    if failed:
        print(f"incomplete: {failed}; resume with --draft {did}")
        return 1
    print("all files present; check the draft in the browser, then publish by hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
