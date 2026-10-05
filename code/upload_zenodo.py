import hashlib, json, os, sys, time
import urllib.request, urllib.error

API = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
PUBLISH = "--publish" in sys.argv


def call(method, url, data=None, ctype="application/json"):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    if data is not None:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    return json.loads(body) if body else {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if not TOKEN:
        sys.exit("set ZENODO_TOKEN first")
    manifest = json.load(open(os.path.join(DIST, "MANIFEST.json")))
    meta = json.load(open(os.path.join(DIST, "zenodo_metadata.json")))

    for e in manifest:
        p = os.path.join(DIST, e["file"])
        if sha256(p) != e["sha256"] or os.path.getsize(p) != e["bytes"]:
            sys.exit(f"{e['file']} does not match MANIFEST.json, nothing was uploaded")
    print(f"{len(manifest)} files match the manifest")

    dep = call("POST", f"{API}/deposit/depositions", b"{}")
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print(f"draft {dep_id} created, reserved at {dep['links'].get('html')}")

    call("PUT", f"{API}/deposit/depositions/{dep_id}",
         json.dumps(meta).encode())

    done = 0
    for e in manifest:
        p = os.path.join(DIST, e["file"])
        with open(p, "rb") as fh:
            req = urllib.request.Request(f"{bucket}/{e['file']}", data=fh, method="PUT")
            req.add_header("Authorization", f"Bearer {TOKEN}")
            req.add_header("Content-Type", "application/octet-stream")
            req.add_header("Content-Length", str(os.path.getsize(p)))
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=3600) as r:
                        got = json.loads(r.read())
                    break
                except urllib.error.URLError as err:
                    if attempt == 2:
                        raise
                    print(f"  retry {e['file']} after {err}")
                    time.sleep(10)
                    fh.seek(0)
        if got.get("checksum", "").replace("md5:", "") and got.get("size") != e["bytes"]:
            sys.exit(f"{e['file']} uploaded at the wrong size")
        done += 1
        print(f"  [{done}/{len(manifest)}] {e['file']} {e['bytes']:,} bytes")

    files = call("GET", f"{API}/deposit/depositions/{dep_id}/files")
    print(f"{len(files)} files on the draft")

    if PUBLISH:
        out = call("POST", f"{API}/deposit/depositions/{dep_id}/actions/publish")
        print("published", out["doi"], out["links"]["record_html"])
    else:
        print(f"draft kept unpublished at {dep['links'].get('html')}")
        print("rerun with --publish to mint the DOI")


if __name__ == "__main__":
    main()
