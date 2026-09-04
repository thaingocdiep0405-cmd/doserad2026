#!/usr/bin/env python3
"""Wait for image import, activate, then submit to Preliminary."""
import json, os, re, sys, time
import urllib.request, urllib.parse

TOKEN_FILE = "/tmp/gc_api_token.txt"
COOKIE_FILE = "/tmp/gc_cookies_new.txt"
IMAGE_PK = "c554ba0d-5c64-4f1c-bda0-4dd5ac2e778f"
ALGORITHM_SLUG = "photon-ct-dose-engine-v1"
ALGORITHM_PK = "322e2c78-1c95-421a-a6f2-f0176d9f6bb6"
MODEL_PK = "21dfed4a-cc93-43e1-810b-f7f9441f9f67"
PHASE_PK = "f2ccbdbf-13e2-4479-b6c0-23fa45bf9a75"
CREATOR_ID = "150744"
GC = "https://grand-challenge.org"
CH = "https://doserad2026.grand-challenge.org"


def load_cookies():
    sid = csrf = None
    with open(COOKIE_FILE) as f:
        for line in f:
            if "sessionid" in line:
                sid = line.strip().split("\t")[-1]
            if "csrftoken" in line:
                csrf = line.strip().split("\t")[-1]
    return sid, csrf


def wait_import(timeout_sec=1200):
    token = open(TOKEN_FILE).read().strip()
    start = time.time()
    while time.time() - start < timeout_sec:
        req = urllib.request.Request(f"{GC}/api/v1/algorithms/images/{IMAGE_PK}/")
        req.add_header("Authorization", f"Bearer {token}")
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read().decode())
            status = data.get("import_status", "Unknown")
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] Import: {status}", flush=True)
            if status == "Completed":
                return True
            if status == "Failed":
                print("  IMPORT FAILED!")
                return False
        except Exception as e:
            print(f"  Check error: {e}")
        time.sleep(30)
    print("  Import timeout!")
    return False


def activate_image(sid, csrf):
    cookie = f"sessionid={sid}; _csrftoken={csrf}"
    req = urllib.request.Request(f"{GC}/algorithms/{ALGORITHM_SLUG}/images/{IMAGE_PK}/")
    req.add_header("Cookie", cookie)
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()
    csrf_f = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
    fc = csrf_f.group(1) if csrf_f else csrf

    data = urllib.parse.urlencode({
        "csrfmiddlewaretoken": fc,
        "algorithm_image": IMAGE_PK,
        "save": "Activate algorithm image",
    }).encode()
    req = urllib.request.Request(
        f"{GC}/algorithms/{ALGORITHM_SLUG}/images/activate/",
        data=data, method="POST",
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Cookie", cookie)
    req.add_header("Referer", f"{GC}/algorithms/{ALGORITHM_SLUG}/images/{IMAGE_PK}/")
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"  Activated: HTTP {resp.status}")


def submit_preliminary(sid, csrf):
    cookie = f"sessionid={sid}; _csrftoken={csrf}"
    req = urllib.request.Request(f"{CH}/evaluation/photon-dose-preliminary-testing/submissions/create/")
    req.add_header("Cookie", cookie)
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()
    csrf_f = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
    fc = csrf_f.group(1) if csrf_f else csrf

    data = urllib.parse.urlencode({
        "csrfmiddlewaretoken": fc,
        "creator": CREATOR_ID,
        "phase": PHASE_PK,
        "algorithm": ALGORITHM_PK,
        "algorithm_image": IMAGE_PK,
        "algorithm_model": MODEL_PK,
    }).encode()
    req = urllib.request.Request(
        f"{CH}/evaluation/photon-dose-preliminary-testing/submissions/create/",
        data=data, method="POST",
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Cookie", cookie)
    req.add_header("Referer", f"{CH}/evaluation/photon-dose-preliminary-testing/submissions/create/")
    resp = urllib.request.urlopen(req, timeout=30)

    if resp.url != f"{CH}/evaluation/photon-dose-preliminary-testing/submissions/create/":
        print(f"  SUCCESS! Redirected to: {resp.url}")
        return True

    body = resp.read().decode()
    if "already exists" in body:
        print("  DUPLICATE - already submitted")
    else:
        errors = re.findall(r'(?:errorlist|alert-danger)[^>]*>(.*?)<', body, re.IGNORECASE)
        clean = [re.sub(r'<[^>]+>', '', e).strip() for e in errors if e.strip()]
        if clean:
            print(f"  Errors: {clean}")
        else:
            print("  Result unclear. Check GC manually.")
    return False


if __name__ == "__main__":
    sid, csrf = load_cookies()
    print(f"Image PK: {IMAGE_PK}")
    print(f"Session: {sid[:10]}...")

    print("\n[1/3] Waiting for import...")
    if not wait_import():
        sys.exit(1)

    print("\n[2/3] Activating image...")
    activate_image(sid, csrf)

    print("\n[3/3] Submitting to preliminary...")
    submit_preliminary(sid, csrf)

    print("\nDone!")
