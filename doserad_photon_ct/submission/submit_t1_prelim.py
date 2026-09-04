#!/usr/bin/env python3
"""
Submit Task 1 (Photon CT) to Preliminary test phase on Grand Challenge.
Uses gcapi for upload, urllib for form-based activate/submit.

Usage: python3 submit_t1_prelim.py
"""
import json, os, re, sys, time
import urllib.request, urllib.parse

# Config
COOKIE_FILE = "/tmp/gc_cookies_new.txt"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "dist", "doserad2026-photon-ct-v6-resub.tar.gz")
ALGORITHM_PK = "322e2c78-1c95-421a-a6f2-f0176d9f6bb6"
ALGORITHM_SLUG = "photon-ct-dose-engine-v1"
MODEL_PK = "21dfed4a-cc93-43e1-810b-f7f9441f9f67"
PHASE_PK = "f2ccbdbf-13e2-4479-b6c0-23fa45bf9a75"
CREATOR_ID = "150744"
GC = "https://grand-challenge.org"
CH = "https://doserad2026.grand-challenge.org"

def load_cookies():
    sid = csrf = None
    with open(COOKIE_FILE) as f:
        for l in f:
            if "sessionid" in l: sid = l.strip().split("\t")[-1]
            if "csrftoken" in l: csrf = l.strip().split("\t")[-1]
    if not sid or not csrf:
        print("ERROR: Missing cookies"); sys.exit(1)
    return sid, csrf

def cookie_str(sid, csrf):
    return f"sessionid={sid}; _csrftoken={csrf}"

def get_api_token(sid, csrf):
    """Get API token from GC settings page."""
    token_cache = "/tmp/gc_api_token.txt"
    if os.path.exists(token_cache):
        t = open(token_cache).read().strip()
        if len(t) >= 40:
            print(f"  Cached token: {t[:8]}...")
            return t

    cookie = cookie_str(sid, csrf)
    req = urllib.request.Request(f"{GC}/settings/api-tokens/")
    req.add_header("Cookie", cookie)
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()

    # Look for existing token in <code> tags
    codes = re.findall(r'<code[^>]*>(.*?)</code>', html)
    for c in codes:
        c = c.strip()
        if len(c) == 40 and all(ch in '0123456789abcdef' for ch in c):
            print(f"  Found API token: {c[:8]}...")
            open(token_cache, "w").write(c)
            return c

    # Look for 40-char hex anywhere
    tokens = re.findall(r'\b[a-f0-9]{40}\b', html)
    if tokens:
        print(f"  Found token: {tokens[0][:8]}...")
        open(token_cache, "w").write(tokens[0])
        return tokens[0]

    print("ERROR: No API token found. Go to https://grand-challenge.org/settings/api-tokens/")
    print("       and copy the token, then save it to /tmp/gc_api_token.txt")
    sys.exit(1)

def upload_image_gcapi(token):
    """Upload algorithm container image using gcapi's upload mechanism."""
    import gcapi

    c = gcapi.Client(token=token)
    file_size = os.path.getsize(IMAGE_PATH)
    filename = os.path.basename(IMAGE_PATH)
    print(f"  File: {filename} ({file_size/1e9:.2f} GB)")

    # Step 1: Upload file using gcapi's UserUpload (presigned URL chunked upload)
    print("  Creating upload session...")
    with open(IMAGE_PATH, "rb") as f:
        upload_result = c.uploads.upload_fileobj(fileobj=f, filename=filename)

    print(f"  Upload complete: {upload_result.api_url}")

    # Step 2: Create algorithm image referencing the upload
    print("  Creating algorithm image...")
    resp = c.post(
        f"{GC}/api/v1/algorithms/images/",
        json={
            "algorithm": f"{GC}/api/v1/algorithms/{ALGORITHM_PK}/",
            "user_upload": upload_result.api_url,
            "comment": "v6-resubmit",
        },
    )

    if resp.status_code >= 300:
        print(f"  ERROR: Create image failed: HTTP {resp.status_code}")
        print(f"  {resp.text[:500]}")
        sys.exit(1)

    result = resp.json()
    print(f"  Algorithm image created: PK={result.get('pk')}")
    return result

def wait_import(image_pk, token, timeout_sec=900):
    """Wait for algorithm image import to complete."""
    import gcapi
    c = gcapi.Client(token=token)

    start = time.time()
    while time.time() - start < timeout_sec:
        resp = c.get(f"{GC}/api/v1/algorithms/images/{image_pk}/")
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("import_status", "Unknown")
            print(f"  Import: {status}", flush=True)
            if status == "Completed":
                return True
            if status == "Failed":
                print("  IMPORT FAILED!")
                return False
        time.sleep(30)
    print("  Import timeout!")
    return False

def activate_image(image_pk, sid, csrf):
    """Activate the algorithm image via web form."""
    cookie = cookie_str(sid, csrf)

    req = urllib.request.Request(f"{GC}/algorithms/{ALGORITHM_SLUG}/images/{image_pk}/")
    req.add_header("Cookie", cookie)
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()
    csrf_f = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
    fc = csrf_f.group(1) if csrf_f else csrf

    data = urllib.parse.urlencode({
        "csrfmiddlewaretoken": fc,
        "algorithm_image": image_pk,
        "save": "Activate algorithm image",
    }).encode()
    req = urllib.request.Request(f"{GC}/algorithms/{ALGORITHM_SLUG}/images/activate/",
                                data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Cookie", cookie)
    req.add_header("Referer", f"{GC}/algorithms/{ALGORITHM_SLUG}/images/{image_pk}/")
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"  Activated: HTTP {resp.status}")

def submit_preliminary(image_pk, sid, csrf):
    """Submit to preliminary test phase."""
    cookie = cookie_str(sid, csrf)

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
        "algorithm_image": image_pk,
        "algorithm_model": MODEL_PK,
    }).encode()
    req = urllib.request.Request(
        f"{CH}/evaluation/photon-dose-preliminary-testing/submissions/create/",
        data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Cookie", cookie)
    req.add_header("Referer", f"{CH}/evaluation/photon-dose-preliminary-testing/submissions/create/")
    resp = urllib.request.urlopen(req, timeout=30)

    if resp.url != f"{CH}/evaluation/photon-dose-preliminary-testing/submissions/create/":
        print(f"  SUCCESS! Redirected to: {resp.url}")
        return True

    body = resp.read().decode()
    if "already exists" in body:
        print("  DUPLICATE - this combination already submitted")
    else:
        errors = re.findall(r'(?:errorlist|alert-danger)[^>]*>(.*?)<', body, re.IGNORECASE)
        clean_errors = [re.sub(r'<[^>]+>', '', e).strip() for e in errors if e.strip()]
        if clean_errors:
            print(f"  Errors: {clean_errors}")
        else:
            print(f"  Result unclear. Check GC manually.")
    return False


if __name__ == "__main__":
    sid, csrf = load_cookies()
    print(f"Session: {sid[:10]}...")

    print("\n[1/5] Getting API token...")
    token = get_api_token(sid, csrf)

    print(f"\n[2/5] Uploading image ({os.path.getsize(IMAGE_PATH)/1e9:.1f} GB)...")
    result = upload_image_gcapi(token)
    image_pk = result.get("pk")
    print(f"  Image PK: {image_pk}")

    print("\n[3/5] Waiting for import...")
    if not wait_import(image_pk, token):
        sys.exit(1)

    print("\n[4/5] Activating image...")
    activate_image(image_pk, sid, csrf)

    print("\n[5/5] Submitting to preliminary...")
    submit_preliminary(image_pk, sid, csrf)

    print("\nDone!")
