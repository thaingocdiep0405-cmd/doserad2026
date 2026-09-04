#!/usr/bin/env python3
"""Upload algorithm image via gcapi + submit to Preliminary Photon CT phase."""
import json, os, re, sys, time
import urllib.request, urllib.parse

TOKEN_FILE = "/tmp/gc_api_token.txt"
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
        for line in f:
            if "sessionid" in line:
                sid = line.strip().split("\t")[-1]
            if "csrftoken" in line:
                csrf = line.strip().split("\t")[-1]
    return sid, csrf


def upload_image(token):
    import gcapi

    c = gcapi.Client(token=token)
    file_size = os.path.getsize(IMAGE_PATH)
    filename = os.path.basename(IMAGE_PATH)
    print(f"  File: {filename} ({file_size / 1e9:.2f} GB)")

    print("  Uploading via gcapi (chunked)...")
    with open(IMAGE_PATH, "rb") as f:
        upload = c.uploads.upload_fileobj(fileobj=f, filename=filename)
    print(f"  Upload complete: {upload.api_url}")

    print("  Creating algorithm image...")
    image = c.algorithm_images.create(
        algorithm=f"{GC}/api/v1/algorithms/{ALGORITHM_PK}/",
        user_upload=upload.api_url,
    )
    print(f"  Image PK: {image.pk}, status: {image.import_status}")
    return image.pk


def wait_import(image_pk, token, timeout_sec=900):
    import gcapi

    c = gcapi.Client(token=token)
    start = time.time()
    while time.time() - start < timeout_sec:
        image = c.algorithm_images.detail(image_pk)
        status = image.import_status
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
    cookie = f"sessionid={sid}; _csrftoken={csrf}"

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
    req = urllib.request.Request(
        f"{GC}/algorithms/{ALGORITHM_SLUG}/images/activate/",
        data=data, method="POST",
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Cookie", cookie)
    req.add_header("Referer", f"{GC}/algorithms/{ALGORITHM_SLUG}/images/{image_pk}/")
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"  Activated: HTTP {resp.status}")


def submit_preliminary(image_pk, sid, csrf):
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
        "algorithm_image": image_pk,
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
    token = open(TOKEN_FILE).read().strip()
    print(f"Token: {token[:8]}...")

    sid, csrf = load_cookies()
    print(f"Session: {sid[:10]}...")

    print(f"\n[1/4] Uploading image ({os.path.getsize(IMAGE_PATH) / 1e9:.1f} GB)...")
    image_pk = upload_image(token)

    print(f"\n[2/4] Waiting for import...")
    if not wait_import(image_pk, token):
        sys.exit(1)

    print(f"\n[3/4] Activating image...")
    activate_image(image_pk, sid, csrf)

    print(f"\n[4/4] Submitting to preliminary...")
    submit_preliminary(image_pk, sid, csrf)

    print("\nDone!")
