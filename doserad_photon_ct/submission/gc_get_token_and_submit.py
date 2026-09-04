#!/usr/bin/env python3
"""
Get GC API token, upload image, and submit to preliminary phase.
Run: python3 gc_get_token_and_submit.py
"""
import json, math, os, re, sys, time
import urllib.request, urllib.parse, urllib.error

COOKIE_FILE = "/tmp/gc_cookies_new.txt"
TOKEN_FILE = "/tmp/gc_api_token.txt"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "dist", "doserad2026-photon-ct-v6-resub.tar.gz")
ALGORITHM_PK = "322e2c78-1c95-421a-a6f2-f0176d9f6bb6"
ALGORITHM_SLUG = "photon-ct-dose-engine-v1"
MODEL_PK = "21dfed4a-cc93-43e1-810b-f7f9441f9f67"
PHASE_PK = "f2ccbdbf-13e2-4479-b6c0-23fa45bf9a75"
CREATOR_ID = "150744"
GC = "https://grand-challenge.org"
CH = "https://doserad2026.grand-challenge.org"

def cookies():
    sid = csrf = None
    with open(COOKIE_FILE) as f:
        for l in f:
            if "sessionid" in l: sid = l.strip().split("\t")[-1]
            if "csrftoken" in l: csrf = l.strip().split("\t")[-1]
    return sid, csrf

def get_or_create_token(sid, csrf):
    if os.path.exists(TOKEN_FILE):
        t = open(TOKEN_FILE).read().strip()
        if len(t) > 10:
            print(f"Using cached token: {t[:8]}...")
            return t

    cookie = f"sessionid={sid}; _csrftoken={csrf}"

    # Get the token page
    req = urllib.request.Request(f"{GC}/settings/change-api-token/")
    req.add_header("Cookie", cookie)
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()

    # Check for existing token display
    tok = re.search(r'<code[^>]*>([a-f0-9]{40})</code>', html)
    if tok:
        token = tok.group(1)
        print(f"Found existing token: {token[:8]}...")
        open(TOKEN_FILE, "w").write(token)
        return token

    # Generate new token
    csrf_form = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
    if not csrf_form:
        print("Cannot find CSRF on token page")
        sys.exit(1)

    data = urllib.parse.urlencode({
        "csrfmiddlewaretoken": csrf_form.group(1),
    }).encode()
    req = urllib.request.Request(f"{GC}/settings/change-api-token/", data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Cookie", cookie)
    req.add_header("Referer", f"{GC}/settings/change-api-token/")
    resp = urllib.request.urlopen(req, timeout=30)
    html2 = resp.read().decode()

    tok = re.search(r'<code[^>]*>([a-f0-9]{40})</code>', html2)
    if tok:
        token = tok.group(1)
        print(f"Generated new token: {token[:8]}...")
        open(TOKEN_FILE, "w").write(token)
        return token

    # Try other patterns
    tok = re.search(r'([a-f0-9]{40})', html2)
    if tok:
        token = tok.group(1)
        print(f"Found token (pattern): {token[:8]}...")
        open(TOKEN_FILE, "w").write(token)
        return token

    print("Failed to get token. Page excerpt:")
    clean = re.sub(r'<[^>]+>', ' ', html2)
    print(re.sub(r'\s+', ' ', clean)[:500])
    sys.exit(1)

def gcapi_upload_and_submit(token):
    import gcapi
    c = gcapi.Client(token=token)

    file_size = os.path.getsize(IMAGE_PATH)
    print(f"\nUploading {IMAGE_PATH} ({file_size/1e9:.2f} GB)...")

    # Upload algorithm image
    # gcapi handles chunked upload internally
    from pathlib import Path

    # Create algorithm image via gcapi
    print("Creating algorithm image upload...")
    algo_image = c.run_external_job(
        algorithm=ALGORITHM_PK,
        inputs=None,
    )
    print(f"Result: {algo_image}")

def upload_via_gcapi(token):
    """Use gcapi to upload container image and submit."""
    import gcapi
    c = gcapi.Client(token=token)

    print("\n=== Uploading algorithm container image ===")
    print(f"File: {IMAGE_PATH} ({os.path.getsize(IMAGE_PATH)/1e9:.2f} GB)")

    # Upload the container image to the algorithm
    from pathlib import Path
    image_path = Path(IMAGE_PATH)

    # Use the raw API to upload
    # Step 1: Create upload session via s3-file-field
    upload_session = c.post(
        f"{GC}/api/v1/algorithms/{ALGORITHM_PK}/images/upload-session/",
        json={"image": str(image_path.name)},
    )
    print(f"Upload session: {upload_session}")

def upload_simple(token):
    """Upload using gcapi's built-in upload mechanism."""
    import gcapi
    import httpx

    c = gcapi.Client(token=token)

    print("\n=== Step 1: Upload algorithm image ===")

    # Use the algorithm images API endpoint
    with open(IMAGE_PATH, "rb") as f:
        resp = c.post(
            url=f"{GC}/api/v1/algorithms/{ALGORITHM_PK}/images/",
            content=None,
            files={"image": (os.path.basename(IMAGE_PATH), f, "application/gzip")},
            data={"comment": "v6-resubmit"},
        )
    print(f"Upload response: {resp}")
    return resp

def upload_with_raw_api(token):
    """Upload using raw httpx with token auth, following GC's upload flow."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"}
    base = GC

    print("\n=== Step 1: Initialize multipart upload ===")
    file_size = os.path.getsize(IMAGE_PATH)
    chunk_size = 64 * 1024 * 1024
    part_count = math.ceil(file_size / chunk_size)
    filename = os.path.basename(IMAGE_PATH)

    with httpx.Client(timeout=120, follow_redirects=True, headers=headers) as client:
        # Initialize upload via s3-file-field
        resp = client.post(f"{base}/api/v1/s3-file-field/upload-initialize/", json={
            "field_id": "algorithm_image-image",
            "file_name": filename,
            "file_size": file_size,
            "content_type": "application/gzip",
        })
        if resp.status_code != 200:
            # Try alternative endpoint
            print(f"s3-file-field init: HTTP {resp.status_code}")
            print(resp.text[:300])

            # Try direct upload to algorithm images
            print("\nTrying direct upload...")
            with open(IMAGE_PATH, "rb") as f:
                resp = client.post(
                    f"{base}/api/v1/algorithms/{ALGORITHM_PK}/images/",
                    files={"image": (filename, f, "application/gzip")},
                    data={"comment": "v6-resubmit"},
                )
            print(f"Direct upload: HTTP {resp.status_code}")
            print(resp.text[:500])
            return resp.json() if resp.status_code < 300 else None

        init_data = resp.json()
        print(f"Upload ID: {init_data.get('upload_id', 'N/A')}")
        print(f"Parts: {len(init_data.get('parts', []))}")

        # Upload parts
        print("\n=== Step 2: Uploading parts ===")
        parts = init_data.get("parts", [])
        completed = []
        with open(IMAGE_PATH, "rb") as f:
            for i, part in enumerate(parts):
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                put_resp = httpx.put(part["url"], content=chunk,
                                     headers={"Content-Type": "application/gzip"},
                                     timeout=600)
                etag = put_resp.headers.get("ETag", "").strip('"')
                completed.append({"part_number": part["part_number"], "etag": etag})
                print(f"  Part {i+1}/{len(parts)} ({(i+1)/len(parts)*100:.0f}%)", flush=True)

        # Complete upload
        print("\n=== Step 3: Complete upload ===")
        resp = client.post(f"{base}/api/v1/s3-file-field/upload-complete/", json={
            "upload_signature": init_data.get("upload_signature", ""),
            "upload_id": init_data["upload_id"],
            "object_key": init_data["object_key"],
            "parts": completed,
        })
        complete_data = resp.json()
        print(f"Complete: {json.dumps(complete_data, indent=2)[:300]}")

        # Create algorithm image
        print("\n=== Step 4: Create algorithm image ===")
        resp = client.post(f"{base}/api/v1/algorithms/{ALGORITHM_PK}/images/", json={
            "image": complete_data.get("file_key", complete_data.get("object_key")),
            "comment": "v6-resubmit",
        })
        print(f"Create image: HTTP {resp.status_code}")
        print(resp.text[:500])
        if resp.status_code < 300:
            return resp.json()
        return None


if __name__ == "__main__":
    sid, csrf = cookies()
    print(f"Session: {sid[:10]}..., CSRF: {csrf[:10]}...")

    # Get API token
    token = get_or_create_token(sid, csrf)

    # Try upload
    result = upload_with_raw_api(token)
    if result:
        image_pk = result.get("pk")
        print(f"\nAlgorithm image PK: {image_pk}")

        # Wait for import
        import httpx
        headers = {"Authorization": f"Bearer {token}"}
        print("\n=== Step 5: Waiting for import ===")
        for attempt in range(30):
            time.sleep(30)
            resp = httpx.get(f"{GC}/api/v1/algorithms/images/{image_pk}/",
                           headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("import_status", "Unknown")
                print(f"  Import: {status}", flush=True)
                if status == "Completed":
                    break
                elif status == "Failed":
                    print("IMPORT FAILED!")
                    sys.exit(1)

        # Activate and submit
        print("\n=== Step 6: Activate image ===")
        cookie = f"sessionid={sid}; _csrftoken={csrf}"
        req = urllib.request.Request(f"{GC}/algorithms/{ALGORITHM_SLUG}/images/{image_pk}/")
        req.add_header("Cookie", cookie)
        resp = urllib.request.urlopen(req, timeout=30)
        html = resp.read().decode()
        csrf_form = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
        form_csrf = csrf_form.group(1) if csrf_form else csrf

        data = urllib.parse.urlencode({
            "csrfmiddlewaretoken": form_csrf,
            "algorithm_image": image_pk,
            "save": "Activate algorithm image",
        }).encode()
        req = urllib.request.Request(f"{GC}/algorithms/{ALGORITHM_SLUG}/images/activate/",
                                     data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        req.add_header("Cookie", cookie)
        req.add_header("Referer", f"{GC}/algorithms/{ALGORITHM_SLUG}/images/{image_pk}/")
        resp = urllib.request.urlopen(req, timeout=30)
        print(f"Activated: HTTP {resp.status}")

        # Submit preliminary
        print("\n=== Step 7: Submit preliminary ===")
        req = urllib.request.Request(f"{CH}/evaluation/photon-dose-preliminary-testing/submissions/create/")
        req.add_header("Cookie", cookie)
        resp = urllib.request.urlopen(req, timeout=30)
        html = resp.read().decode()
        csrf_form = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
        form_csrf = csrf_form.group(1) if csrf_form else csrf

        data = urllib.parse.urlencode({
            "csrfmiddlewaretoken": form_csrf,
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
            print(f"SUCCESS! Redirected to: {resp.url}")
        else:
            body = resp.read().decode()
            if "already exists" in body:
                print("DUPLICATE - already submitted this combination")
            else:
                errors = re.findall(r'(?:error|alert)[^>]*>(.*?)<', body, re.IGNORECASE)
                print(f"Result unclear. Errors: {[e.strip() for e in errors if e.strip()][:5]}")
    else:
        print("Upload failed!")
        sys.exit(1)
