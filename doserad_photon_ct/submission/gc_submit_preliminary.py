#!/usr/bin/env python3
"""Upload algorithm image to GC and submit to Preliminary Photon CT phase."""
import json
import math
import os
import sys
import time
import urllib.request
import urllib.parse
import re

# --- Config ---
ALGORITHM_PK = "322e2c78-1c95-421a-a6f2-f0176d9f6bb6"
ALGORITHM_SLUG = "photon-ct-dose-engine-v1"
PHASE_PK = "f2ccbdbf-13e2-4479-b6c0-23fa45bf9a75"  # Preliminary Photon CT
CREATOR_ID = "150744"
MODEL_PK = "21dfed4a-cc93-43e1-810b-f7f9441f9f67"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "dist", "doserad2026-photon-ct-v6-resub.tar.gz")
COMMENT = "v6-resubmit"
COOKIE_FILE = "/tmp/gc_cookies_new.txt"
GC_BASE = "https://grand-challenge.org"
CHALLENGE_BASE = "https://doserad2026.grand-challenge.org"
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB

def load_cookies(path):
    sid = csrf = None
    with open(path) as f:
        for line in f:
            if 'sessionid' in line:
                sid = line.strip().split('\t')[-1]
            if '_csrftoken' in line or 'csrftoken' in line:
                csrf = line.strip().split('\t')[-1]
    return sid, csrf

def api_request(url, sid, csrf, method='GET', data=None, content_type='application/json'):
    if data and isinstance(data, dict):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    if method in ('POST', 'PUT', 'PATCH'):
        req.add_header('X-CSRFToken', csrf)
        req.add_header('Referer', GC_BASE + '/')
    if content_type and data:
        req.add_header('Content-Type', content_type)
    resp = urllib.request.urlopen(req, timeout=120)
    return resp

def step1_create_upload(sid, csrf):
    """Create a UserUpload session."""
    filename = os.path.basename(IMAGE_PATH)
    data = json.dumps({"filename": filename}).encode()
    req = urllib.request.Request(
        f"{GC_BASE}/api/v1/cases/uploads/",
        data=data, method='POST'
    )
    req.add_header('Content-Type', 'application/json')
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    req.add_header('X-CSRFToken', csrf)
    req.add_header('Referer', GC_BASE + '/')
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read().decode())

def step1b_upload_via_form(sid, csrf):
    """Upload image via the algorithm page form (chunked upload via S3 presigned URLs)."""
    # GC uses django-s3-file-field for large uploads
    # Step 1: Get presigned URL for multipart upload
    file_size = os.path.getsize(IMAGE_PATH)
    filename = os.path.basename(IMAGE_PATH)

    # Initialize multipart upload via s3-file-field
    init_data = json.dumps({
        "upload_signature": "",
        "field_id": "images-image",
        "file_name": filename,
        "file_size": file_size,
        "content_type": "application/gzip",
        "part_count": math.ceil(file_size / CHUNK_SIZE),
    }).encode()

    req = urllib.request.Request(
        f"{GC_BASE}/api/v1/s3-file-field/upload-initialize/",
        data=init_data, method='POST'
    )
    req.add_header('Content-Type', 'application/json')
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    req.add_header('X-CSRFToken', csrf)
    req.add_header('Referer', GC_BASE + '/')

    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read().decode())
    print(f"Upload initialized: {json.dumps(result, indent=2)[:500]}")
    return result

def upload_parts(init_result, sid, csrf):
    """Upload file parts to S3 using presigned URLs."""
    parts = init_result.get('parts', [])
    file_size = os.path.getsize(IMAGE_PATH)
    upload_id = init_result.get('upload_id', '')
    object_key = init_result.get('object_key', '')

    completed_parts = []
    with open(IMAGE_PATH, 'rb') as f:
        for i, part in enumerate(parts):
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            url = part['url']
            req = urllib.request.Request(url, data=chunk, method='PUT')
            req.add_header('Content-Type', 'application/gzip')
            resp = urllib.request.urlopen(req, timeout=600)
            etag = resp.headers.get('ETag', '').strip('"')
            completed_parts.append({
                'part_number': part['part_number'],
                'etag': etag,
            })
            pct = (i + 1) / len(parts) * 100
            print(f"  Uploaded part {i+1}/{len(parts)} ({pct:.0f}%)", flush=True)

    # Complete multipart upload
    complete_data = json.dumps({
        "upload_signature": init_result.get('upload_signature', ''),
        "upload_id": upload_id,
        "object_key": object_key,
        "parts": completed_parts,
    }).encode()

    req = urllib.request.Request(
        f"{GC_BASE}/api/v1/s3-file-field/upload-complete/",
        data=complete_data, method='POST'
    )
    req.add_header('Content-Type', 'application/json')
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    req.add_header('X-CSRFToken', csrf)
    req.add_header('Referer', GC_BASE + '/')

    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read().decode())
    print(f"Upload complete: {json.dumps(result, indent=2)[:300]}")
    return result

def create_algorithm_image(upload_result, sid, csrf):
    """Create algorithm image from upload."""
    data = json.dumps({
        "algorithm": f"{GC_BASE}/api/v1/algorithms/{ALGORITHM_PK}/",
        "image": upload_result.get('file_key', upload_result.get('object_key', '')),
        "comment": COMMENT,
    }).encode()

    req = urllib.request.Request(
        f"{GC_BASE}/api/v1/algorithms/images/",
        data=data, method='POST'
    )
    req.add_header('Content-Type', 'application/json')
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    req.add_header('X-CSRFToken', csrf)
    req.add_header('Referer', GC_BASE + '/')

    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read().decode())
    print(f"Algorithm image created: PK={result.get('pk')}")
    return result

def wait_for_import(image_pk, sid, csrf, timeout=600):
    """Wait for algorithm image import to complete."""
    print(f"Waiting for import of {image_pk}...")
    start = time.time()
    while time.time() - start < timeout:
        req = urllib.request.Request(f"{GC_BASE}/api/v1/algorithms/images/{image_pk}/")
        req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read().decode())
            status = data.get('import_status', 'Unknown')
            print(f"  Import status: {status}", flush=True)
            if status == 'Completed':
                return data
            elif status == 'Failed':
                print("Import FAILED!")
                return None
        except Exception as e:
            print(f"  Check failed: {e}")
        time.sleep(30)
    print("Import timeout!")
    return None

def activate_image(image_pk, sid, csrf):
    """Activate the algorithm image."""
    # Get the form page first for fresh CSRF
    req = urllib.request.Request(
        f"{GC_BASE}/algorithms/{ALGORITHM_SLUG}/images/{image_pk}/"
    )
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()
    csrf_match = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
    form_csrf = csrf_match.group(1) if csrf_match else csrf

    form_data = urllib.parse.urlencode({
        'csrfmiddlewaretoken': form_csrf,
        'algorithm_image': image_pk,
        'save': 'Activate algorithm image'
    }).encode()

    req = urllib.request.Request(
        f"{GC_BASE}/algorithms/{ALGORITHM_SLUG}/images/activate/",
        data=form_data, method='POST'
    )
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    req.add_header('Referer', f"{GC_BASE}/algorithms/{ALGORITHM_SLUG}/images/{image_pk}/")

    resp = urllib.request.urlopen(req, timeout=30)
    print(f"Image activated: HTTP {resp.status}, URL: {resp.url}")
    return True

def submit_preliminary(image_pk, sid, csrf):
    """Submit to preliminary test phase."""
    # Get fresh CSRF from form
    req = urllib.request.Request(
        f"{CHALLENGE_BASE}/evaluation/photon-dose-preliminary-testing/submissions/create/"
    )
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()
    csrf_match = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
    form_csrf = csrf_match.group(1) if csrf_match else csrf

    form_data = urllib.parse.urlencode({
        'csrfmiddlewaretoken': form_csrf,
        'creator': CREATOR_ID,
        'phase': PHASE_PK,
        'algorithm': ALGORITHM_PK,
        'algorithm_image': image_pk,
        'algorithm_model': MODEL_PK,
    }).encode()

    req = urllib.request.Request(
        f"{CHALLENGE_BASE}/evaluation/photon-dose-preliminary-testing/submissions/create/",
        data=form_data, method='POST'
    )
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    req.add_header('Cookie', f'sessionid={sid}; _csrftoken={csrf}')
    req.add_header('Referer', f"{CHALLENGE_BASE}/evaluation/photon-dose-preliminary-testing/submissions/create/")

    resp = urllib.request.urlopen(req, timeout=30)
    body = resp.read().decode()

    if resp.url != f"{CHALLENGE_BASE}/evaluation/photon-dose-preliminary-testing/submissions/create/":
        print(f"Submission successful! Redirected to: {resp.url}")
        return True

    # Check for errors
    errors = re.findall(r'already exists|duplicate|error', body, re.IGNORECASE)
    if errors:
        print(f"Submission failed: {errors}")
        return False

    print(f"Submission result unclear. URL: {resp.url}")
    return False


if __name__ == '__main__':
    sid, csrf = load_cookies(COOKIE_FILE)
    if not sid or not csrf:
        print("Missing session cookies! Please login first.")
        sys.exit(1)

    print(f"Session: {sid[:10]}..., CSRF: {csrf[:10]}...")
    print(f"Image: {IMAGE_PATH}")
    print(f"File size: {os.path.getsize(IMAGE_PATH) / 1e9:.2f} GB")
    print()

    # Step 1: Initialize upload
    print("=== Step 1: Initialize upload ===")
    init_result = step1b_upload_via_form(sid, csrf)

    # Step 2: Upload parts
    print("\n=== Step 2: Upload file parts ===")
    upload_result = upload_parts(init_result, sid, csrf)

    # Step 3: Create algorithm image
    print("\n=== Step 3: Create algorithm image ===")
    algo_image = create_algorithm_image(upload_result, sid, csrf)
    image_pk = algo_image.get('pk')

    # Step 4: Wait for import
    print("\n=== Step 4: Wait for import ===")
    imported = wait_for_import(image_pk, sid, csrf, timeout=900)
    if not imported:
        print("Import failed or timed out!")
        sys.exit(1)

    # Step 5: Activate image
    print("\n=== Step 5: Activate image ===")
    activate_image(image_pk, sid, csrf)

    # Step 6: Submit
    print("\n=== Step 6: Submit to preliminary ===")
    success = submit_preliminary(image_pk, sid, csrf)

    if success:
        print("\n=== DONE! Submission submitted successfully ===")
    else:
        print("\n=== Submission may have failed, check GC manually ===")
