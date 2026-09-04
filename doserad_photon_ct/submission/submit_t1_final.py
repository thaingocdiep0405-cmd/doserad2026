#!/usr/bin/env python3
"""Submit Task 1 (Photon CT) to Final phase with PDF."""
import os, re, sys
import urllib.request
import http.client
import mimetypes

COOKIE_FILE = "/tmp/gc_cookies_new.txt"
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "paper", "DoseRAD2026_Task1_PhotonCT_thaingocdiep.pdf")
ALGORITHM_PK = "322e2c78-1c95-421a-a6f2-f0176d9f6bb6"
IMAGE_PK = "c554ba0d-5c64-4f1c-bda0-4dd5ac2e778f"
MODEL_PK = "21dfed4a-cc93-43e1-810b-f7f9441f9f67"
PHASE_PK = "6717f650-4432-40ae-85e6-1783e4fe7e8c"
CREATOR_ID = "150744"
CH = "https://doserad2026.grand-challenge.org"
SUBMIT_URL = f"{CH}/evaluation/final-testing-photon-dose-on-ct/submissions/create/"


def load_cookies():
    sid = csrf = None
    with open(COOKIE_FILE) as f:
        for line in f:
            if "sessionid" in line:
                sid = line.strip().split("\t")[-1]
            if "csrftoken" in line:
                csrf = line.strip().split("\t")[-1]
    return sid, csrf


def multipart_encode(fields, files):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = b""
    for key, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    for key, (filename, filedata, content_type) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode()
        body += filedata
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def submit_final(sid, csrf):
    cookie = f"sessionid={sid}; _csrftoken={csrf}"

    # Get fresh CSRF
    req = urllib.request.Request(SUBMIT_URL)
    req.add_header("Cookie", cookie)
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()
    csrf_match = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)', html)
    form_csrf = csrf_match.group(1) if csrf_match else csrf

    # Read PDF
    with open(PDF_PATH, "rb") as f:
        pdf_data = f.read()
    print(f"PDF: {PDF_PATH} ({len(pdf_data)} bytes)")

    fields = {
        "csrfmiddlewaretoken": form_csrf,
        "creator": CREATOR_ID,
        "phase": PHASE_PK,
        "algorithm": ALGORITHM_PK,
        "algorithm_image": IMAGE_PK,
        "algorithm_model": MODEL_PK,
    }
    files = {
        "supplementary_file": (
            os.path.basename(PDF_PATH),
            pdf_data,
            "application/pdf",
        ),
    }

    body, content_type = multipart_encode(fields, files)
    print(f"Request body: {len(body)} bytes")

    req = urllib.request.Request(SUBMIT_URL, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Cookie", cookie)
    req.add_header("Referer", SUBMIT_URL)

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        result_url = resp.url
        result_body = resp.read().decode()

        if result_url != SUBMIT_URL:
            print(f"SUCCESS! Redirected to: {result_url}")
            return True

        if "already exists" in result_body.lower():
            print("DUPLICATE - already submitted this combination")
        else:
            errors = re.findall(
                r'(?:errorlist|alert-danger|error)[^>]*>(.*?)<',
                result_body,
                re.IGNORECASE,
            )
            clean = [re.sub(r"<[^>]+>", "", e).strip() for e in errors if e.strip()]
            if clean:
                print(f"Errors: {clean}")
            else:
                print(f"Result unclear. URL: {result_url}")
        return False
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:500]}")
        return False


if __name__ == "__main__":
    sid, csrf = load_cookies()
    if not sid or not csrf:
        print("Missing cookies!")
        sys.exit(1)

    print(f"Session: {sid[:10]}...")
    print(f"Algorithm: {ALGORITHM_PK}")
    print(f"Image: {IMAGE_PK}")
    print(f"Model: {MODEL_PK}")
    print(f"Phase: Final Photon CT ({PHASE_PK[:8]})")
    print()

    success = submit_final(sid, csrf)
    if success:
        print("\nFinal submission complete!")
    else:
        print("\nSubmission may have failed. Check GC manually.")
