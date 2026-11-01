"""
End-to-end smoke test of the real FastAPI app (needs fastapi + httpx installed).

    python -m tests.smoke_test
"""
from fastapi.testclient import TestClient

from app.main import app

with TestClient(app) as client:   # "with" runs the startup (lifespan) code
    for path in ["/", "/classify", "/metrics", "/about", "/docs"]:
        r = client.get(path)
        assert r.status_code == 200, (path, r.status_code)
        print("GET ", path, r.status_code)

    r = client.post("/classify", data={"subject": "Charged twice",
                                       "body": "I was billed twice for my subscription, please refund."})
    assert r.status_code == 200 and "Billing" in r.text
    print("POST /classify (billing)", r.status_code)

    r = client.post("/classify", data={"subject": "Service is down", "body": "Not working, fix immediately."})
    assert "Urgent" in r.text
    print("POST /classify (urgent) ok")

    r = client.post("/classify", data={"subject": "xqzv", "body": "zzzz qqqq wwww"})
    assert "Needs Human Review" in r.text
    print("POST /classify (low confidence) ok")

    r = client.post("/classify", data={"subject": "", "body": ""})
    assert r.status_code == 422 and "Enter a subject" in r.text
    print("POST /classify (empty) rejected ok")

    assert client.get("/static/css/style.css").status_code == 200
    print("static css ok\nALL SMOKE TESTS PASSED")
