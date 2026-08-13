import httpx


BASE_URL = "http://127.0.0.1:8000"


def test_normal_prediction():
    response = httpx.post(
        f"{BASE_URL}/predict",
        json={
            "text": "Saya suka membuat desain UI dan cukup mahir menggunakan Figma."
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "text" in data
    assert "predictions" in data

    for label in ["love", "good_at", "world_needs", "paid_for"]:
        assert label in data["predictions"]

    print("[PASS] Normal prediction")


def test_multiple_aspects():
    response = httpx.post(
        f"{BASE_URL}/predict",
        json={
            "text": (
                "Saya suka membuat aplikasi, cukup mahir programming, "
                "dan ingin kemampuan saya bermanfaat bagi orang lain."
            )
        }
    )

    assert response.status_code == 200

    print("[PASS] Multiple aspects")


def test_short_text():
    response = httpx.post(
        f"{BASE_URL}/predict",
        json={
            "text": "Saya suka coding."
        }
    )

    assert response.status_code == 200

    print("[PASS] Short text")


def test_empty_text():
    response = httpx.post(
        f"{BASE_URL}/predict",
        json={
            "text": ""
        }
    )

    assert response.status_code == 422

    print("[PASS] Empty text")


def test_missing_text():
    response = httpx.post(
        f"{BASE_URL}/predict",
        json={}
    )

    assert response.status_code == 422

    print("[PASS] Missing text")


def test_long_text():
    response = httpx.post(
        f"{BASE_URL}/predict",
        json={
            "text": (
                "Saya suka belajar dan membuat berbagai macam hal baru. "
                "Saya menikmati proses mengembangkan kemampuan saya, "
                "membantu orang lain, bekerja sama dengan orang lain, "
                "dan mencari pekerjaan yang sesuai dengan kemampuan saya."
            )
        }
    )

    assert response.status_code == 200

    print("[PASS] Long text")


if __name__ == "__main__":

    print("=" * 60)
    print("MINDREST AI API - AUTOMATED TEST")
    print("=" * 60)

    tests = [
        test_normal_prediction,
        test_multiple_aspects,
        test_short_text,
        test_empty_text,
        test_missing_text,
        test_long_text,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}")
            print(f"       {e}")

    print("=" * 60)
    print(f"RESULT: {passed}/{len(tests)} TEST PASSED")
    print("=" * 60)