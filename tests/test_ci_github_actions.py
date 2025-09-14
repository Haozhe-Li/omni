# For testing functionality for Github Actions CI checks
import subprocess
import time
import requests


def test_main_compile() -> None:
    """Test compilation of main.py"""
    try:
        subprocess.run(["python", "-m", "py_compile", "main.py"], check=True)
    except subprocess.CalledProcessError:
        assert False, "Failed to compile main.py"
    assert True


def test_main_import() -> None:
    """Test import of main module"""
    try:
        import main
    except ImportError:
        assert False, "Failed to import main module"
    assert True


def test_fastapi_server() -> None:
    """Test FastAPI server"""
    try:
        with subprocess.Popen(["uvicorn", "main:app", "--reload"]) as process:
            time.sleep(2)
            res = requests.get("http://127.0.0.1:8000/health")
            assert res.status_code == 200
            process.terminate()
            process.wait()
    except Exception as e:
        assert False, f"Failed to start/stop FastAPI server: {e}"
    assert True
