# For testing functionality for Github Actions CI checks
import subprocess
import time


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
        process = subprocess.Popen(["uvicorn", "main:app", "--reload"])
        time.sleep(2)
        process.terminate()
        process.wait()
    except Exception as e:
        assert False, f"Failed to start/stop FastAPI server: {e}"
    assert True


def test_omni_supervisor_agent() -> None:
    """Test Omni Supervisor Agent"""
    try:
        from main import supervisor

        result = supervisor.invoke({"messages": [{"role": "user", "content": "Hi!"}]})
    except Exception as e:
        assert False, f"Failed to invoke supervisor: {e}"
    assert result is not None, f"Supervisor returned None: {result}"


def test_omni_light_agent() -> None:
    """Test Omni Light Agent"""
    try:
        from main import light

        result = light.invoke({"messages": [{"role": "user", "content": "Hi!"}]})
    except Exception as e:
        assert False, f"Failed to invoke light agent: {e}"
    assert result is not None, f"Light agent returned None: {result}"
