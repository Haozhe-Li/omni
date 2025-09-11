# For testing functionality for Github Actions CI checks


def test_main_import():
    try:
        import main
    except ImportError:
        assert False, "Failed to import main module"
    assert True
