import importlib


def test_the_benchmark_and_the_modal_app_import() -> None:
    importlib.import_module("modal_app")
    importlib.import_module("scripts.benchmark")
