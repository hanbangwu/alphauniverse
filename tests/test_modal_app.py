import importlib


def test_the_benchmark_and_the_modal_app_import() -> None:
    benchmark = importlib.import_module("scripts.benchmark")

    assert benchmark.app is importlib.import_module("modal_app").app
