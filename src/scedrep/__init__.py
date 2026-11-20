import sys


def main() -> None:
    from scedrep.pipeline import run_date

    run_date(sys.argv[1])
