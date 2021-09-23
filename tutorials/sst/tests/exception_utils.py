from click.testing import Result


def print_exception(result: Result):
    if result.exception:
        print(result.exception)
        print(result.stdout)
        print(result.stderr)
