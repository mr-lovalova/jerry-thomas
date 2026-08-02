from jerrythomas.cli.version import short_version, version_report


def handle() -> None:
    print(short_version())


def handle_env() -> None:
    print(version_report())
