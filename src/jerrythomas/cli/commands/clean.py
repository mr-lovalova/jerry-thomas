from jerrythomas.services.temp_cleanup import (
    clean_temp_dirs,
    format_age,
    format_bytes,
    parse_age,
)


def handle(*, yes: bool, older_than: str | None = None) -> None:
    try:
        age = parse_age(older_than)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    result = clean_temp_dirs(yes=yes, older_than=age)
    if not result.candidates:
        print("No Jerry sort spill directories found.")
        return

    action = "Removed" if yes else "Found"
    noun = "directory" if len(result.candidates) == 1 else "directories"
    print(f"{action} {len(result.candidates)} Jerry sort spill {noun}:")
    for item in result.candidates:
        print(
            f"  {item.path}  size={format_bytes(item.size_bytes)}  age={format_age(item.age_seconds)}"
        )
    print(f"Total: {format_bytes(result.total_bytes)}")
    if not yes:
        print("Dry run only. Run `jerry clean --yes` to delete these directories.")
