import logging

_LOGGER = logging.getLogger("jerrythomas.cli")


def prompt_required(prompt: str) -> str:
    value = input(f"{prompt}: ").strip()
    if not value:
        raise SystemExit(f"{prompt} is required")
    return value


def choose_name(prompt: str, default: str | None = None) -> str:
    if not default:
        return prompt_required(prompt)
    _LOGGER.info("%s:", prompt)
    _LOGGER.info("  [1] %s (default)", default)
    _LOGGER.info("  [2] Custom name")
    while True:
        selection = input("> ").strip()
        if selection in {"", "1"}:
            return default
        if selection == "2":
            return prompt_required(prompt)
        _LOGGER.info("Please enter a number from the list.")


def pick_from_list(prompt: str, options: list[str]) -> str:
    _LOGGER.info(prompt)
    for index, option in enumerate(options, 1):
        _LOGGER.info("  [%d] %s", index, option)
    while True:
        selection = input("> ").strip()
        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(options):
                return options[index - 1]
        _LOGGER.info("Please enter a number from the list.")


def pick_from_menu(
    prompt: str,
    options: list[tuple[str, str]],
    allow_default: bool = True,
) -> str:
    _LOGGER.info(prompt)
    for index, (_, label) in enumerate(options, 1):
        _LOGGER.info("  [%d] %s", index, label)
    while True:
        selection = input("> ").strip()
        if selection == "" and allow_default:
            return options[0][0]
        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(options):
                return options[index - 1][0]
        _LOGGER.info("Please enter a number from the list.")


def pick_multiple_from_list(prompt: str, options: list[str]) -> list[str]:
    _LOGGER.info(prompt)
    for index, option in enumerate(options, 1):
        _LOGGER.info("  [%d] %s", index, option)
    selection = input("> ").strip()
    try:
        indexes = [int(value) for value in selection.split(",") if value.strip()]
    except ValueError:
        raise SystemExit("Invalid selection.") from None
    if not indexes:
        raise SystemExit("No inputs selected.")
    if any(index < 1 or index > len(options) for index in indexes):
        raise SystemExit("Invalid selection.")
    return [options[index - 1] for index in indexes]


def choose_dto(
    existing: list[str],
    default: str | None = None,
) -> tuple[str, bool]:
    _LOGGER.info("DTO:")
    _LOGGER.info("  [1] Create new DTO (default)")
    _LOGGER.info("  [2] Select existing DTO")
    while True:
        selection = input("> ").strip() or "1"
        if selection == "1":
            return choose_name("DTO class name", default), True
        if selection == "2":
            if not existing:
                raise SystemExit("No existing DTO found.")
            return pick_from_list("Select DTO:", existing), False
        _LOGGER.info("Please enter a number from the list.")


def choose_domain(
    existing: list[str],
    default: str | None = None,
) -> tuple[str, bool]:
    _LOGGER.info("Domain:")
    _LOGGER.info("  [1] Create new domain (default)")
    _LOGGER.info("  [2] Select existing domain")
    while True:
        selection = input("> ").strip() or "1"
        if selection == "1":
            return choose_name("Domain name", default), True
        if selection == "2":
            if not existing:
                raise SystemExit("No existing domain found.")
            return pick_from_list("Select domain:", existing), False
        _LOGGER.info("Please enter a number from the list.")
