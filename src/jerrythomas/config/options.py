"""Option values shared by configuration models and CLI parsers."""

OUTPUT_TRANSPORTS = ("stdout", "fs")
OUTPUT_FORMATS = ("jsonl", "csv", "parquet", "pickle")
OUTPUT_INSPECT_FORMATS = ("jsonl", "csv", "pickle", "txt", "html")
OUTPUT_STDOUT_FORMATS = ("jsonl", "txt")
OUTPUT_VIEWS = ("raw", "flat")

VISUAL_CHOICES = ("on", "off")
LOG_TRANSPORT_CHOICES = ("stderr", "stdout", "fs")
LOG_SCOPE_CHOICES = ("global", "execution")
