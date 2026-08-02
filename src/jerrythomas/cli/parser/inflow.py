import argparse


def add_inflow_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "inflow",
        help="create end-to-end inflow scaffolds",
        parents=[common],
    )
    inflow_sub = parser.add_subparsers(dest="inflow_cmd", required=True)
    inflow_sub.add_parser("create", help="create an inflow")
