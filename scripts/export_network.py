"""Export the replica's estimated network for one date: shift factors and the resource-to-settlement-point map.

The shift factors are fitted on the 14 days before DATE from public prices (LMP - lambda regressed on ERCOT's
published shadow prices), exactly as the replica uses them for DATE. They are estimates, not ERCOT's own shift
factors, and cover only constraints that bound in that window: a constraint missing from the file is unknown,
not zero. Needs ERCOT Public API credentials in .env.

  uv run --env-file .env python scripts/export_network.py 2026-09-25 network_export
"""

import sys

import gridstatus
import pandas as pd
from gridstatus.ercot_api.ercot_api import ErcotAPI

from scedrep.fetch import get_sced_disclosure
from scedrep.network import export_network
from scedrep.pipeline import fit_network


def main(date, out_dir="network_export"):
    disc = get_sced_disclosure(date)  # resource names of generators and batteries on DATE
    resources = pd.concat([disc["sced_gen_resource"], disc["sced_esr"]])
    node, SF = fit_network(ErcotAPI(), gridstatus.Ercot(), date, resources)
    for path in export_network(SF, node, out_dir, date):
        print("wrote", path)
    print(f"{SF.shape[0]} constraints x {SF.shape[1]} settlement points; {int(node.notna().sum())} resources mapped")


if __name__ == "__main__":
    main(*sys.argv[1:3])
