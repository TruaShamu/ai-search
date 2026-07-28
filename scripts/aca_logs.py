"""Query Azure Container Apps console logs from Log Analytics.

The ACA environment ships container stdout to Log Analytics. Reading it via
`az monitor log-analytics query` from PowerShell is painful because the KQL and
JSON quoting fight the shell, so this wraps it.

Examples::

    python -m scripts.aca_logs --containers          # what is logging at all
    python -m scripts.aca_logs --app embed-worker    # one container's output
    python -m scripts.aca_logs --app qdrant --grep ERROR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

DEFAULT_WORKSPACE = "6aadb6cf-5985-4835-9aed-9c32b6f7e7b8"
TABLE = "ContainerAppConsoleLogs_CL"


def run_query(workspace: str, kql: str) -> list[dict]:
    # shell=False, and az.cmd by name: routing through cmd.exe lets it consume
    # the pipe characters that KQL is built from, which corrupts every query.
    exe = "az.cmd" if sys.platform == "win32" else "az"
    proc = subprocess.run(
        [exe, "monitor", "log-analytics", "query", "--workspace", workspace, "--analytics-query", kql, "-o", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "").strip()
    if not out.startswith("["):
        sys.exit(f"Query failed (exit {proc.returncode}):\n{proc.stderr or out}")
    return json.loads(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Log Analytics workspace customer ID")
    parser.add_argument("--app", help="Container app or job name to filter on")
    parser.add_argument("--minutes", type=int, default=60, help="Look back this many minutes")
    parser.add_argument("--limit", type=int, default=100, help="Max log lines")
    parser.add_argument("--grep", help="Only show lines containing this substring")
    parser.add_argument("--containers", action="store_true", help="List which containers are logging, then exit")
    args = parser.parse_args()

    if args.containers:
        kql = (
            f"{TABLE} | where TimeGenerated > ago({args.minutes}m) "
            "| summarize cnt=count(), last=max(TimeGenerated) "
            "by ContainerAppName_s, ContainerJobName_s"
        )
        for row in run_query(args.workspace, kql):
            app = row.get("ContainerAppName_s") or "-"
            job = row.get("ContainerJobName_s") or "-"
            print(f"app={app:20} job={job:20} lines={row.get('cnt'):>6}  last={str(row.get('last'))[11:19]}")
        return

    where = [f"TimeGenerated > ago({args.minutes}m)"]
    if args.app:
        where.append(f'(ContainerAppName_s == "{args.app}" or ContainerJobName_s == "{args.app}")')
    if args.grep:
        where.append(f'Log_s contains "{args.grep}"')

    kql = (
        f"{TABLE} | where {' and '.join(where)} "
        "| project TimeGenerated, ContainerJobName_s, Log_s "
        f"| order by TimeGenerated asc | take {args.limit}"
    )
    rows = run_query(args.workspace, kql)
    print(f"{len(rows)} log lines\n")
    for row in rows:
        print(f"{str(row.get('TimeGenerated'))[11:19]}  {row.get('Log_s')}")


if __name__ == "__main__":
    main()
