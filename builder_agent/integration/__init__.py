"""The integration fields of contracts.yaml that no teammate has confirmed yet.

    python -m builder_agent.integration [--contracts contracts.yaml] [--strict]

Lists every field with confirmed: false, grouped by who confirms it, with its value and, if it was changed,
its default. --strict exits 1 while any field is unconfirmed. The questions to ask: docs/INTEGRATION.md.
"""

from __future__ import annotations

import json

from ..decide.contracts import Contracts, IntegrationField


def unconfirmed(c: Contracts) -> list[IntegrationField]:
    return [f for f in c.integration_fields() if not f.confirmed]


def _show(value) -> str:
    return json.dumps(value) if not isinstance(value, str) else value


def report(c: Contracts) -> str:
    fields = c.integration_fields()
    todo = [f for f in fields if not f.confirmed]
    out = [f"INTEGRATION: {len(todo)} of {len(fields)} field(s) not confirmed yet "
           f"({sum(f.at_default for f in todo)} still at the Builder's default)"]
    for group in dict.fromkeys(f.group for f in todo):
        out.append(f"  {group} (to confirm):")
        for f in (f for f in todo if f.group == group):
            note = "default" if f.at_default else f"changed, default {_show(f.default)}"
            out.append(f"    {f.name} = {_show(f.value)}  [{note}]")
    return "\n".join(out)
