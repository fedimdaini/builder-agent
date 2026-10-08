"""Decide layer: RepoContext + contracts.yaml -> BuildPlan, by rules only (no LLM).

    from builder_agent.scan import scan_repo
    from builder_agent.decide import load_contracts, plan_build
    plan = plan_build(scan_repo(repo), load_contracts("contracts.yaml"))
    print(plan.summary())
"""

from __future__ import annotations

from ..scan.models import RepoContext
from .contracts import Contracts, load_contracts
from .models import BuildPlan, SamplePlan
from .rules import choose_python, plan_install, plan_serving, plan_targets

__all__ = ["plan_build", "load_contracts", "BuildPlan", "Contracts"]


def plan_build(ctx: RepoContext, contracts: Contracts) -> BuildPlan:
    python, py_need = choose_python(ctx, contracts)
    serving, serving_need = plan_serving(ctx, contracts)
    install, install_need = plan_install(ctx, contracts, serving)
    targets, target_needs = plan_targets(ctx, contracts, install, serving)

    needs = [n for n in (py_need, install_need, serving_need) if n] + target_needs
    return BuildPlan(
        repo=ctx.name,
        contract_version=contracts.contract_version,
        python=python,
        install=install,
        serving=serving,
        make_targets=targets,
        sample=SamplePlan(variable=contracts.sample_mode.variable, fraction=contracts.sample_mode.fraction),
        needs_llm=needs,
    )
