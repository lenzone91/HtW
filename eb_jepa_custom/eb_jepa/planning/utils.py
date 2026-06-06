import sys
from importlib import import_module
from pathlib import Path


planner_name_map = {
    "cem": "CEMPlanner",
    "mppi": "MPPIPlanner",
    "neural_stochastic": "NeuralStochasticPlanner",
}
objective_name_map = {
    "repr_dist": "ReprTargetDistMPCObjective",
}


def import_from_target(target: str):
    """Import a class/function from a dotted target path."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def import_from_planning_export(attr_name: str):
    """Resolve exported planning symbols at runtime for patch/mock compatibility."""
    planning_module = import_module("eb_jepa.planning")
    return getattr(planning_module, attr_name)
