from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


# --- benchq==0.7.0 imports (adaptation layer) ---

from benchq.quantum_hardware_modeling.hardware_architecture_models import SCModel
from benchq.magic_state_distillation.litinski_factories import iter_litinski_factories

# In benchq==0.7.0 the callable is footprint_estimator; alias it to openfermion_estimator for Max's code.
from benchq.resource_estimators.footprint_estimators.openfermion_estimator import (
    footprint_estimator as openfermion_estimator,
)


class NoFactoriesFoundError(RuntimeError):
    pass



def get_physical_cost(
    num_logical_qubits: int,
    num_T_gates: int,
    num_toffoli_gates: int,
    hardware_failure_tolerance_per_shot: float,
    n_factories: int,
    physical_error_rate: float,
    cycle_time_us: float,
) -> Any:
    assert num_T_gates == 0, "T gates are not supported."
    HARDWARE_ARCHITECTURE_MODEL = SCModel(
        physical_qubit_error_rate=physical_error_rate,
        surface_code_cycle_time_in_seconds=cycle_time_us * 1e-6,
    )
    try:
        return openfermion_estimator(
            num_logical_qubits=num_logical_qubits,
            num_toffoli=4 * num_toffoli_gates, # Note that the factor of four is a hacky workaround for https://github.com/zapatacomputing/benchq/issues/165.
            hardware_failure_tolerance=hardware_failure_tolerance_per_shot,
            architecture_model=HARDWARE_ARCHITECTURE_MODEL,
            magic_state_factory_iterator=iter_litinski_factories(
                HARDWARE_ARCHITECTURE_MODEL
            ),
            factory_count=n_factories,
        )
    except RuntimeError as e:
        if (
            "Failed to find parameters that yield an acceptable failure probability."
            in str(e)
        ):
            raise NoFactoriesFoundError
        else:
            raise e


def _extract_core_fields(info: Any) -> Dict[str, Any]:
    # ResourceInfo in benchq==0.7.0 stores useful fields directly on the object.
    # We only extract the core fields you care about for plotting/analysis.
    out: Dict[str, Any] = {}

    for k in [
        "n_physical_qubits",
        "code_distance",
        "total_time_in_seconds",
        "logical_error_rate",
        "magic_state_factory_name",
        "routing_to_measurement_volume_ratio",
    ]:
        if hasattr(info, k):
            out[k] = getattr(info, k)

    # Extra is often OpenFermionExtra; keep it if serializable-ish.
    if hasattr(info, "extra"):
        extra = getattr(info, "extra")
        if hasattr(extra, "__dict__"):
            out["extra"] = dict(extra.__dict__)
        else:
            out["extra"] = repr(extra)

    return out


def add_physical_estimates(
    entries: List[Dict[str, Any]],
    hardware_failure_tolerance_per_shot: float,
    n_factories: int,
    physical_error_rate: float,
    cycle_time_us: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    assumptions = {
        "hardware_failure_tolerance_per_shot": hardware_failure_tolerance_per_shot,
        "n_factories": n_factories,
        "physical_error_rate": physical_error_rate,
        "cycle_time_us": cycle_time_us,
        "notes": {
            "t_gates_not_supported": True,
            "benchq_issue_165_num_toffoli_multiplier": 4,
            "factory_family": "Litinski",
        },
    }

    for entry in entries:
        e = dict(entry)

        logical_qubits = int(e["logical_qubits"])
        toffoli_gates = int(e["toffoli_count"])

        payload: Dict[str, Any] = {"assumptions": assumptions}

        try:
            info = get_physical_cost(
                num_logical_qubits=logical_qubits,
                num_T_gates=0,
                num_toffoli_gates=toffoli_gates,
                hardware_failure_tolerance_per_shot=hardware_failure_tolerance_per_shot,
                n_factories=n_factories,
                physical_error_rate=physical_error_rate,
                cycle_time_us=cycle_time_us,
            )
            payload["status"] = "ok"
            payload["core"] = _extract_core_fields(info)

        except NoFactoriesFoundError as ex:
            payload["status"] = "no_factories_found"
            payload["message"] = str(ex)

        e["physical_cost_model"] = payload
        out.append(e)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add physical resource estimates to a benchq logical-results JSON (benchq==0.7.0)."
    )
    parser.add_argument("input_json", help="Path to input JSON (list of entries).")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. Default: <input>.with_physical.json",
    )

    # Defaults are placeholders; override to match your project's baseline assumptions.
    parser.add_argument("--hardware_failure_tolerance_per_shot", type=float, default=1e-3)
    parser.add_argument("--n_factories", type=int, default=4)
    parser.add_argument("--physical_error_rate", type=float, default=1e-3)
    parser.add_argument("--cycle_time_us", type=float, default=1.0)

    args = parser.parse_args()

    in_path = args.input_json
    out_path = args.output
    if out_path is None:
        out_path = in_path[:-5] + ".with_physical.json" if in_path.endswith(".json") else in_path + ".with_physical.json"

    with open(in_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of dict entries.")

    out_entries = add_physical_estimates(
        entries=data,
        hardware_failure_tolerance_per_shot=args.hardware_failure_tolerance_per_shot,
        n_factories=args.n_factories,
        physical_error_rate=args.physical_error_rate,
        cycle_time_us=args.cycle_time_us,
    )

    with open(out_path, "w") as f:
        json.dump(out_entries, f, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
