"""Локальный экспорт правил наблюдения из состоявшегося прогона GNet9."""

from ipaddress import ip_address
import json
from pathlib import Path
from uuid import NAMESPACE_OID, uuid5

from .attacks import AttackKind, MITRE_TENSOR_MAPPINGS


OBSERVED_CLASSIFICATION_SOURCES = {
    "precursor": "precursor_heuristic",
    "observed_signature": "observed_numeric_hypothesis",
}
SIMULATED_EVENT_SOURCE = "simulated_active_event"
USER_EXECUTION_FIXTURE = "GNET9_SYNTHETIC_USER_EXECUTION\ntechnique_id=T1204\n"


def export_detection_rules(model, snapshots, directory: Path) -> None:
    directory.mkdir(exist_ok=True)
    mappings = {item.technique_id: item for item in MITRE_TENSOR_MAPPINGS if item.technique_id}
    known_kinds = {kind.value for kind in AttackKind}
    observations = []
    sigma_selectors = set()
    syn_rates = {}
    has_user_execution_fixture = False
    for snapshot in snapshots:
        position = {key: snapshot[key] for key in ("step_index", "time_seconds")}
        koopman = snapshot["koopman"]
        signatures = koopman.get("signature_entity_records", ())
        forecasts = [record for record in koopman["forecast_entity_records"] if record not in signatures]
        for record in (*forecasts, *signatures):
            source = OBSERVED_CLASSIFICATION_SOURCES.get(record.get("observation_type"))
            if source is None:
                continue
            attack_kind = record.get("kind")
            technique_ids = [
                technique_id for technique_id in record.get("mitre_technique_candidates", [])
                if technique_id in mappings
            ]
            observation = position | {
                "target_id": record["target_id"],
                "attack_kind": attack_kind,
                "classification_source": source,
                **{key: record[key] for key in ("attack_family", "attack_mechanism", "attack_label_ru") if key in record},
            }
            if "feature_match" in record:
                observation["feature_match"] = record["feature_match"]
            for technique_id in technique_ids:
                observations.append(observation | {"technique_id": technique_id})
                sigma_selectors.add(("technique_id", technique_id))
            if not technique_ids and attack_kind in known_kinds:
                observations.append(observation | {"technique_id": None})
                sigma_selectors.add(("attack_kind", attack_kind))
        for event in snapshot["attacks"].get("events", []):
            if event.get("is_recovery_effect"):
                continue
            target = event["target_id"]
            technique_id = event.get("mitre_technique_id")
            attack_kind = event["kind"]
            observations.append(position | {
                "target_id": target, "technique_id": technique_id,
                "attack_kind": attack_kind, "attack_id": event.get("attack_id"),
                "classification_source": SIMULATED_EVENT_SOURCE,
                **{key: event[key] for key in ("attack_family", "attack_mechanism", "attack_label_ru") if key in event},
            })
            if attack_kind == AttackKind.USER_EXECUTION.value and technique_id == "T1204":
                has_user_execution_fixture = True
            rate = event.get("effective_packet_rate_pps", 0)
            if attack_kind == AttackKind.SYN_FLOOD.value and event.get("routes") and rate > 0:
                address = model.graph.nodes[target].get("ip_address")
                if address:
                    destination = str(ip_address(address))
                    syn_rates[destination] = min(rate, syn_rates.get(destination, rate))

    sigma = "\n---\n".join(
        _sigma_rule(field, value, mappings[value].mitre_url if field == "technique_id" else None)
        for field, value in sorted(sigma_selectors)
    )
    suricata = "".join(
        _suricata_syn_rule(destination, rate, sid=9_000_001 + index)
        for index, (destination, rate) in enumerate(sorted(syn_rates.items()))
    )
    _write_optional(directory / "gnet9.yml", sigma)
    _write_optional(directory / "gnet9.rules", suricata)
    _write_optional(
        directory / "observations.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in observations),
    )
    yara_file = directory / "gnet9.yar"
    yara_sample = directory / "user_execution_fixture.txt"
    _write_optional(yara_file, _user_execution_yara_rule() if has_user_execution_fixture else "")
    _write_optional(yara_sample, USER_EXECUTION_FIXTURE if has_user_execution_fixture else "")
    manifest = {
        "scope": "simulation_review_only",
        "sigma": {
            "rule_count": len(sigma_selectors), "file": "gnet9.yml" if sigma else None,
            "input": "observations.jsonl" if observations else None,
            "semantics": "matches_gnet9_observed_hypotheses",
            "required_fields": ["classification_source", *sorted({field for field, _ in sigma_selectors})],
            "observed_classification_sources": list(OBSERVED_CLASSIFICATION_SOURCES.values()),
            "simulation_truth_classification_source": SIMULATED_EVENT_SOURCE,
        },
        "suricata": {
            "rule_count": len(syn_rates), "file": "gnet9.rules" if suricata else None,
            "semantics": "alert_only_at_simulated_syn_rates",
            "syn_rate_thresholds_packets_per_second": syn_rates,
        },
        "yara": {
            "rule_count": int(has_user_execution_fixture),
            "file": yara_file.name if has_user_execution_fixture else None,
            "sample": yara_sample.name if has_user_execution_fixture else None,
            "semantics": "benign_simulator_fixture_only",
            "reason": None if has_user_execution_fixture else "no_synthetic_user_execution_fixture",
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_optional(path: Path, content: str) -> None:
    if content:
        path.write_text(content, encoding="utf-8", newline="\n")
    else:
        path.unlink(missing_ok=True)


def _sigma_rule(field: str, value: str, mitre_url: str | None) -> str:
    rule_id = uuid5(NAMESPACE_OID, f"gnet9.sigma.{field}.{value}")
    reference = f"references:\n    - '{mitre_url}'\n" if mitre_url else ""
    tags = f"tags:\n    - 'attack.{value.lower()}'\n" if mitre_url else ""
    sources = "".join(f"            - '{source}'\n" for source in OBSERVED_CLASSIFICATION_SOURCES.values())
    return (
        f"title: 'GNet9 Observation {value}'\n"
        f"id: '{rule_id}'\n"
        "status: 'experimental'\n"
        "description: 'Отбирает наблюдаемые гипотезы локального стенда GNet9.'\n"
        f"{reference}"
        "logsource:\n"
        "    product: 'gnet9'\n"
        "    service: 'simulation'\n"
        "    definition: 'Импортировать observations.jsonl как записи JSON.'\n"
        "detection:\n"
        "    selection:\n"
        f"        {field}: '{value}'\n"
        "        classification_source:\n"
        f"{sources}"
        "    condition: 'selection'\n"
        "falsepositives:\n"
        "    - 'Допустимые отклонения с теми же признаками.'\n"
        "level: 'informational'\n"
        f"{tags}"
    )


def _user_execution_yara_rule() -> str:
    literal = json.dumps(USER_EXECUTION_FIXTURE)
    size = len(USER_EXECUTION_FIXTURE.encode("ascii"))
    return (
        "rule gnet9_synthetic_user_execution {\n"
        "    meta:\n"
        '        description = "Matches the benign GNet9 simulation fixture"\n'
        '        technique_id = "T1204"\n'
        '        scope = "gnet9_synthetic_fixture"\n'
        "    strings:\n"
        f"        $fixture = {literal}\n"
        "    condition:\n"
        f"        $fixture at 0 and filesize == {size}\n"
        "}\n"
    )


def _suricata_syn_rule(destination: str, rate: int, *, sid: int) -> str:
    # Один отсчёт порога соответствует средней интенсивности за секунду модели.
    return (
        f'alert tcp any any -> {destination} any (msg:"GNet9 simulated SYN rate"; '
        f"tcp.flags:S,CE; threshold:type both,track by_dst,count {rate},seconds 1; "
        f"sid:{sid}; rev:1;)\n"
    )
