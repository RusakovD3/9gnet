"""Explicit scope audit for protocol standards and equipment characteristics."""

from __future__ import annotations

from typing import Any

from .packet_simulator import TRAFFIC_APPS


def build_standards_and_equipment_audit(model) -> dict[str, Any]:
    """Report what the simulator models, verifies, assumes and omits.

    This is a coverage audit, deliberately not an RFC certification or a claim
    that scenario envelopes for RAN/OLT are vendor specifications.
    """
    rfc_coverage = [
        {
            "standard": "RFC 791 IPv4",
            "url": "https://www.rfc-editor.org/rfc/rfc791.html",
            "status": "modeled_header_fields",
            "evidence": "IPv4 protocol number, total length, identification and TTL are represented in memory.",
            "not_implemented": ["wire checksum", "fragment reassembly", "host-stack interoperability"],
        },
        {
            "standard": "RFC 9293 TCP",
            "url": "https://www.rfc-editor.org/info/rfc9293/",
            "status": "modeled_header_and_syn_pressure",
            "evidence": "TCP header sizing, MSS accounting and SYN-flood state pressure are modeled.",
            "not_implemented": ["complete TCP state machine", "retransmission", "wire interoperability"],
        },
        {
            "standard": "RFC 768 UDP",
            "url": "https://www.rfc-editor.org/info/rfc768/",
            "status": "modeled_header_fields",
            "evidence": "UDP header size, protocol number and payload accounting are modeled.",
            "not_implemented": ["wire checksum", "host-stack interoperability"],
        },
        {
            "standard": "RFC 3550 RTP",
            "url": "https://www.rfc-editor.org/rfc/rfc3550.html",
            "status": "modeled_media_metadata",
            "evidence": "RTP header size, payload type and clock-rate metadata are carried by media flows.",
            "not_implemented": ["real RTP timing", "jitter buffer", "network transmission"],
        },
    ]
    applications = [
        {
            "traffic_kind": kind,
            "application": profile["application"],
            "transport": profile["transport"],
            "service_node": profile["service_node"],
            "model_scope": "in_memory_packet_and_flow_model",
        }
        for kind, profile in sorted(TRAFFIC_APPS.items())
    ]
    equipment = []
    for node_id, attrs in sorted(model.graph.nodes(data=True)):
        if attrs.get("level") != "L2":
            continue
        profile = attrs.get("l2_profile", {})
        if not isinstance(profile, dict):
            profile = {}
        vendor = str(profile.get("vendor", attrs.get("platform_family", "")))
        is_vendor_verified = vendor not in {"", "GNet9 scenario"}
        equipment.append(
            {
                "node_id": node_id,
                "role": attrs.get("role"),
                "profile": profile.get("name", attrs.get("platform_profile")),
                "vendor": vendor,
                "characteristics_status": (
                    "vendor_published_fields_plus_explicit_model_assumptions"
                    if is_vendor_verified
                    else "gnet9_scenario_assumptions_not_vendor_specification"
                ),
                "source_url": profile.get("source_url"),
                "verified_fields": profile.get("verified_fields", []),
                "assumed_fields": profile.get("assumed_fields", []),
                "power_semantics": profile.get("power_value_semantics"),
            }
        )
    return {
        "scope": "documentation_and_model_coverage_audit_not_conformance_certification",
        "rfc_coverage": rfc_coverage,
        "application_profiles": applications,
        "equipment_characteristics": equipment,
        "conclusion_ru": (
            "Модель согласует выбранные заголовки, размеры и профили с указанными стандартами, "
            "но не реализует полный сетевой стек и не является RFC-сертифицированным продуктом. "
            "Cisco и Dell поля отделены от инженерных допущений; RAN и OLT пока остаются сценарными профилями."
        ),
    }
