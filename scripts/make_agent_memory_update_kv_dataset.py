#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import write_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock


SCENARIOS = [
    {
        "id": "airport",
        "category": "travel",
        "slot": "default departure airport",
        "stale": "SFO",
        "current": "AUS",
        "stale_text": "User memory: default departure airport is SFO for future trips.",
        "current_text": "User memory: default departure airport is AUS for future trips.",
    },
    {
        "id": "timezone",
        "category": "profile",
        "slot": "timezone",
        "stale": "Pacific Time",
        "current": "Central Time",
        "stale_text": "User memory: timezone is Pacific Time for scheduling.",
        "current_text": "User memory: timezone is Central Time for scheduling.",
    },
    {
        "id": "invoice_email",
        "category": "billing",
        "slot": "invoice email",
        "stale": "billing@oldco.example",
        "current": "ap@newco.example",
        "stale_text": "Account memory: send invoices to billing@oldco.example.",
        "current_text": "Account memory: send invoices to ap@newco.example.",
    },
    {
        "id": "standup_time",
        "category": "calendar",
        "slot": "daily standup time",
        "stale": "9:00 AM",
        "current": "10:30 AM",
        "stale_text": "Team memory: daily standup time is 9:00 AM.",
        "current_text": "Team memory: daily standup time is 10:30 AM.",
    },
    {
        "id": "deploy_branch",
        "category": "engineering",
        "slot": "deployment branch",
        "stale": "master",
        "current": "main",
        "stale_text": "Project memory: deploy releases from the master branch.",
        "current_text": "Project memory: deploy releases from the main branch.",
    },
    {
        "id": "package_manager",
        "category": "engineering",
        "slot": "package manager",
        "stale": "npm",
        "current": "pnpm",
        "stale_text": "Repository memory: use npm for dependency installation.",
        "current_text": "Repository memory: use pnpm for dependency installation.",
    },
    {
        "id": "api_endpoint",
        "category": "engineering",
        "slot": "search API endpoint",
        "stale": "/v1/search",
        "current": "/v2/query",
        "stale_text": "API memory: call /v1/search for search requests.",
        "current_text": "API memory: call /v2/query for search requests.",
    },
    {
        "id": "alert_channel",
        "category": "operations",
        "slot": "incident alert channel",
        "stale": "#alerts-old",
        "current": "#incident-ops",
        "stale_text": "Operations memory: post incident alerts in #alerts-old.",
        "current_text": "Operations memory: post incident alerts in #incident-ops.",
    },
    {
        "id": "meeting_location",
        "category": "calendar",
        "slot": "weekly review location",
        "stale": "Room 3B",
        "current": "Zoom",
        "stale_text": "Calendar memory: weekly review location is Room 3B.",
        "current_text": "Calendar memory: weekly review location is Zoom.",
    },
    {
        "id": "oncall_system",
        "category": "operations",
        "slot": "on-call escalation system",
        "stale": "PagerDuty Team A",
        "current": "OpsGenie Primary",
        "stale_text": "Operations memory: escalate urgent incidents to PagerDuty Team A.",
        "current_text": "Operations memory: escalate urgent incidents to OpsGenie Primary.",
    },
    {
        "id": "budget_cap",
        "category": "finance",
        "slot": "monthly tooling budget cap",
        "stale": "$500",
        "current": "$750",
        "stale_text": "Finance memory: monthly tooling budget cap is $500.",
        "current_text": "Finance memory: monthly tooling budget cap is $750.",
    },
    {
        "id": "document_format",
        "category": "writing",
        "slot": "draft review format",
        "stale": "PDF",
        "current": "Google Doc",
        "stale_text": "Writing memory: send draft reviews as PDF files.",
        "current_text": "Writing memory: send draft reviews as Google Doc links.",
    },
    {
        "id": "shipping_address",
        "category": "profile",
        "slot": "shipping address",
        "stale": "12 Market St",
        "current": "88 Lake Ave",
        "stale_text": "User memory: shipping address is 12 Market St.",
        "current_text": "User memory: shipping address is 88 Lake Ave.",
    },
    {
        "id": "crm_stage",
        "category": "sales",
        "slot": "Acme CRM stage",
        "stale": "Lead",
        "current": "Trial",
        "stale_text": "CRM memory: Acme account stage is Lead.",
        "current_text": "CRM memory: Acme account stage is Trial.",
    },
    {
        "id": "feature_flag",
        "category": "engineering",
        "slot": "checkout feature flag",
        "stale": "legacy_checkout",
        "current": "checkout_v2",
        "stale_text": "Release memory: use feature flag legacy_checkout for checkout tests.",
        "current_text": "Release memory: use feature flag checkout_v2 for checkout tests.",
    },
    {
        "id": "support_tier",
        "category": "support",
        "slot": "customer support tier",
        "stale": "Basic",
        "current": "Pro",
        "stale_text": "Support memory: customer support tier is Basic.",
        "current_text": "Support memory: customer support tier is Pro.",
    },
    {
        "id": "reminder_day",
        "category": "calendar",
        "slot": "weekly reminder day",
        "stale": "Friday",
        "current": "Monday",
        "stale_text": "User memory: weekly reminder day is Friday.",
        "current_text": "User memory: weekly reminder day is Monday.",
    },
    {
        "id": "code_editor",
        "category": "profile",
        "slot": "preferred code editor",
        "stale": "VS Code",
        "current": "Zed",
        "stale_text": "User memory: preferred code editor is VS Code.",
        "current_text": "User memory: preferred code editor is Zed.",
    },
    {
        "id": "analytics_tool",
        "category": "analytics",
        "slot": "analytics dashboard tool",
        "stale": "Looker",
        "current": "Metabase",
        "stale_text": "Analytics memory: dashboard reports should use Looker.",
        "current_text": "Analytics memory: dashboard reports should use Metabase.",
    },
    {
        "id": "voice_style",
        "category": "writing",
        "slot": "client update tone",
        "stale": "formal",
        "current": "direct and concise",
        "stale_text": "Writing memory: client updates should use a formal tone.",
        "current_text": "Writing memory: client updates should be direct and concise.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic agent-memory update nonce-option cases.")
    parser.add_argument("--output", default="data/generated/agent_memory_update_nonce_option_probe80.jsonl")
    parser.add_argument("--max-items", type=int, default=80)
    args = parser.parse_args()

    items: list[BenchmarkItem] = []
    for scenario in SCENARIOS:
        for order in ["current_first", "stale_first"]:
            for assignment in ["code_a_current", "code_b_current"]:
                items.append(agent_memory_item(scenario, order=order, assignment=assignment))
                if len(items) >= args.max_items:
                    break
            if len(items) >= args.max_items:
                break
        if len(items) >= args.max_items:
            break

    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} agent-memory nonce-option items to {ROOT / args.output}")


def agent_memory_item(scenario: dict[str, str], order: str, assignment: str) -> BenchmarkItem:
    base_id = f"agent_memory_{scenario['id']}_{order}_{assignment}"
    code_a, code_b = nonce_codes(f"agent_memory_{scenario['id']}")
    current_value = scenario["current"]
    stale_value = scenario["stale"]
    if assignment == "code_a_current":
        options = {code_a: current_value, code_b: stale_value}
        current_code = code_a
        stale_code = code_b
    elif assignment == "code_b_current":
        options = {code_a: stale_value, code_b: current_value}
        current_code = code_b
        stale_code = code_a
    else:
        raise KeyError(assignment)

    current_block = ContextBlock(
        id=f"{scenario['id']}_current",
        timestamp="2024-02-15T09:00:00+00:00",
        type="agent_memory",
        trust="high",
        status="unknown",
        text=scenario["current_text"],
        metadata={
            "category": scenario["category"],
            "slot": scenario["slot"],
            "value": current_value,
            "supersedes": [f"{scenario['id']}_stale"],
        },
    )
    stale_block = ContextBlock(
        id=f"{scenario['id']}_stale",
        timestamp="2024-01-10T09:00:00+00:00",
        type="agent_memory",
        trust="high",
        status="unknown",
        text=scenario["stale_text"],
        metadata={
            "category": scenario["category"],
            "slot": scenario["slot"],
            "value": stale_value,
            "superseded_by": [f"{scenario['id']}_current"],
        },
    )
    blocks = [current_block, stale_block] if order == "current_first" else [stale_block, current_block]
    option_text = " ".join(f"{code} = {value!r}." for code, value in options.items())
    query = (
        f"Which option code is the current memory value for the user's {scenario['slot']}? "
        f"{option_text} Answer with only the option code."
    )
    return BenchmarkItem(
        id=base_id,
        domain="agent_memory",
        question_type="agent_memory_nonce_option",
        context_blocks=blocks,
        query=query,
        gold_answer=current_code,
        gold_evidence_ids=[current_block.id],
        stale_ids=[stale_block.id],
        metadata={
            "category": scenario["category"],
            "slot": scenario["slot"],
            "variant": order,
            "distractor_ratio": 0.0,
            "nonce_option": True,
            "nonce_assignment": assignment,
            "option_labels": dict(options),
            "target_value": current_code,
            "stale_value": stale_code,
            "target_raw_value": current_value,
            "stale_raw_value": stale_value,
        },
    )


def nonce_codes(seed: str) -> tuple[str, str]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    encoded = base64.b32encode(digest).decode("ascii").lower()
    return f"c{encoded[:5]}", f"c{encoded[5:10]}"


if __name__ == "__main__":
    main()
