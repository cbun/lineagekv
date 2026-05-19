#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import write_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "flight_origin",
        "split": "dev",
        "category": "travel",
        "memory_key": "travel.flight.default_origin",
        "task": "prepare the departure airport for a flight booking",
        "answer_hint": "airport code",
        "stale": "SFO",
        "current": "AUS",
        "stale_text": "Memory key: travel.flight.default_origin. Earlier travel notes said to start flight searches from SFO because that was the usual airport.",
        "current_text": "Memory key: travel.flight.default_origin. After the move, new flight bookings should use AUS as the departure airport.",
    },
    {
        "id": "invoice_destination",
        "split": "dev",
        "category": "billing",
        "memory_key": "finance.invoice.destination",
        "task": "address the next invoice email",
        "answer_hint": "email address",
        "stale": "billing@oldco.example",
        "current": "ap@newco.example",
        "stale_text": "Memory key: finance.invoice.destination. The old accounting workflow routed invoices to billing@oldco.example.",
        "current_text": "Memory key: finance.invoice.destination. The new accounts-payable contact for invoices is ap@newco.example.",
    },
    {
        "id": "incident_room",
        "split": "dev",
        "category": "operations",
        "memory_key": "ops.incident.alert_room",
        "task": "post a production incident alert",
        "answer_hint": "chat channel",
        "stale": "#alerts-old",
        "current": "#incident-ops",
        "stale_text": "Memory key: ops.incident.alert_room. Runbooks from January still mention #alerts-old for incident broadcasts.",
        "current_text": "Memory key: ops.incident.alert_room. Current incident broadcasts go to #incident-ops so the on-call group sees them.",
    },
    {
        "id": "deploy_source",
        "split": "dev",
        "category": "engineering",
        "memory_key": "repo.release.deploy_branch",
        "task": "choose the branch for a release deployment",
        "answer_hint": "branch name",
        "stale": "master",
        "current": "main",
        "stale_text": "Memory key: repo.release.deploy_branch. An old release checklist says production deploys are cut from master.",
        "current_text": "Memory key: repo.release.deploy_branch. The repository renamed the release branch; deploy production from main.",
    },
    {
        "id": "support_plan",
        "split": "dev",
        "category": "support",
        "memory_key": "customer.acme.support_plan",
        "task": "label Acme's support plan",
        "answer_hint": "plan name",
        "stale": "Basic",
        "current": "Pro",
        "stale_text": "Memory key: customer.acme.support_plan. At signup, Acme was entered as a Basic support customer.",
        "current_text": "Memory key: customer.acme.support_plan. Acme upgraded last week; use Pro for their support plan.",
    },
    {
        "id": "doc_review_format",
        "split": "dev",
        "category": "writing",
        "memory_key": "writing.review.delivery_format",
        "task": "send a draft review",
        "answer_hint": "delivery format",
        "stale": "PDF",
        "current": "Google Doc",
        "stale_text": "Memory key: writing.review.delivery_format. Older review handoffs were exported as PDF attachments.",
        "current_text": "Memory key: writing.review.delivery_format. Reviews should now be shared as Google Doc links so comments stay live.",
    },
    {
        "id": "tool_budget",
        "split": "dev",
        "category": "finance",
        "memory_key": "finance.tools.monthly_cap",
        "task": "check the monthly tooling budget cap",
        "answer_hint": "dollar amount",
        "stale": "$500",
        "current": "$750",
        "stale_text": "Memory key: finance.tools.monthly_cap. The first budget note capped tooling at $500 per month.",
        "current_text": "Memory key: finance.tools.monthly_cap. The approved monthly tooling cap is now $750.",
    },
    {
        "id": "crm_phase",
        "split": "dev",
        "category": "sales",
        "memory_key": "sales.acme.crm_phase",
        "task": "update Acme's CRM phase",
        "answer_hint": "CRM phase",
        "stale": "Lead",
        "current": "Trial",
        "stale_text": "Memory key: sales.acme.crm_phase. Acme was originally tracked as a Lead.",
        "current_text": "Memory key: sales.acme.crm_phase. Acme entered the product trial; the current CRM phase is Trial.",
    },
    {
        "id": "meeting_place",
        "split": "test",
        "category": "calendar",
        "memory_key": "calendar.weekly_review.location",
        "task": "schedule the weekly review",
        "answer_hint": "meeting location",
        "stale": "Room 3B",
        "current": "Zoom",
        "stale_text": "Memory key: calendar.weekly_review.location. The standing review used to happen in Room 3B.",
        "current_text": "Memory key: calendar.weekly_review.location. The weekly review moved remote; use Zoom for the meeting location.",
    },
    {
        "id": "search_route",
        "split": "test",
        "category": "engineering",
        "memory_key": "api.search.route",
        "task": "configure the search request",
        "answer_hint": "API route",
        "stale": "/v1/search",
        "current": "/v2/query",
        "stale_text": "Memory key: api.search.route. Legacy integrations sent search traffic to /v1/search.",
        "current_text": "Memory key: api.search.route. New search requests should be sent to /v2/query.",
    },
    {
        "id": "shipping_drop",
        "split": "test",
        "category": "profile",
        "memory_key": "profile.shipping.default_address",
        "task": "prepare a shipping label",
        "answer_hint": "street address",
        "stale": "12 Market St",
        "current": "88 Lake Ave",
        "stale_text": "Memory key: profile.shipping.default_address. Packages previously went to 12 Market St.",
        "current_text": "Memory key: profile.shipping.default_address. The current default shipping address is 88 Lake Ave.",
    },
    {
        "id": "reminder_cadence",
        "split": "test",
        "category": "calendar",
        "memory_key": "calendar.weekly_reminder.day",
        "task": "schedule the weekly reminder",
        "answer_hint": "weekday",
        "stale": "Friday",
        "current": "Monday",
        "stale_text": "Memory key: calendar.weekly_reminder.day. Weekly reminders were originally queued for Friday.",
        "current_text": "Memory key: calendar.weekly_reminder.day. The weekly reminder should now fire on Monday.",
    },
    {
        "id": "editor_choice",
        "split": "test",
        "category": "profile",
        "memory_key": "profile.coding.preferred_editor",
        "task": "open the user's preferred editor",
        "answer_hint": "editor name",
        "stale": "VS Code",
        "current": "Zed",
        "stale_text": "Memory key: profile.coding.preferred_editor. Earlier setup notes said VS Code was the default editor.",
        "current_text": "Memory key: profile.coding.preferred_editor. The preferred editor is now Zed.",
    },
    {
        "id": "analytics_dashboard",
        "split": "test",
        "category": "analytics",
        "memory_key": "analytics.dashboard.tool",
        "task": "prepare a dashboard link",
        "answer_hint": "analytics tool",
        "stale": "Looker",
        "current": "Metabase",
        "stale_text": "Memory key: analytics.dashboard.tool. Dashboard reports were once built in Looker.",
        "current_text": "Memory key: analytics.dashboard.tool. Current dashboard reports should be built in Metabase.",
    },
    {
        "id": "client_tone",
        "split": "test",
        "category": "writing",
        "memory_key": "writing.client_update.tone",
        "task": "draft a client update",
        "answer_hint": "tone",
        "stale": "formal",
        "current": "direct and concise",
        "stale_text": "Memory key: writing.client_update.tone. Older client updates were written in a formal tone.",
        "current_text": "Memory key: writing.client_update.tone. Client updates should now be direct and concise.",
    },
    {
        "id": "checkout_flag",
        "split": "test",
        "category": "engineering",
        "memory_key": "release.checkout.feature_flag",
        "task": "choose the checkout feature flag",
        "answer_hint": "feature flag",
        "stale": "legacy_checkout",
        "current": "checkout_v2",
        "stale_text": "Memory key: release.checkout.feature_flag. The legacy checkout path used legacy_checkout.",
        "current_text": "Memory key: release.checkout.feature_flag. Checkout testing should use checkout_v2.",
    },
]


DISTRACTORS = [
    {
        "id": "timezone",
        "memory_key": "profile.timezone",
        "text": "Memory key: profile.timezone. Scheduling should use Central Time.",
    },
    {
        "id": "voice",
        "memory_key": "assistant.voice.default",
        "text": "Memory key: assistant.voice.default. Voice replies should stay calm and concise.",
    },
    {
        "id": "package_manager",
        "memory_key": "repo.install.package_manager",
        "text": "Memory key: repo.install.package_manager. Use pnpm for dependency installation.",
    },
    {
        "id": "oncall_tool",
        "memory_key": "ops.escalation.tool",
        "text": "Memory key: ops.escalation.tool. Urgent incidents escalate through OpsGenie Primary.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate natural multi-memory action cases.")
    parser.add_argument("--output", default="data/generated/agent_memory_action_probe64.jsonl")
    parser.add_argument("--dev-output", default="data/generated/agent_memory_action_dev32.jsonl")
    parser.add_argument("--test-output", default="data/generated/agent_memory_action_test32.jsonl")
    args = parser.parse_args()

    items = all_items()
    count = write_jsonl(items, ROOT / args.output)
    dev_count = write_jsonl([item for item in items if item.metadata["split"] == "dev"], ROOT / args.dev_output)
    test_count = write_jsonl([item for item in items if item.metadata["split"] == "test"], ROOT / args.test_output)
    print(f"wrote {count} action-memory items to {ROOT / args.output}")
    print(f"wrote {dev_count} dev items to {ROOT / args.dev_output}")
    print(f"wrote {test_count} test items to {ROOT / args.test_output}")


def all_items() -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for scenario in SCENARIOS:
        for order in ["current_first", "stale_first"]:
            for query_style in ["direct_value", "task_action"]:
                items.append(action_item(scenario, order=order, query_style=query_style))
    return items


def action_item(scenario: dict[str, Any], order: str, query_style: str) -> BenchmarkItem:
    current_id = f"{scenario['id']}_current"
    stale_id = f"{scenario['id']}_stale"
    current_block = ContextBlock(
        id=current_id,
        timestamp="2024-03-15T09:00:00+00:00",
        type="agent_memory",
        trust="high",
        status="unknown",
        text=scenario["current_text"],
        metadata={
            "category": scenario["category"],
            "memory_key": scenario["memory_key"],
            "value": scenario["current"],
            "supersedes": [stale_id],
        },
    )
    stale_block = ContextBlock(
        id=stale_id,
        timestamp="2024-01-12T09:00:00+00:00",
        type="agent_memory",
        trust="high",
        status="unknown",
        text=scenario["stale_text"],
        metadata={
            "category": scenario["category"],
            "memory_key": scenario["memory_key"],
            "value": scenario["stale"],
            "superseded_by": [current_id],
        },
    )
    target_blocks = [current_block, stale_block] if order == "current_first" else [stale_block, current_block]
    context_blocks = interleave_distractors(target_blocks, scenario["id"])
    query = query_for(scenario, query_style)
    return BenchmarkItem(
        id=f"agent_memory_action_{scenario['id']}_{order}_{query_style}",
        domain="agent_memory_action",
        question_type="agent_memory_action_value",
        context_blocks=context_blocks,
        query=query,
        gold_answer=scenario["current"],
        gold_evidence_ids=[current_id],
        stale_ids=[stale_id],
        distractor_ids=[block.id for block in context_blocks if block.id not in {current_id, stale_id}],
        metadata={
            "category": scenario["category"],
            "memory_key": scenario["memory_key"],
            "split": scenario["split"],
            "variant": order,
            "query_style": query_style,
            "distractor_ratio": 0.5,
            "target_value": scenario["current"],
            "stale_value": scenario["stale"],
        },
    )


def interleave_distractors(target_blocks: list[ContextBlock], scenario_id: str) -> list[ContextBlock]:
    first, second = distractors_for(scenario_id)
    return [first, target_blocks[0], second, target_blocks[1]]


def distractors_for(scenario_id: str) -> tuple[ContextBlock, ContextBlock]:
    digest = int(hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()[:8], 16)
    first = DISTRACTORS[digest % len(DISTRACTORS)]
    second = DISTRACTORS[(digest // len(DISTRACTORS) + 1) % len(DISTRACTORS)]
    if second["id"] == first["id"]:
        second = DISTRACTORS[(digest + 2) % len(DISTRACTORS)]
    return distractor_block(first, "a"), distractor_block(second, "b")


def distractor_block(spec: dict[str, str], suffix: str) -> ContextBlock:
    return ContextBlock(
        id=f"distractor_{spec['id']}_{suffix}",
        timestamp="2024-02-20T09:00:00+00:00",
        type="agent_memory",
        trust="medium",
        status="unknown",
        text=spec["text"],
        metadata={"memory_key": spec["memory_key"]},
    )


def query_for(scenario: dict[str, Any], query_style: str) -> str:
    if query_style == "direct_value":
        return (
            f"For memory key '{scenario['memory_key']}', what current {scenario['answer_hint']} "
            f"should be used? Answer with only the value."
        )
    if query_style == "task_action":
        return (
            f"Use the current memory for key '{scenario['memory_key']}' to {scenario['task']}. "
            f"What {scenario['answer_hint']} should the assistant use? Answer with only the value."
        )
    raise KeyError(query_style)


if __name__ == "__main__":
    main()
