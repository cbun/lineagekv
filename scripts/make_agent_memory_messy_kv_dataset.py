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
        "id": "review_location",
        "split": "dev",
        "category": "calendar",
        "memory_key": "calendar.review.location",
        "task": "schedule the weekly review",
        "answer_hint": "location",
        "older": "Room 3B",
        "middle": "Huddle Room A",
        "current": "Zoom",
        "older_text": "Memory slot: calendar.review.location. Back when everyone was in-office, the weekly review was penciled into Room 3B.",
        "middle_text": "memory_key=calendar.review.location; During the office remodel, the review moved to Huddle Room A for a few weeks.",
        "current_text": "Memory key: calendar.review.location. The current plan is remote; use Zoom for the weekly review.",
    },
    {
        "id": "invoice_contact",
        "split": "dev",
        "category": "billing",
        "memory_key": "finance.invoice.contact",
        "task": "send the invoice",
        "answer_hint": "email address",
        "older": "billing@oldco.example",
        "middle": "finance@newco.example",
        "current": "ap@newco.example",
        "older_text": "memory_key=finance.invoice.contact; The legacy vendor profile listed billing@oldco.example.",
        "middle_text": "Memory slot: finance.invoice.contact. A migration note temporarily routed invoices to finance@newco.example.",
        "current_text": "Memory key: finance.invoice.contact. The approved accounts-payable inbox is ap@newco.example.",
    },
    {
        "id": "deploy_branch",
        "split": "dev",
        "category": "engineering",
        "memory_key": "repo.deploy.branch",
        "task": "choose a release branch",
        "answer_hint": "branch",
        "older": "master",
        "middle": "release",
        "current": "main",
        "older_text": "Memory key: repo.deploy.branch. Old deployment notes still mention master as the production branch.",
        "middle_text": "memory_key=repo.deploy.branch; The release-candidate experiment briefly used release for production deploys.",
        "current_text": "Memory slot: repo.deploy.branch. Production deploys now come from main.",
    },
    {
        "id": "support_tier",
        "split": "dev",
        "category": "support",
        "memory_key": "customer.acme.support_tier",
        "task": "label Acme's support tier",
        "answer_hint": "tier",
        "older": "Basic",
        "middle": "Trial",
        "current": "Enterprise",
        "older_text": "Memory slot: customer.acme.support_tier. Acme started on the Basic tier.",
        "middle_text": "Memory key: customer.acme.support_tier. During onboarding, Acme was treated as Trial.",
        "current_text": "memory_key=customer.acme.support_tier; Contract signed: Acme is now Enterprise.",
    },
    {
        "id": "writing_tone",
        "split": "dev",
        "category": "writing",
        "memory_key": "writing.client_update.tone",
        "task": "draft a client update",
        "answer_hint": "tone",
        "older": "formal",
        "middle": "friendly",
        "current": "direct and concise",
        "older_text": "Memory key: writing.client_update.tone. Early client updates were supposed to sound formal.",
        "middle_text": "Memory slot: writing.client_update.tone. A later note asked for a friendly tone.",
        "current_text": "memory_key=writing.client_update.tone; Current preference: keep client updates direct and concise.",
    },
    {
        "id": "dashboard_tool",
        "split": "test",
        "category": "analytics",
        "memory_key": "analytics.dashboard.tool",
        "task": "prepare a dashboard link",
        "answer_hint": "tool",
        "older": "Looker",
        "middle": "Mode",
        "current": "Metabase",
        "older_text": "Memory slot: analytics.dashboard.tool. The original reports lived in Looker.",
        "middle_text": "memory_key=analytics.dashboard.tool; The data team piloted Mode for dashboards during Q1.",
        "current_text": "Memory key: analytics.dashboard.tool. Dashboard links should now point to Metabase.",
    },
    {
        "id": "shipping_address",
        "split": "test",
        "category": "profile",
        "memory_key": "profile.shipping.address",
        "task": "prepare a shipping label",
        "answer_hint": "address",
        "older": "12 Market St",
        "middle": "44 Pine Rd",
        "current": "88 Lake Ave",
        "older_text": "memory_key=profile.shipping.address; Packages used to go to 12 Market St.",
        "middle_text": "Memory key: profile.shipping.address. A temporary forwarding address was 44 Pine Rd.",
        "current_text": "Memory slot: profile.shipping.address. The current default shipping address is 88 Lake Ave.",
    },
    {
        "id": "feature_flag",
        "split": "test",
        "category": "engineering",
        "memory_key": "release.checkout.flag",
        "task": "choose the checkout feature flag",
        "answer_hint": "feature flag",
        "older": "legacy_checkout",
        "middle": "checkout_beta",
        "current": "checkout_v2",
        "older_text": "Memory key: release.checkout.flag. The old checkout path used legacy_checkout.",
        "middle_text": "Memory slot: release.checkout.flag. Beta testing briefly used checkout_beta.",
        "current_text": "memory_key=release.checkout.flag; Current checkout testing should use checkout_v2.",
    },
    {
        "id": "editor_preference",
        "split": "test",
        "category": "profile",
        "memory_key": "profile.editor.default",
        "task": "open the preferred editor",
        "answer_hint": "editor",
        "older": "VS Code",
        "middle": "Vim",
        "current": "Zed",
        "older_text": "Memory slot: profile.editor.default. Setup notes first listed VS Code.",
        "middle_text": "memory_key=profile.editor.default; A keyboard-only experiment switched the default to Vim.",
        "current_text": "Memory key: profile.editor.default. The preferred editor is now Zed.",
    },
    {
        "id": "reminder_day",
        "split": "test",
        "category": "calendar",
        "memory_key": "calendar.weekly_reminder.day",
        "task": "schedule the weekly reminder",
        "answer_hint": "weekday",
        "older": "Friday",
        "middle": "Wednesday",
        "current": "Monday",
        "older_text": "Memory key: calendar.weekly_reminder.day. Weekly reminders were once queued for Friday.",
        "middle_text": "Memory slot: calendar.weekly_reminder.day. A trial cadence moved reminders to Wednesday.",
        "current_text": "memory_key=calendar.weekly_reminder.day; The weekly reminder should now fire on Monday.",
    },
]


DISTRACTORS = [
    {
        "id": "timezone",
        "memory_key": "profile.timezone",
        "text": "Memory key: profile.timezone. Scheduling should use Central Time.",
    },
    {
        "id": "package_manager",
        "memory_key": "repo.install.package_manager",
        "text": "memory_key=repo.install.package_manager; Use pnpm for dependency installation.",
    },
    {
        "id": "escalation",
        "memory_key": "ops.escalation.primary",
        "text": "Memory slot: ops.escalation.primary. Urgent incidents escalate through OpsGenie Primary.",
    },
    {
        "id": "voice",
        "memory_key": "assistant.voice.default",
        "text": "Memory key: assistant.voice.default. Keep voice replies calm and concise.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noisy multi-update agent-memory action cases.")
    parser.add_argument("--output", default="data/generated/agent_memory_messy_probe40.jsonl")
    parser.add_argument("--dev-output", default="data/generated/agent_memory_messy_dev20.jsonl")
    parser.add_argument("--test-output", default="data/generated/agent_memory_messy_test20.jsonl")
    args = parser.parse_args()

    items = all_items()
    count = write_jsonl(items, ROOT / args.output)
    dev_count = write_jsonl([item for item in items if item.metadata["split"] == "dev"], ROOT / args.dev_output)
    test_count = write_jsonl([item for item in items if item.metadata["split"] == "test"], ROOT / args.test_output)
    print(f"wrote {count} messy-memory items to {ROOT / args.output}")
    print(f"wrote {dev_count} dev items to {ROOT / args.dev_output}")
    print(f"wrote {test_count} test items to {ROOT / args.test_output}")


def all_items() -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for scenario in SCENARIOS:
        for order in ["chronological", "scrambled"]:
            for query_style in ["direct_value", "task_action"]:
                items.append(messy_item(scenario, order=order, query_style=query_style))
    return items


def messy_item(scenario: dict[str, Any], order: str, query_style: str) -> BenchmarkItem:
    older_id = f"{scenario['id']}_older"
    middle_id = f"{scenario['id']}_middle"
    current_id = f"{scenario['id']}_current"
    older_block = memory_block(scenario, "older", older_id, "2024-01-05T09:00:00+00:00", superseded_by=[middle_id, current_id])
    middle_block = memory_block(scenario, "middle", middle_id, "2024-02-10T09:00:00+00:00", supersedes=[older_id], superseded_by=[current_id])
    current_block = memory_block(scenario, "current", current_id, "2024-03-20T09:00:00+00:00", supersedes=[older_id, middle_id])
    if order == "chronological":
        target_blocks = [older_block, middle_block, current_block]
    elif order == "scrambled":
        target_blocks = [middle_block, current_block, older_block]
    else:
        raise KeyError(order)
    context_blocks = interleave_distractors(target_blocks, scenario["id"])
    return BenchmarkItem(
        id=f"agent_memory_messy_{scenario['id']}_{order}_{query_style}",
        domain="agent_memory_messy",
        question_type="agent_memory_messy_value",
        context_blocks=context_blocks,
        query=query_for(scenario, query_style),
        gold_answer=scenario["current"],
        gold_evidence_ids=[current_id],
        stale_ids=[older_id, middle_id],
        distractor_ids=[block.id for block in context_blocks if block.id not in {older_id, middle_id, current_id}],
        metadata={
            "category": scenario["category"],
            "memory_key": scenario["memory_key"],
            "split": scenario["split"],
            "variant": order,
            "query_style": query_style,
            "distractor_ratio": 0.4,
            "target_value": scenario["current"],
            "stale_values": [scenario["older"], scenario["middle"]],
            "stale_value": scenario["older"],
        },
    )


def memory_block(
    scenario: dict[str, Any],
    version: str,
    block_id: str,
    timestamp: str,
    supersedes: list[str] | None = None,
    superseded_by: list[str] | None = None,
) -> ContextBlock:
    metadata: dict[str, Any] = {
        "category": scenario["category"],
        "memory_key": scenario["memory_key"],
        "value": scenario[version],
        "version": version,
    }
    if supersedes:
        metadata["supersedes"] = supersedes
    if superseded_by:
        metadata["superseded_by"] = superseded_by
    return ContextBlock(
        id=block_id,
        timestamp=timestamp,
        type="agent_memory",
        trust="high",
        status="unknown",
        text=scenario[f"{version}_text"],
        metadata=metadata,
    )


def interleave_distractors(target_blocks: list[ContextBlock], scenario_id: str) -> list[ContextBlock]:
    first, second = distractors_for(scenario_id)
    return [first, target_blocks[0], target_blocks[1], second, target_blocks[2]]


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
        timestamp="2024-02-22T09:00:00+00:00",
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
