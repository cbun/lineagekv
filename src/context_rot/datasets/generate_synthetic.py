from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from random import Random

from .schema import BenchmarkItem, ContextBlock


DISTRACTOR_RATIOS = [0.0, 0.25, 0.5, 0.75, 0.9]
VARIANTS = [
    "current_late",
    "current_early_stale_late",
    "stale_repeated_after",
    "wrong_duplicate_10x",
    "split_relevant_middle",
]


@dataclass(frozen=True)
class SeedScenario:
    domain: str
    question_type: str
    entity: str
    property_name: str
    stale_value: str
    current_value: str
    query: str
    gold_answer: str
    block_type: str
    distractor_theme: str


DOMAIN_TEMPLATES = [
    (
        "user_preference",
        "preference_update",
        "user preference",
        "communication channel",
        [
            ("Casey", "weekly email digest", "Slack DM only", "What channel should be used for Casey's weekly summary?"),
            ("Morgan", "formal status memo", "brief bullet list", "How should Morgan receive status updates now?"),
            ("Riley", "morning delivery", "end-of-day delivery", "When should Riley receive reminders?"),
            ("Avery", "all details inline", "links plus one-line summaries", "What format should Avery's research updates use?"),
            ("Jordan", "calendar invites for every hold", "only confirmed calendar invites", "Which calendar behavior is current for Jordan?"),
        ],
    ),
    (
        "policy_approval",
        "conflict_resolution",
        "policy",
        "approval rule",
        [
            ("Slack external replies", "allowed without review", "require approval unless directly requested", "Should an external Slack reply be sent without approval?"),
            ("Production deploys", "assistant may deploy after tests", "human approval required before deploy", "Can the assistant deploy to production by itself?"),
            ("Purchasing", "tools under $200 are auto-approved", "all paid tools require confirmation", "Can a $50 tool subscription be purchased automatically?"),
            ("Email forwarding", "forward client email freely", "summarize only unless forwarding is approved", "May the assistant forward a client email verbatim?"),
            ("Public posts", "drafts may be posted after lint", "posting requires explicit approval", "Can the assistant publish a prepared post?"),
        ],
    ),
    (
        "crm_followup",
        "next_action",
        "lead",
        "next action",
        [
            ("Northstar Labs", "send pricing deck", "wait for security questionnaire response", "What is the current next step for Northstar Labs?"),
            ("Cedar Health", "schedule intro call", "send revised DPA", "What is the current next action for Cedar Health?"),
            ("Vantage Robotics", "mark as cold", "book technical validation", "What should happen next for Vantage Robotics?"),
            ("Pioneer Bank", "send generic case study", "prepare SOC2 mapping notes", "What is the current follow-up for Pioneer Bank?"),
            ("Helio Retail", "wait until next quarter", "send pilot scope by Friday", "What is the current next action for Helio Retail?"),
        ],
    ),
    (
        "deployment_state",
        "runtime_state",
        "service",
        "deployment target",
        [
            ("billing-api", "rollback to v1.18.2", "keep v1.19.1 and patch config", "What deployment state is current for billing-api?"),
            ("web-dashboard", "deploy to staging only", "production deploy completed", "Where does web-dashboard stand now?"),
            ("search-worker", "disable queue consumer", "consumer re-enabled after hotfix", "Should search-worker queue consumption be enabled?"),
            ("auth-service", "use legacy token verifier", "use JWKS verifier", "Which verifier should auth-service use?"),
            ("notifier", "pin Node 20.10", "pin Node 22.2", "Which Node version is current for notifier?"),
        ],
    ),
    (
        "tool_trace",
        "final_state",
        "tool run",
        "final result",
        [
            ("invoice sync", "failed with 403", "succeeded after token refresh", "What was the final result of the invoice sync?"),
            ("calendar import", "timed out", "completed on retry", "What is the final calendar import state?"),
            ("S3 upload", "checksum mismatch", "verified after reupload", "What happened with the S3 upload?"),
            ("database migration", "dry run failed", "applied successfully", "What is the final migration state?"),
            ("CRM export", "rate-limited", "completed after backoff", "What is the final CRM export state?"),
        ],
    ),
    (
        "task_state",
        "remaining_work",
        "task",
        "remaining blocker",
        [
            ("analytics dashboard", "blocked on schema choice", "blocked on chart QA only", "What remains blocked for the analytics dashboard?"),
            ("onboarding flow", "copy review", "mobile layout regression", "What is the current blocker for onboarding flow?"),
            ("billing cleanup", "missing invoice data", "waiting for legal approval", "What is the current blocker for billing cleanup?"),
            ("search relevance", "need embeddings", "need evaluation labels", "What is the current remaining blocker for search relevance?"),
            ("team settings", "needs API design", "needs migration rollback plan", "What remains to unblock team settings?"),
        ],
    ),
    (
        "knowledge_base",
        "source_trust",
        "knowledge note",
        "trusted answer",
        [
            ("refund window", "14 days", "30 days for annual plans", "What refund window should be used for annual plans?"),
            ("support tier", "premium includes phone support", "premium is chat-only", "What support channel is current for premium?"),
            ("data retention", "logs retained 180 days", "logs retained 90 days", "How long are logs retained now?"),
            ("SLA target", "99.5 percent", "99.9 percent", "What SLA target is current?"),
            ("backup cadence", "weekly backups", "daily backups", "What backup cadence is current?"),
        ],
    ),
    (
        "multi_hop_memory",
        "multi_hop",
        "project memory",
        "combined decision",
        [
            ("Trailmind launch", "launch in Discord with long-form notes", "launch in Slack with a short demo clip", "Where and how should Trailmind launch?"),
            ("Q3 research post", "publish as a PDF", "publish as blog post with appendix", "What is the current publishing plan for the Q3 research post?"),
            ("demo day", "show CRM workflow", "show memory hygiene workflow", "What workflow should demo day show?"),
            ("alpha onboarding", "invite 50 users by email", "invite 10 design partners by Slack", "Who should be invited for alpha onboarding?"),
            ("investor update", "lead with revenue", "lead with retention and product velocity", "What should the investor update lead with?"),
        ],
    ),
    (
        "billing_ops",
        "account_state",
        "account",
        "billing action",
        [
            ("Acct-1042", "issue credit immediately", "hold credit pending usage audit", "What billing action is current for Acct-1042?"),
            ("Acct-2031", "downgrade to starter", "keep pro through renewal", "What plan action is current for Acct-2031?"),
            ("Acct-7788", "refund duplicate payment", "duplicate payment already refunded", "What should happen for Acct-7788?"),
            ("Acct-5510", "pause invoices", "resume invoices after PO received", "What is the current invoice state for Acct-5510?"),
            ("Acct-8891", "send collections notice", "do not contact while dispute is open", "What billing contact action is current for Acct-8891?"),
        ],
    ),
    (
        "support_case",
        "case_resolution",
        "support case",
        "resolution path",
        [
            ("Case 310", "ask user to reinstall", "escalate to backend because logs show 500s", "What is the current path for Case 310?"),
            ("Case 429", "close as duplicate", "keep open pending customer logs", "What is the current state for Case 429?"),
            ("Case 512", "ship workaround steps", "wait for patch release", "What should support do for Case 512?"),
            ("Case 644", "route to billing", "route to identity team", "Which team should handle Case 644?"),
            ("Case 777", "refund request denied", "refund approved by manager", "What is the current resolution for Case 777?"),
        ],
    ),
]


DISTRACTOR_SENTENCES = [
    "A nearby note discussed onboarding copy but did not change this decision.",
    "A tool log mentioned a transient timeout unrelated to the requested state.",
    "A historical summary repeated an old planning assumption without approval metadata.",
    "A meeting note captured open questions for a different customer.",
    "A policy excerpt covered internal drafts, not external action.",
    "A task checklist included completed cleanup work from another thread.",
    "A CRM note referenced a similar account with a different next step.",
    "A deployment note mentioned staging smoke tests for another service.",
    "A support note described reproduction steps for a separate ticket.",
    "A reminder captured a preference that belongs to a different user.",
]


def build_seed_scenarios(limit: int | None = None) -> list[SeedScenario]:
    seeds: list[SeedScenario] = []
    for domain, question_type, block_type, prop, rows in DOMAIN_TEMPLATES:
        for entity, stale_value, current_value, query in rows:
            seeds.append(
                SeedScenario(
                    domain=domain,
                    question_type=question_type,
                    entity=entity,
                    property_name=prop,
                    stale_value=stale_value,
                    current_value=current_value,
                    query=query,
                    gold_answer=current_value,
                    block_type=block_type,
                    distractor_theme=domain.replace("_", " "),
                )
            )
    return seeds[:limit] if limit else seeds


def generate_items(
    base_limit: int | None = None,
    distractor_ratios: list[float] | None = None,
    variants: list[str] | None = None,
    seed: int = 20260515,
) -> list[BenchmarkItem]:
    rng = Random(seed)
    ratios = distractor_ratios or DISTRACTOR_RATIOS
    variant_names = variants or VARIANTS
    items: list[BenchmarkItem] = []
    for base_idx, scenario in enumerate(build_seed_scenarios(base_limit), start=1):
        for ratio in ratios:
            for variant in variant_names:
                items.append(_build_item(scenario, base_idx, ratio, variant, rng))
    return items


def _build_item(
    scenario: SeedScenario,
    base_idx: int,
    distractor_ratio: float,
    variant: str,
    rng: Random,
) -> BenchmarkItem:
    base_time = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc) + timedelta(hours=base_idx)
    stale_id = f"b{base_idx:03d}_stale"
    current_id = f"b{base_idx:03d}_current"
    support_id = f"b{base_idx:03d}_support"
    stale = ContextBlock(
        id=stale_id,
        timestamp=_iso(base_time),
        type=scenario.block_type,
        trust="high",
        status="superseded",
        text=(
            f"Earlier note for {scenario.entity}: {scenario.property_name} was "
            f"'{scenario.stale_value}'. This was later revisited."
        ),
        metadata={"entity": scenario.entity, "property": scenario.property_name, "value": scenario.stale_value},
    )
    current = ContextBlock(
        id=current_id,
        timestamp=_iso(base_time + timedelta(days=7)),
        type=scenario.block_type,
        trust="high",
        status="current",
        text=(
            f"Current decision for {scenario.entity}: {scenario.property_name} is "
            f"'{scenario.current_value}'. This supersedes the earlier value."
        ),
        metadata={"entity": scenario.entity, "property": scenario.property_name, "value": scenario.current_value},
    )
    support = ContextBlock(
        id=support_id,
        timestamp=_iso(base_time + timedelta(days=7, hours=1)),
        type="decision_record",
        trust="medium",
        status="current",
        text=(
            f"Decision record: for questions about {scenario.entity}, prefer the newest "
            f"high-trust current decision and ignore superseded duplicates."
        ),
        metadata={"entity": scenario.entity, "supports": current_id},
    )

    relevant_count = 3 if variant == "split_relevant_middle" else 2
    distractor_count = _distractor_count(relevant_count, distractor_ratio)
    distractors = _make_distractors(scenario, base_idx, distractor_count, base_time, rng)
    duplicate_blocks = _duplicate_stale_blocks(stale, base_idx, base_time, variant)

    if variant == "current_late":
        blocks = [stale] + distractors + duplicate_blocks + [current]
        relevant_position = "late"
    elif variant == "current_early_stale_late":
        blocks = [current] + distractors + duplicate_blocks + [stale]
        relevant_position = "early"
    elif variant == "stale_repeated_after":
        blocks = [stale, current] + distractors[: max(1, len(distractors) // 2)] + duplicate_blocks + distractors[max(1, len(distractors) // 2):]
        relevant_position = "early"
    elif variant == "wrong_duplicate_10x":
        blocks = [current] + distractors + duplicate_blocks + [stale]
        relevant_position = "early"
    elif variant == "split_relevant_middle":
        pivot = len(distractors) // 2
        blocks = distractors[:pivot] + [stale, support, current] + distractors[pivot:] + duplicate_blocks
        relevant_position = "middle"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    duplicate_wrong_count = len(duplicate_blocks)
    distractor_ids = [b.id for b in distractors]
    stale_ids = [stale_id] + [b.id for b in duplicate_blocks]
    gold_ids = [current_id, support_id] if variant == "split_relevant_middle" else [current_id]
    item_id = f"{scenario.domain}_{base_idx:03d}_{variant}_d{int(distractor_ratio * 100):02d}"
    return BenchmarkItem(
        id=item_id,
        domain=scenario.domain,
        question_type=scenario.question_type,
        context_blocks=blocks,
        query=scenario.query,
        gold_answer=scenario.gold_answer,
        gold_evidence_ids=gold_ids,
        stale_ids=stale_ids,
        distractor_ids=distractor_ids,
        metadata={
            "base_index": base_idx,
            "entity": scenario.entity,
            "property": scenario.property_name,
            "target_value": scenario.current_value,
            "stale_value": scenario.stale_value,
            "distractor_ratio": distractor_ratio,
            "variant": variant,
            "contradiction_count": 1,
            "duplicate_wrong_count": duplicate_wrong_count,
            "relevant_position": relevant_position,
            "trust_metadata": True,
            "timestamp_metadata": True,
        },
    )


def _distractor_count(relevant_count: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    if ratio >= 0.9:
        return max(12, math.ceil((ratio * relevant_count) / (1 - ratio)))
    return math.ceil((ratio * relevant_count) / (1 - ratio))


def _make_distractors(
    scenario: SeedScenario,
    base_idx: int,
    count: int,
    base_time: datetime,
    rng: Random,
) -> list[ContextBlock]:
    blocks: list[ContextBlock] = []
    for idx in range(count):
        sentence = rng.choice(DISTRACTOR_SENTENCES)
        blocks.append(
            ContextBlock(
                id=f"b{base_idx:03d}_d{idx + 1:02d}",
                timestamp=_iso(base_time + timedelta(days=idx % 9, minutes=idx)),
                type="memory_note",
                trust=rng.choice(["low", "medium"]),
                status="irrelevant",
                text=(
                    f"Related-looking {scenario.distractor_theme} note {idx + 1}: "
                    f"{sentence} It does not update {scenario.entity}'s {scenario.property_name}."
                ),
                metadata={"distractor": True, "entity": f"other-{idx % 5}"},
            )
        )
    return blocks


def _duplicate_stale_blocks(
    stale: ContextBlock,
    base_idx: int,
    base_time: datetime,
    variant: str,
) -> list[ContextBlock]:
    duplicate_count = 0
    if variant == "stale_repeated_after":
        duplicate_count = 3
    elif variant == "wrong_duplicate_10x":
        duplicate_count = 10
    blocks: list[ContextBlock] = []
    for idx in range(duplicate_count):
        blocks.append(
            ContextBlock(
                id=f"b{base_idx:03d}_stale_dup{idx + 1:02d}",
                timestamp=_iso(base_time + timedelta(days=8, minutes=idx)),
                type=stale.type,
                trust=stale.trust,
                status="superseded",
                text=f"Duplicate stale note {idx + 1}: {stale.text}",
                metadata={**stale.metadata, "duplicate_of": stale.id},
            )
        )
    return blocks


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
