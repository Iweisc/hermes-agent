#!/usr/bin/env python3
import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Hermes Rust port worktree and branch status."
    )
    parser.add_argument(
        "--worktree-root",
        default="..",
        help="Directory containing agent-* worktrees (default: ../ from the lead worktree).",
    )
    parser.add_argument(
        "--lead-worktree",
        default="agent-01",
        help="Lead integration worktree directory name.",
    )
    parser.add_argument(
        "--expected-workers",
        type=int,
        default=21,
        help="Expected number of worker worktrees excluding the lead.",
    )
    parser.add_argument(
        "--main-ref",
        default="main",
        help="Reference used for drift checks (default: main).",
    )
    parser.add_argument(
        "--tracker",
        default="plans/rust_port_integration.json",
        help="Optional integration tracker JSON used for validation.",
    )
    parser.add_argument(
        "--select",
        help="Dot-separated path to print a subsection of the JSON payload.",
    )
    parser.add_argument(
        "--sync-tracker",
        action="store_true",
        help="Refresh the tracker JSON with live branch state, checklist evidence, and blockers.",
    )
    parser.add_argument(
        "--apply-tracker-edit-plan",
        action="store_true",
        help="Apply non-ambiguous tracker_edit_plan mutations to the tracker JSON and refresh synced state.",
    )
    parser.add_argument(
        "--apply-tracker-choice-path",
        help="Pending tracker choice path to assign to a specific owner worktree.",
    )
    parser.add_argument(
        "--apply-tracker-choice-owner",
        help="Owner worktree to assign for --apply-tracker-choice-path.",
    )
    return parser.parse_args()


def git(args: list[str], cwd: Path, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.rstrip("\n")


def discover_worktrees(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("agent-")
    )


def parse_status_lines(status_lines: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in status_lines:
        if len(line) < 4:
            continue
        parsed[line[3:]] = line[:2]
    return parsed


def collect_worktree_status(path: Path, main_ref: str, lead_head: str | None) -> dict:
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"], path)
    head = git(["rev-parse", "--short", "HEAD"], path)
    subject = git(["log", "-1", "--format=%s"], path)
    upstream = git(["for-each-ref", "--format=%(upstream:short)", f"refs/heads/{branch}"], path)
    status_lines = [line for line in git(["status", "--porcelain"], path).splitlines() if line]
    status_by_path = parse_status_lines(status_lines)
    committed_files = [
        line for line in git(["diff", "--name-only", f"{main_ref}...HEAD"], path).splitlines() if line
    ]
    dirty_files = [
        line for line in git(["diff", "--name-only"], path).splitlines() if line
    ]
    staged_files = [
        line for line in git(["diff", "--cached", "--name-only"], path).splitlines() if line
    ]
    untracked_files = [
        line
        for line in git(["ls-files", "--others", "--exclude-standard"], path).splitlines()
        if line
    ]
    ahead_behind_raw = git(["rev-list", "--left-right", "--count", f"{main_ref}...HEAD"], path)
    behind_main, ahead_main = [int(part) for part in ahead_behind_raw.split()]
    status = "idle"
    if dirty_files or staged_files or untracked_files:
        status = "dirty"
    elif lead_head and head != lead_head:
        status = "diverged"
    active_files = sorted(set(committed_files + dirty_files + staged_files + untracked_files))
    return {
        "worktree": path.name,
        "branch": branch,
        "head": head,
        "subject": subject,
        "upstream": upstream or None,
        "status": status,
        "dirty_file_count": len(dirty_files) + len(staged_files) + len(untracked_files),
        "committed_file_count_vs_main": len(committed_files),
        "ahead_of_main": ahead_main,
        "behind_main": behind_main,
        "active_file_count": len(active_files),
        "active_files": active_files,
        "committed_files_vs_main": committed_files,
        "dirty_files": sorted(set(dirty_files + staged_files + untracked_files)),
        "status_by_path": status_by_path,
        "status_lines": status_lines,
    }


def build_overlap_map(entries: list[dict]) -> list[dict]:
    owners: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        for file_path in entry["active_files"]:
            owners[file_path].append(entry["worktree"])
    overlaps = []
    for file_path, worktrees in sorted(owners.items()):
        if len(worktrees) > 1:
            overlaps.append({"path": file_path, "worktrees": sorted(worktrees)})
    return overlaps


def compute_state_fingerprint(
    lead_entry: dict,
    worker_entries: list[dict],
    main_head: str,
    main_drift_files: list[str],
) -> str:
    fingerprint_payload = {
        "lead": {
            "head": lead_entry["head"],
            "status": lead_entry["status"],
            "active_files": lead_entry["active_files"],
        },
        "workers": [
            {
                "worktree": entry["worktree"],
                "head": entry["head"],
                "status": entry["status"],
                "active_files": entry["active_files"],
            }
            for entry in sorted(worker_entries, key=lambda item: item["worktree"])
        ],
        "main_head": main_head,
        "main_drift_files": main_drift_files,
    }
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_sync_drift(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {
            "summary": {
                "tracker_loaded": False,
                "requires_sync": False,
                "changed_worker_count": 0,
                "workers_with_status_changes": 0,
                "workers_with_file_changes": 0,
                "workers_with_head_changes": 0,
                "new_overlap_path_count": 0,
                "resolved_overlap_path_count": 0,
                "changed_overlap_path_count": 0,
                "lead_head_changed": False,
                "main_head_changed": False,
            },
            "workers": [],
            "overlaps": {
                "new_paths": [],
                "resolved_paths": [],
                "changed_paths": [],
            },
            "baseline": {},
        }

    tracker_workers = {
        item["worktree"]: item
        for item in tracker.get("worker_branches", [])
        if item.get("worktree")
    }
    live_workers = {item["worktree"]: item for item in payload["workers"]}
    worker_changes = []
    for worktree in sorted(set(tracker_workers) | set(live_workers)):
        tracked = tracker_workers.get(worktree)
        live = live_workers.get(worktree)
        if tracked and not live:
            worker_changes.append(
                {
                    "worktree": worktree,
                    "change_type": "missing_live_worker",
                    "tracked_branch": tracked.get("branch"),
                }
            )
            continue
        if live and not tracked:
            worker_changes.append(
                {
                    "worktree": worktree,
                    "change_type": "untracked_live_worker",
                    "live_branch": live.get("branch"),
                }
            )
            continue

        tracked_files = set(tracked.get("observed_dirty_files", []))
        live_files = set(live.get("dirty_files", []))
        added_files = sorted(live_files - tracked_files)
        removed_files = sorted(tracked_files - live_files)
        status_changed = tracked.get("status") != live.get("status")
        head_changed = tracked.get("head") != live.get("head")
        branch_changed = tracked.get("branch") != live.get("branch")
        if not any([added_files, removed_files, status_changed, head_changed, branch_changed]):
            continue
        worker_changes.append(
            {
                "worktree": worktree,
                "change_type": "changed",
                "tracked_branch": tracked.get("branch"),
                "live_branch": live.get("branch"),
                "tracked_status": tracked.get("status"),
                "live_status": live.get("status"),
                "tracked_head": tracked.get("head"),
                "live_head": live.get("head"),
                "added_files": added_files,
                "removed_files": removed_files,
            }
        )

    tracked_overlaps = {
        item["path"]: sorted(item.get("workers", []))
        for item in tracker.get("live_conflict_clusters", [])
        if item.get("path")
    }
    live_overlaps = {
        item["path"]: sorted(item.get("worktrees", []))
        for item in payload.get("shared_file_overlaps", [])
        if item.get("path")
    }
    new_overlap_paths = sorted(set(live_overlaps) - set(tracked_overlaps))
    resolved_overlap_paths = sorted(set(tracked_overlaps) - set(live_overlaps))
    changed_overlap_paths = [
        {
            "path": path,
            "tracked_workers": tracked_overlaps[path],
            "live_workers": live_overlaps[path],
        }
        for path in sorted(set(tracked_overlaps) & set(live_overlaps))
        if tracked_overlaps[path] != live_overlaps[path]
    ]

    tracked_scope = tracker.get("scope", {})
    tracked_drift_files = set(tracked_scope.get("main_drift_files", []))
    live_drift_files = set(payload["summary"].get("lead_vs_main_file_drift", []))
    baseline = {
        "tracked_lead_head": tracked_scope.get("baseline_head"),
        "live_lead_head": payload["lead"].get("head"),
        "tracked_main_head": tracked_scope.get("main_head"),
        "live_main_head": payload["summary"].get("main_head"),
        "tracked_state_fingerprint": tracker.get("state_fingerprint"),
        "live_state_fingerprint": payload["summary"].get("state_fingerprint"),
        "tracker_matches_live_state": tracker.get("state_fingerprint") == payload["summary"].get("state_fingerprint"),
        "drift_files_added": sorted(live_drift_files - tracked_drift_files),
        "drift_files_removed": sorted(tracked_drift_files - live_drift_files),
    }

    return {
        "summary": {
            "tracker_loaded": True,
            "requires_sync": not baseline["tracker_matches_live_state"],
            "changed_worker_count": len(worker_changes),
            "workers_with_status_changes": sum(
                1
                for item in worker_changes
                if item.get("change_type") == "changed"
                and item.get("tracked_status") != item.get("live_status")
            ),
            "workers_with_file_changes": sum(
                1
                for item in worker_changes
                if item.get("change_type") == "changed"
                and (item.get("added_files") or item.get("removed_files"))
            ),
            "workers_with_head_changes": sum(
                1
                for item in worker_changes
                if item.get("change_type") == "changed"
                and item.get("tracked_head") != item.get("live_head")
            ),
            "new_overlap_path_count": len(new_overlap_paths),
            "resolved_overlap_path_count": len(resolved_overlap_paths),
            "changed_overlap_path_count": len(changed_overlap_paths),
            "lead_head_changed": tracked_scope.get("baseline_head") != payload["lead"].get("head"),
            "main_head_changed": tracked_scope.get("main_head") != payload["summary"].get("main_head"),
        },
        "workers": worker_changes,
        "overlaps": {
            "new_paths": new_overlap_paths,
            "resolved_paths": resolved_overlap_paths,
            "changed_paths": changed_overlap_paths,
        },
        "baseline": baseline,
    }


def build_published_head_coverage(payload: dict, repo_root: Path) -> dict:
    workers = []
    for entry in sorted(payload["workers"], key=lambda item: item["worktree"]):
        branch = entry["branch"]
        remote_branch = f"origin/{branch}"
        remote_head = git(
            ["rev-parse", "--verify", "--short", f"refs/remotes/{remote_branch}"],
            repo_root,
            allow_failure=True,
        )
        remote_head = remote_head or None
        remote_exists = bool(remote_head)
        remote_head_integrated = False
        if remote_exists:
            result = subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    f"refs/remotes/{remote_branch}",
                    "HEAD",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            remote_head_integrated = result.returncode == 0

        local_head_matches_remote = (
            entry["head"] == remote_head if remote_head is not None else None
        )
        has_uncommitted_local_changes = bool(entry["dirty_files"])
        has_unpublished_local_head = remote_exists and local_head_matches_remote is False
        published_head_fully_absorbed = (
            remote_head_integrated
            and local_head_matches_remote is True
            and not has_uncommitted_local_changes
        )
        follow_up_needed = remote_head_integrated and (
            has_uncommitted_local_changes or has_unpublished_local_head
        )

        if not remote_exists:
            follow_up_reason = "missing_remote_head"
        elif not remote_head_integrated:
            follow_up_reason = "remote_head_not_integrated"
        elif has_unpublished_local_head:
            follow_up_reason = "local_head_ahead_of_remote_after_merge"
        elif has_uncommitted_local_changes:
            follow_up_reason = "uncommitted_local_changes_after_remote_merge"
        else:
            follow_up_reason = None

        workers.append(
            {
                "worktree": entry["worktree"],
                "branch": branch,
                "remote_branch": remote_branch if remote_exists else None,
                "remote_head": remote_head,
                "local_head": entry["head"],
                "remote_exists": remote_exists,
                "remote_head_integrated": remote_head_integrated,
                "local_head_matches_remote": local_head_matches_remote,
                "has_uncommitted_local_changes": has_uncommitted_local_changes,
                "has_unpublished_local_head": has_unpublished_local_head,
                "published_head_fully_absorbed": published_head_fully_absorbed,
                "follow_up_needed": follow_up_needed,
                "follow_up_reason": follow_up_reason,
                "dirty_files": entry["dirty_files"],
            }
        )

    remote_workers = [item for item in workers if item["remote_exists"]]
    integrated_remote_workers = [
        item for item in remote_workers if item["remote_head_integrated"]
    ]
    follow_up_workers = [item for item in workers if item["follow_up_needed"]]
    return {
        "summary": {
            "tracked_worker_count": len(workers),
            "workers_with_remote_heads": len(remote_workers),
            "workers_without_remote_heads": sum(
                1 for item in workers if not item["remote_exists"]
            ),
            "published_remote_heads_integrated": len(integrated_remote_workers),
            "published_remote_heads_pending": sum(
                1 for item in remote_workers if not item["remote_head_integrated"]
            ),
            "published_heads_fully_absorbed": sum(
                1 for item in workers if item["published_head_fully_absorbed"]
            ),
            "workers_needing_post_merge_follow_up": len(follow_up_workers),
            "workers_with_uncommitted_local_changes_after_merge": sum(
                1
                for item in workers
                if item["follow_up_reason"]
                == "uncommitted_local_changes_after_remote_merge"
            ),
            "workers_with_unpublished_local_heads_after_merge": sum(
                1
                for item in workers
                if item["follow_up_reason"] == "local_head_ahead_of_remote_after_merge"
            ),
            "first_pending_remote_head": next(
                (
                    item["worktree"]
                    for item in workers
                    if item["follow_up_reason"] == "remote_head_not_integrated"
                ),
                None,
            ),
            "first_post_merge_follow_up": follow_up_workers[0]["worktree"]
            if follow_up_workers
            else None,
        },
        "workers": workers,
        "post_merge_follow_up": follow_up_workers,
    }


def build_live_conflict_clusters(payload: dict) -> list[dict]:
    clusters = []
    for item in payload["tracker_validation"]["remediation"]["path_actions"]:
        if len(item["active_workers"]) <= 1:
            continue
        clusters.append(
            {
                "path": item["path"],
                "workers": item["active_workers"],
                "owner": item["owner"],
                "reason": item["owner_reason"],
            }
        )
    return clusters


def load_tracker(path_str: str) -> tuple[Path | None, dict | None]:
    if not path_str:
        return None, None
    path = Path(path_str)
    if not path.exists():
        return path, None
    try:
        tracker = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"tracker file is not valid JSON: {path} ({exc})") from exc
    if not isinstance(tracker, dict):
        raise RuntimeError(f"tracker root must be an object: {path}")
    return path, tracker


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def tokenize_path(path: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", path.lower()) if token}


GENERIC_PATH_TOKENS = {
    "crates",
    "hermes",
    "rs",
    "src",
    "core",
    "py",
    "cmd",
    "main",
    "tests",
    "test",
    "tools",
    "tool",
    "cli",
}


def meaningful_tokens(path: str) -> set[str]:
    return {token for token in tokenize_path(path) if token not in GENERIC_PATH_TOKENS}


def score_lane_candidate(path: str, plan: dict) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    candidate = Path(path)
    candidate_tokens = meaningful_tokens(path)
    allowed_paths = plan.get("allowed_paths", [])
    proposed_scope = plan.get("proposed_scope", "")
    scope_tokens = meaningful_tokens(proposed_scope)

    root_match = False
    parent_match = False
    overlap_tokens: set[str] = set()
    for allowed in allowed_paths:
        allowed_path = Path(allowed)
        allowed_tokens = meaningful_tokens(allowed)
        overlap = candidate_tokens & allowed_tokens
        if candidate.parts and allowed_path.parts and candidate.parts[0] == allowed_path.parts[0]:
            root_match = True
        if candidate.parent == allowed_path.parent and overlap:
            parent_match = True
        if overlap:
            overlap_tokens.update(overlap)
            reasons.append(f"shares tokens {sorted(overlap)} with {allowed}")

    scope_overlap = candidate_tokens & scope_tokens
    if root_match:
        score += 1
        reasons.append("shares top-level module area")
    if parent_match:
        score += 3
        reasons.append("shares directory with an overlapping lane path")
    if overlap_tokens:
        score += 2 * len(overlap_tokens)
    if scope_overlap:
        score += len(scope_overlap)
        reasons.append(f"matches proposed scope tokens {sorted(scope_overlap)}")
    return score, unique_preserve_order(reasons)


def build_lane_gap_report(
    branch_actions: list[dict],
    branch_plans: dict[str, dict],
    path_actions: list[dict],
) -> dict:
    unowned_paths = sorted(
        {
            path
            for item in branch_actions
            for path in item["unowned_paths"]
        }
    )
    path_activity = {item["path"]: item["active_workers"] for item in path_actions}
    path_recommendations = []
    for path in unowned_paths:
        scored_candidates = []
        for worktree, plan in branch_plans.items():
            score, reasons = score_lane_candidate(path, plan)
            if score > 0:
                scored_candidates.append(
                    {
                        "worktree": worktree,
                        "score": score,
                        "reasons": reasons,
                    }
                )
        scored_candidates.sort(key=lambda item: (-item["score"], item["worktree"]))
        top = scored_candidates[0] if scored_candidates else None
        next_score = scored_candidates[1]["score"] if len(scored_candidates) > 1 else None
        if top and top["score"] >= 5 and (next_score is None or top["score"] >= next_score + 2):
            suggestion = "assign_to_existing_lane"
            confidence = "high"
            suggested_owner = top["worktree"]
            rationale = top["reasons"]
        elif top and top["score"] >= 2:
            suggestion = "review_existing_lane"
            confidence = "medium"
            suggested_owner = top["worktree"]
            rationale = top["reasons"]
        else:
            suggestion = "create_or_expand_lane"
            confidence = "low"
            suggested_owner = None
            rationale = ["no strong existing lane match from allowed_paths or proposed_scope"]

        path_recommendations.append(
            {
                "path": path,
                "active_workers": path_activity.get(path, []),
                "suggestion": suggestion,
                "confidence": confidence,
                "suggested_owner": suggested_owner,
                "rationale": rationale,
                "candidate_scores": scored_candidates[:5],
            }
        )

    path_recommendation_map = {item["path"]: item for item in path_recommendations}
    branch_recommendations = []
    for item in branch_actions:
        if not item["unowned_paths"]:
            continue
        branch_paths = [path_recommendation_map[path] for path in item["unowned_paths"]]
        branch_recommendations.append(
            {
                "worktree": item["worktree"],
                "branch": item["branch"],
                "unowned_paths": branch_paths,
                "high_confidence_assignments": [
                    path for path in branch_paths if path["confidence"] == "high"
                ],
                "needs_new_lane": [
                    path for path in branch_paths if path["suggestion"] == "create_or_expand_lane"
                ],
                "recommended_next_action": (
                    "Assign the high-confidence paths, then decide whether the remaining files justify a new lane."
                    if any(path["confidence"] == "high" for path in branch_paths)
                    else "No strong owner match exists; either define a new lane or explicitly expand an existing one."
                ),
            }
        )

    return {
        "paths": path_recommendations,
        "branches": branch_recommendations,
        "summary": {
            "unowned_path_count": len(path_recommendations),
            "branches_with_unowned_paths": len(branch_recommendations),
            "high_confidence_assignments": sum(
                1 for item in path_recommendations if item["confidence"] == "high"
            ),
            "medium_confidence_assignments": sum(
                1 for item in path_recommendations if item["confidence"] == "medium"
            ),
            "new_lane_candidates": sum(
                1 for item in path_recommendations if item["suggestion"] == "create_or_expand_lane"
            ),
        },
    }


def build_tracker_realignments(
    branch_triage: dict,
    lane_gap_report: dict,
    branch_plans: dict[str, dict],
) -> dict:
    branch_triage_map = {item["worktree"]: item for item in branch_triage["branches"]}
    allowed_path_additions = []
    review_candidates = []
    new_lane_candidates = []

    for path_item in lane_gap_report["paths"]:
        suggested_owner = path_item["suggested_owner"]
        if path_item["suggestion"] == "assign_to_existing_lane" and suggested_owner:
            plan = branch_plans.get(suggested_owner, {})
            existing_allowed = plan.get("allowed_paths", [])
            if path_item["path"] not in existing_allowed:
                allowed_path_additions.append(
                    {
                        "owner": suggested_owner,
                        "path": path_item["path"],
                        "confidence": path_item["confidence"],
                        "reason": path_item["rationale"],
                        "current_allowed_paths": existing_allowed,
                    }
                )
        elif path_item["suggestion"] == "review_existing_lane" and suggested_owner:
            review_candidates.append(
                {
                    "owner": suggested_owner,
                    "path": path_item["path"],
                    "confidence": path_item["confidence"],
                    "reason": path_item["rationale"],
                    "candidate_scores": path_item["candidate_scores"],
                }
            )
        else:
            branch_holders = [branch_triage_map.get(worker) for worker in path_item["active_workers"]]
            new_lane_candidates.append(
                {
                    "path": path_item["path"],
                    "active_workers": path_item["active_workers"],
                    "holding_dispositions": [
                        {
                            "worktree": item["worktree"],
                            "disposition": item["disposition"],
                            "lifecycle": item.get("lifecycle_state"),
                        }
                        for item in branch_holders
                        if item
                    ],
                    "reason": path_item["rationale"],
                    "candidate_scores": path_item["candidate_scores"],
                }
            )

    branch_actions = []
    for branch in lane_gap_report["branches"]:
        additions = [
            item for item in allowed_path_additions if item["path"] in {path["path"] for path in branch["unowned_paths"]}
        ]
        reviews = [
            item for item in review_candidates if item["path"] in {path["path"] for path in branch["unowned_paths"]}
        ]
        new_lanes = [
            item for item in new_lane_candidates if item["path"] in {path["path"] for path in branch["unowned_paths"]}
        ]
        if not any([additions, reviews, new_lanes]):
            continue
        branch_actions.append(
            {
                "worktree": branch["worktree"],
                "branch": branch["branch"],
                "add_to_existing_lane": additions,
                "review_with_owner": reviews,
                "requires_new_lane": new_lanes,
                "recommended_tracker_action": (
                    "Apply the allowed_paths additions, then resolve the remaining review or new-lane items."
                    if additions
                    else "Resolve owner review items or define new lanes before treating this branch as aligned."
                ),
            }
        )

    return {
        "allowed_path_additions": allowed_path_additions,
        "review_candidates": review_candidates,
        "new_lane_candidates": new_lane_candidates,
        "branch_actions": branch_actions,
        "summary": {
            "allowed_path_addition_count": len(allowed_path_additions),
            "review_candidate_count": len(review_candidates),
            "new_lane_candidate_count": len(new_lane_candidates),
            "branches_with_tracker_actions": len(branch_actions),
        },
    }


def resolve_owner(
    path: str,
    branch_plans: dict[str, dict],
    lead_only: set[str],
    ownership_overrides: list[dict],
) -> dict | None:
    if path in lead_only:
        return {
            "owner": "agent-01",
            "reason": "Reserved lead-only integration file.",
            "source": "lead_only",
        }

    for rule in ownership_overrides:
        rule_path = rule.get("path")
        owner = rule.get("owner")
        if rule_path == path and owner:
            return {
                "owner": owner,
                "reason": rule.get("reason", "Tracker ownership override."),
                "source": "ownership_override",
            }

    matches: list[tuple[str, str]] = []
    for worktree, plan in branch_plans.items():
        for pattern in plan.get("allowed_paths", []):
            if fnmatch.fnmatch(path, pattern):
                matches.append((worktree, pattern))

    unique_worktrees = sorted({worktree for worktree, _ in matches})
    if len(unique_worktrees) == 1:
        return {
            "owner": unique_worktrees[0],
            "reason": f"Only {unique_worktrees[0]} claims this path in allowed_paths.",
            "source": "allowed_paths",
        }
    if len(unique_worktrees) > 1:
        return {
            "owner": None,
            "candidates": unique_worktrees,
            "reason": "Multiple workers claim this path in allowed_paths.",
            "source": "ambiguous_allowed_paths",
        }
    return None


def classify_branch_disposition(
    keep_count: int,
    release_count: int,
    ambiguous_count: int,
    unowned_count: int,
    lead_only_count: int,
) -> tuple[str, str]:
    if not any([release_count, ambiguous_count, unowned_count, lead_only_count]):
        return "merge_ready", "Branch only holds owned paths."
    if keep_count == 0 and release_count > 0 and not ambiguous_count and not unowned_count:
        return "release_only", "Branch is only carrying other owners' files and should just release them."
    if keep_count == 0 and (unowned_count or ambiguous_count):
        return "retask_or_stop", "Branch has no retained lane and needs reassignment or shutdown."
    if keep_count > 0 and (unowned_count or ambiguous_count):
        return "salvage_with_reassignment", "Branch keeps some lane-owned files but still needs ownership cleanup."
    if keep_count > 0 and (release_count or lead_only_count):
        return "salvage_in_lane", "Branch has a real lane but must shed cross-lane files first."
    return "manual_review", "Branch needs manual review."


def build_branch_triage(branch_actions: list[dict], owner_runbook: list[dict]) -> dict:
    owner_wait_map = {item["owner"]: item for item in owner_runbook}
    triage = []
    for item in branch_actions:
        keep_count = len(item["keep_paths"])
        release_count = len(item["release_paths"])
        ambiguous_count = len(item["ambiguous_paths"])
        unowned_count = len(item["unowned_paths"])
        lead_only_count = len(item["lead_only_hits"])
        active_count = keep_count + release_count + ambiguous_count + unowned_count
        disposition, rationale = classify_branch_disposition(
            keep_count,
            release_count,
            ambiguous_count,
            unowned_count,
            lead_only_count,
        )
        dominant_release_owner = None
        if item["release_paths"]:
            owner_counts: dict[str, int] = defaultdict(int)
            for release in item["release_paths"]:
                owner_counts[release["owner"]] += 1
            dominant_release_owner = sorted(
                owner_counts.items(),
                key=lambda pair: (-pair[1], pair[0]),
            )[0][0]
        owner_state = owner_wait_map.get(item["worktree"])
        triage.append(
            {
                "worktree": item["worktree"],
                "branch": item["branch"],
                "disposition": disposition,
                "rationale": rationale,
                "active_count": active_count,
                "keep_count": keep_count,
                "release_count": release_count,
                "ambiguous_count": ambiguous_count,
                "unowned_count": unowned_count,
                "lead_only_count": lead_only_count,
                "keep_ratio": round((keep_count / active_count), 3) if active_count else 0.0,
                "dominant_release_owner": dominant_release_owner,
                "awaiting_incoming_releases": (
                    owner_state["awaiting_release_from"] if owner_state else []
                ),
                "incoming_release_count": (
                    len(owner_state["awaiting_release_from"]) if owner_state else 0
                ),
                "recommended_next_action": {
                    "merge_ready": "Rebase and publish the branch.",
                    "release_only": "Run the release batch, then decide whether the branch still needs to exist.",
                    "retask_or_stop": "Do not merge as-is; reassign the remaining files or shut the branch down.",
                    "salvage_with_reassignment": "Keep owned files, strip the rest, and resolve unowned paths before restacking.",
                    "salvage_in_lane": "Strip cross-lane files, then keep the branch in its declared lane.",
                    "manual_review": "Inspect this branch manually before any merge work.",
                }[disposition],
                "command_batch": item["command_batch"],
            }
        )

    prioritized = sorted(
        triage,
        key=lambda item: (
            {
                "retask_or_stop": 0,
                "release_only": 1,
                "salvage_with_reassignment": 2,
                "salvage_in_lane": 3,
                "merge_ready": 4,
                "manual_review": 5,
            }[item["disposition"]],
            -item["lead_only_count"],
            -item["release_count"],
            item["worktree"],
        ),
    )
    by_disposition: dict[str, int] = defaultdict(int)
    for item in triage:
        by_disposition[item["disposition"]] += 1
    return {
        "branches": triage,
        "prioritized": prioritized,
        "summary": {
            "merge_ready": by_disposition["merge_ready"],
            "release_only": by_disposition["release_only"],
            "retask_or_stop": by_disposition["retask_or_stop"],
            "salvage_with_reassignment": by_disposition["salvage_with_reassignment"],
            "salvage_in_lane": by_disposition["salvage_in_lane"],
            "manual_review": by_disposition["manual_review"],
            "branches_with_incoming_releases": sum(
                1 for item in triage if item["incoming_release_count"] > 0
            ),
        },
    }


def build_branch_lifecycle(branch_triage: dict, owner_runbook: list[dict]) -> dict:
    owner_map = {item["owner"]: item for item in owner_runbook}
    lifecycle = []
    for item in branch_triage["branches"]:
        owner_state = owner_map.get(item["worktree"])
        incoming_paths = (
            [path["path"] for path in owner_state["owned_paths"] if path["release_from"]]
            if owner_state
            else []
        )
        lifecycle_state = "manual_review"
        rationale = "Inspect branch manually before deciding whether it should persist."
        if item["disposition"] == "release_only":
            if item["incoming_release_count"] > 0:
                lifecycle_state = "receive_then_retain"
                rationale = (
                    "Branch currently carries no owned files, but other workers still need to hand lane-owned paths back to it."
                )
            else:
                lifecycle_state = "retire_after_release"
                rationale = "Branch is only a release carrier and can be retired after cleanup, rebase, and publish."
        elif item["disposition"] == "retask_or_stop":
            lifecycle_state = "retask_or_stop"
            rationale = "Branch has no viable retained lane and should be reassigned or shut down after release."
        elif item["disposition"] == "salvage_with_reassignment":
            lifecycle_state = "realign_then_retain"
            rationale = "Branch should stay alive only if its unowned files are reassigned and its owned lane remains useful."
        elif item["disposition"] == "salvage_in_lane":
            lifecycle_state = "retain_after_cleanup"
            rationale = "Branch has a defensible lane and should continue after dropping cross-lane files."
        elif item["disposition"] == "merge_ready":
            lifecycle_state = "merge_then_close"
            rationale = "Branch can merge directly and then be closed."

        lifecycle.append(
            {
                "worktree": item["worktree"],
                "branch": item["branch"],
                "disposition": item["disposition"],
                "lifecycle_state": lifecycle_state,
                "rationale": rationale,
                "incoming_release_count": item["incoming_release_count"],
                "incoming_paths": incoming_paths,
                "recommended_end_state": {
                    "receive_then_retain": "Keep the branch alive through incoming handoffs, then re-evaluate merge readiness.",
                    "retire_after_release": "Do not plan a merge; publish only if needed for auditability, then retire the branch.",
                    "retask_or_stop": "Do not merge as-is; either move the remaining intent elsewhere or close the branch.",
                    "realign_then_retain": "Resolve ownership gaps first, then keep the branch in service for its remaining lane.",
                    "retain_after_cleanup": "Keep the branch active in its declared lane after cleanup.",
                    "merge_then_close": "Merge the branch and close it.",
                    "manual_review": "Hold for manual branch review.",
                }[lifecycle_state],
                "command_batch": item["command_batch"],
            }
        )

    prioritized = sorted(
        lifecycle,
        key=lambda item: (
            {
                "retask_or_stop": 0,
                "retire_after_release": 1,
                "receive_then_retain": 2,
                "realign_then_retain": 3,
                "retain_after_cleanup": 4,
                "merge_then_close": 5,
                "manual_review": 6,
            }[item["lifecycle_state"]],
            -item["incoming_release_count"],
            item["worktree"],
        ),
    )
    summary: dict[str, int] = defaultdict(int)
    for item in lifecycle:
        summary[item["lifecycle_state"]] += 1
    return {
        "branches": lifecycle,
        "prioritized": prioritized,
        "summary": {
            "receive_then_retain": summary["receive_then_retain"],
            "retire_after_release": summary["retire_after_release"],
            "retask_or_stop": summary["retask_or_stop"],
            "realign_then_retain": summary["realign_then_retain"],
            "retain_after_cleanup": summary["retain_after_cleanup"],
            "merge_then_close": summary["merge_then_close"],
            "manual_review": summary["manual_review"],
        },
    }


def build_conflict_runbook(
    path_actions: list[dict],
    branch_actions: list[dict],
    branch_plans: dict[str, dict],
    hotspot_paths: set[str],
) -> dict:
    branch_action_map = {item["worktree"]: item for item in branch_actions}
    owner_wave_map = {
        worktree: plan.get("merge_wave")
        for worktree, plan in branch_plans.items()
    }
    clusters = []
    for item in path_actions:
        if len(item["active_workers"]) <= 1:
            continue
        owner = item["owner"]
        release_operations = []
        for worktree in item["release_from"]:
            branch_action = branch_action_map.get(worktree)
            release_entry = None
            if branch_action:
                release_entry = next(
                    (
                        release_path
                        for release_path in branch_action["release_paths"]
                        if release_path["path"] == item["path"]
                    ),
                    None,
                )
            if not branch_action or not release_entry:
                continue
            release_operations.append(
                {
                    "worktree": worktree,
                    "branch": branch_action["branch"],
                    "cleanup_command": release_entry["cleanup_command"],
                    "status_code": release_entry["status_code"],
                    "remaining_release_count": len(branch_action["release_paths"]),
                    "remaining_unowned_count": len(branch_action["unowned_paths"]),
                    "remaining_ambiguous_count": len(branch_action["ambiguous_paths"]),
                    "lead_only_count": len(branch_action["lead_only_hits"]),
                    "full_branch_batch": branch_action["command_batch"],
                }
            )
        involved_waves = sorted(
            {
                owner_wave_map.get(worktree)
                for worktree in item["active_workers"]
                if owner_wave_map.get(worktree) is not None
            }
        )
        owner_wave = 0 if owner == "agent-01" else owner_wave_map.get(owner)
        blocking_wave = min(
            [wave for wave in ([owner_wave] + involved_waves) if wave is not None],
            default=None,
        )
        owner_branch = branch_action_map.get(owner) if owner else None
        if item["owner_source"] == "lead_only":
            severity = "critical"
            blocker_type = "lead_only_overlap"
        elif item["path"] in hotspot_paths:
            severity = "high"
            blocker_type = "hotspot_overlap"
        elif owner is None:
            severity = "high"
            blocker_type = "ownerless_overlap"
        else:
            severity = "medium"
            blocker_type = "owned_overlap"
        clusters.append(
            {
                "path": item["path"],
                "severity": severity,
                "blocker_type": blocker_type,
                "blocking_wave": blocking_wave,
                "active_workers": item["active_workers"],
                "owner": owner,
                "owner_reason": item["owner_reason"],
                "owner_source": item["owner_source"],
                "owner_ready_after_cleanup": owner_branch["ready_after_cleanup"]
                if owner_branch
                else owner == "agent-01",
                "owner_branch_release_count": len(owner_branch["release_paths"]) if owner_branch else 0,
                "owner_branch_unowned_count": len(owner_branch["unowned_paths"]) if owner_branch else 0,
                "owner_branch_ambiguous_count": len(owner_branch["ambiguous_paths"]) if owner_branch else 0,
                "release_operations": release_operations,
                "command_batch": unique_preserve_order(
                    [operation["cleanup_command"] for operation in release_operations]
                ),
                "release_count": len(release_operations),
            }
        )

    prioritized = sorted(
        clusters,
        key=lambda item: (
            item["blocking_wave"] if item["blocking_wave"] is not None else 999,
            {
                "critical": 0,
                "high": 1,
                "medium": 2,
            }[item["severity"]],
            -len(item["active_workers"]),
            item["path"],
        ),
    )
    return {
        "clusters": clusters,
        "prioritized": prioritized,
        "summary": {
            "overlap_path_count": len(clusters),
            "critical_count": sum(1 for item in clusters if item["severity"] == "critical"),
            "high_count": sum(1 for item in clusters if item["severity"] == "high"),
            "medium_count": sum(1 for item in clusters if item["severity"] == "medium"),
            "ownerless_overlap_count": sum(1 for item in clusters if item["owner"] is None),
            "lead_only_overlap_count": sum(
                1 for item in clusters if item["blocker_type"] == "lead_only_overlap"
            ),
            "hotspot_overlap_count": sum(
                1 for item in clusters if item["blocker_type"] == "hotspot_overlap"
            ),
            "wave_zero_overlap_count": sum(1 for item in clusters if item["blocking_wave"] == 0),
        },
    }


def build_decision_queue(
    lane_gap_report: dict,
    tracker_realignments: dict,
    branch_triage: dict,
    branch_lifecycle: dict,
    conflict_runbook: dict,
    branch_actions: list[dict],
    branch_plans: dict[str, dict],
) -> dict:
    lane_map = {item["path"]: item for item in lane_gap_report["paths"]}
    triage_map = {item["worktree"]: item for item in branch_triage["branches"]}
    lifecycle_map = {item["worktree"]: item for item in branch_lifecycle["branches"]}
    conflict_map = {item["path"]: item for item in conflict_runbook["clusters"]}
    branch_action_map = {item["worktree"]: item for item in branch_actions}

    def worker_contexts(active_workers: list[str]) -> list[dict]:
        contexts = []
        for worktree in active_workers:
            triage = triage_map.get(worktree, {})
            lifecycle = lifecycle_map.get(worktree, {})
            branch_action = branch_action_map.get(worktree, {})
            plan = branch_plans.get(worktree, {})
            contexts.append(
                {
                    "worktree": worktree,
                    "merge_wave": plan.get("merge_wave"),
                    "disposition": triage.get("disposition"),
                    "lifecycle_state": lifecycle.get("lifecycle_state"),
                    "release_count": len(branch_action.get("release_paths", [])),
                    "unowned_count": len(branch_action.get("unowned_paths", [])),
                    "ambiguous_count": len(branch_action.get("ambiguous_paths", [])),
                    "command_batch": branch_action.get("command_batch", []),
                }
            )
        return contexts

    def base_decision(
        path: str,
        decision_type: str,
        suggested_owner: str | None,
        confidence: str,
        reason: list[str],
        candidate_scores: list[dict],
    ) -> dict:
        lane_item = lane_map[path]
        conflict_item = conflict_map.get(path)
        active_workers = lane_item["active_workers"]
        owner_wave = None
        if suggested_owner == "agent-01":
            owner_wave = 0
        elif suggested_owner:
            owner_wave = branch_plans.get(suggested_owner, {}).get("merge_wave")
        if conflict_item:
            blocking_wave = conflict_item["blocking_wave"]
            overlap_severity = conflict_item["severity"]
            overlap_type = conflict_item["blocker_type"]
        else:
            worker_waves = [
                branch_plans.get(worktree, {}).get("merge_wave")
                for worktree in active_workers
                if branch_plans.get(worktree, {}).get("merge_wave") is not None
            ]
            blocking_wave = min(
                [wave for wave in worker_waves + [owner_wave] if wave is not None],
                default=None,
            )
            overlap_severity = None
            overlap_type = None
        return {
            "path": path,
            "decision_type": decision_type,
            "confidence": confidence,
            "suggested_owner": suggested_owner,
            "suggested_owner_merge_wave": owner_wave,
            "active_workers": active_workers,
            "worker_contexts": worker_contexts(active_workers),
            "blocking_wave": blocking_wave,
            "overlap": bool(conflict_item),
            "overlap_severity": overlap_severity,
            "overlap_type": overlap_type,
            "rationale": reason,
            "candidate_scores": candidate_scores,
            "current_suggestion": lane_item["suggestion"],
        }

    decisions = []
    for item in tracker_realignments["allowed_path_additions"]:
        decision = base_decision(
            item["path"],
            "apply_allowed_path_addition",
            item["owner"],
            item["confidence"],
            item["reason"],
            lane_map[item["path"]]["candidate_scores"],
        )
        decision["recommended_tracker_action"] = (
            f"Add {item['path']} to {item['owner']} allowed_paths and keep the file in that lane."
        )
        decisions.append(decision)

    for item in tracker_realignments["review_candidates"]:
        decision = base_decision(
            item["path"],
            "review_existing_lane",
            item["owner"],
            item["confidence"],
            item["reason"],
            item["candidate_scores"],
        )
        decision["recommended_tracker_action"] = (
            f"Review whether {item['owner']} should explicitly own {item['path']} before cleanup continues."
        )
        decisions.append(decision)

    for item in tracker_realignments["new_lane_candidates"]:
        decision = base_decision(
            item["path"],
            "define_new_lane",
            None,
            "low",
            item["reason"],
            item["candidate_scores"],
        )
        decision["holding_dispositions"] = item["holding_dispositions"]
        decision["recommended_tracker_action"] = (
            f"Define a new lane or expand an existing one for {item['path']} before treating the holding branches as aligned."
        )
        decisions.append(decision)

    prioritized = sorted(
        decisions,
        key=lambda item: (
            item["blocking_wave"] if item["blocking_wave"] is not None else 999,
            {
                "critical": 0,
                "high": 1,
                "medium": 2,
                None: 3,
            }[item["overlap_severity"]],
            {
                "apply_allowed_path_addition": 0,
                "review_existing_lane": 1,
                "define_new_lane": 2,
            }[item["decision_type"]],
            -len(item["active_workers"]),
            item["path"],
        ),
    )
    return {
        "decisions": decisions,
        "prioritized": prioritized,
        "summary": {
            "decision_count": len(decisions),
            "allowed_path_addition_count": sum(
                1 for item in decisions if item["decision_type"] == "apply_allowed_path_addition"
            ),
            "review_existing_lane_count": sum(
                1 for item in decisions if item["decision_type"] == "review_existing_lane"
            ),
            "define_new_lane_count": sum(
                1 for item in decisions if item["decision_type"] == "define_new_lane"
            ),
            "overlap_decision_count": sum(1 for item in decisions if item["overlap"]),
            "wave_zero_decision_count": sum(1 for item in decisions if item["blocking_wave"] == 0),
            "retask_or_stop_worker_count": len(
                {
                    context["worktree"]
                    for item in decisions
                    for context in item["worker_contexts"]
                    if context["disposition"] == "retask_or_stop"
                }
            ),
        },
    }


def build_lane_proposals(decision_queue: dict, branch_plans: dict[str, dict]) -> dict:
    grouped_single_holder: dict[str, list[dict]] = defaultdict(list)
    dedicated_lane_candidates: list[dict] = []
    for item in decision_queue["prioritized"]:
        if item["decision_type"] != "define_new_lane":
            continue
        if len(item["active_workers"]) == 1:
            grouped_single_holder[item["active_workers"][0]].append(item)
        else:
            dedicated_lane_candidates.append(item)

    proposals = []
    for worktree, items in sorted(grouped_single_holder.items()):
        plan = branch_plans.get(worktree, {})
        all_paths = sorted(item["path"] for item in items)
        blocking_wave = min(
            (item["blocking_wave"] for item in items if item["blocking_wave"] is not None),
            default=None,
        )
        disposition = items[0]["worker_contexts"][0].get("disposition") if items[0]["worker_contexts"] else None
        lifecycle = items[0]["worker_contexts"][0].get("lifecycle_state") if items[0]["worker_contexts"] else None
        scope_tokens = sorted(
            {
                token
                for item in items
                for token in meaningful_tokens(item["path"])
            }
        )
        proposal_type = "expand_existing_lane"
        recommendation = (
            f"Expand {worktree} allowed_paths to cover these files and keep them in the current lane."
        )
        if disposition == "retask_or_stop":
            proposal_type = "retask_as_expanded_lane"
            recommendation = (
                f"Retask {worktree} by expanding its lane to include these files; otherwise shut the branch down."
            )
        proposals.append(
            {
                "proposal_type": proposal_type,
                "target_worktree": worktree,
                "merge_wave": plan.get("merge_wave"),
                "blocking_wave": blocking_wave,
                "current_scope": plan.get("proposed_scope"),
                "paths": all_paths,
                "path_count": len(all_paths),
                "scope_tokens": scope_tokens,
                "current_disposition": disposition,
                "current_lifecycle_state": lifecycle,
                "source_decisions": [item["path"] for item in items],
                "recommended_tracker_action": recommendation,
            }
        )

    for item in dedicated_lane_candidates:
        candidate_workers = sorted(
            {
                context["worktree"]
                for context in item["worker_contexts"]
            }
        )
        candidate_wave = min(
            (
                context["merge_wave"]
                for context in item["worker_contexts"]
                if context.get("merge_wave") is not None
            ),
            default=None,
        )
        proposals.append(
            {
                "proposal_type": "create_dedicated_lane",
                "target_worktree": None,
                "merge_wave": candidate_wave,
                "blocking_wave": item["blocking_wave"],
                "current_scope": None,
                "paths": [item["path"]],
                "path_count": 1,
                "scope_tokens": sorted(meaningful_tokens(item["path"])),
                "candidate_workers": candidate_workers,
                "holding_dispositions": item.get("holding_dispositions", []),
                "source_decisions": [item["path"]],
                "recommended_tracker_action": (
                    f"Create a dedicated lane for {item['path']} or explicitly reassign it to one of {candidate_workers}."
                ),
            }
        )

    prioritized = sorted(
        proposals,
        key=lambda item: (
            item["blocking_wave"] if item["blocking_wave"] is not None else 999,
            {
                "create_dedicated_lane": 0,
                "retask_as_expanded_lane": 1,
                "expand_existing_lane": 2,
            }[item["proposal_type"]],
            -item["path_count"],
            item["paths"][0],
        ),
    )
    return {
        "proposals": proposals,
        "prioritized": prioritized,
        "summary": {
            "proposal_count": len(proposals),
            "expand_existing_lane_count": sum(
                1 for item in proposals if item["proposal_type"] == "expand_existing_lane"
            ),
            "retask_as_expanded_lane_count": sum(
                1 for item in proposals if item["proposal_type"] == "retask_as_expanded_lane"
            ),
            "create_dedicated_lane_count": sum(
                1 for item in proposals if item["proposal_type"] == "create_dedicated_lane"
            ),
            "affected_path_count": sum(item["path_count"] for item in proposals),
            "earliest_blocking_wave": min(
                (item["blocking_wave"] for item in proposals if item["blocking_wave"] is not None),
                default=None,
            ),
        },
    }


def humanize_scope_fragment(path: str) -> str:
    candidate = Path(path)
    name = candidate.stem if candidate.suffix else candidate.name
    return " ".join(token for token in re.split(r"[_\\-]+", name) if token)


def build_tracker_mutations(
    worktree: str,
    branch_plans: dict[str, dict],
    add_allowed_paths: list[str],
    scope_fragments: list[str],
) -> list[dict]:
    plan = branch_plans.get(worktree, {})
    mutations = []
    if add_allowed_paths:
        mutations.append(
            {
                "field": "allowed_paths",
                "operation": "append_unique",
                "values": add_allowed_paths,
            }
        )
    if scope_fragments:
        mutations.append(
            {
                "field": "proposed_scope",
                "operation": "append_scope_fragments",
                "current_value": plan.get("proposed_scope"),
                "values": scope_fragments,
            }
        )
    return mutations


def merge_scope_fragments(current_scope: str | None, scope_fragments: list[str]) -> str:
    existing_parts = [
        part.strip()
        for part in (current_scope or "").split(",")
        if part.strip()
    ]
    normalized = {part.lower() for part in existing_parts}
    merged = list(existing_parts)
    for fragment in scope_fragments:
        candidate = fragment.strip()
        if not candidate:
            continue
        normalized_candidate = candidate.lower()
        if normalized_candidate in normalized:
            continue
        merged.append(candidate)
        normalized.add(normalized_candidate)
    return ", ".join(merged)


def build_tracker_patch_preview(
    worktree: str,
    branch_plans: dict[str, dict],
    add_allowed_paths: list[str],
    scope_fragments: list[str],
) -> dict:
    plan = branch_plans.get(worktree, {})
    current_allowed_paths = plan.get("allowed_paths", [])
    next_allowed_paths = unique_preserve_order(current_allowed_paths + add_allowed_paths)
    current_scope = plan.get("proposed_scope")
    next_scope = merge_scope_fragments(current_scope, scope_fragments)
    preview = {}
    if add_allowed_paths:
        preview["allowed_paths"] = {
            "before": current_allowed_paths,
            "after": next_allowed_paths,
        }
    if scope_fragments:
        preview["proposed_scope"] = {
            "before": current_scope,
            "after": next_scope,
        }
    return preview


def validate_non_empty_strings(values: object, field_name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise RuntimeError(f"{field_name} must be a list of strings")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"{field_name} must contain only non-empty strings")
        normalized.append(value)
    return normalized


def apply_tracker_mutations_to_branch_plan(
    tracker: dict,
    branch_plan: dict,
    worktree: str,
    mutations: list[dict],
) -> dict:
    scope_patterns = tracker.setdefault("scope_patterns", {})
    applied_paths = []
    applied_scope_fragments = []
    for mutation in mutations:
        field = mutation.get("field")
        operation = mutation.get("operation")
        if field == "allowed_paths" and operation == "append_unique":
            values = validate_non_empty_strings(mutation.get("values"), "allowed_paths.values")
            if not values:
                continue
            current_allowed_paths = validate_non_empty_strings(
                branch_plan.get("allowed_paths", []),
                f"{worktree}.allowed_paths",
            )
            next_allowed_paths = unique_preserve_order(current_allowed_paths + values)
            branch_plan["allowed_paths"] = next_allowed_paths
            if worktree in scope_patterns:
                current_scope_patterns = validate_non_empty_strings(
                    scope_patterns.get(worktree, []),
                    f"scope_patterns.{worktree}",
                )
                scope_patterns[worktree] = unique_preserve_order(current_scope_patterns + values)
            applied_paths.extend(
                path for path in next_allowed_paths if path not in current_allowed_paths
            )
            continue
        if field == "proposed_scope" and operation == "append_scope_fragments":
            fragments = validate_non_empty_strings(
                mutation.get("values"),
                "proposed_scope.values",
            )
            if not fragments:
                continue
            current_scope = branch_plan.get("proposed_scope")
            if current_scope is not None and not isinstance(current_scope, str):
                raise RuntimeError(f"{worktree}.proposed_scope must be a string when present")
            branch_plan["proposed_scope"] = merge_scope_fragments(current_scope, fragments)
            applied_scope_fragments.extend(fragments)
            continue
        raise RuntimeError(
            f"unsupported tracker mutation for {worktree}: field={field!r} operation={operation!r}"
        )
    return {
        "applied_allowed_paths": unique_preserve_order(applied_paths),
        "applied_scope_fragments": unique_preserve_order(applied_scope_fragments),
    }


def apply_tracker_edit_plan_to_tracker(
    tracker: dict,
    tracker_edit_plan: dict,
    applied_at: str,
) -> dict:
    updated = deepcopy(tracker)
    worker_branches = updated.get("worker_branches")
    if not isinstance(worker_branches, list):
        raise RuntimeError("tracker.worker_branches must be a list")
    branch_plans = {
        item.get("worktree"): item
        for item in worker_branches
        if isinstance(item, dict) and item.get("worktree")
    }
    if len(branch_plans) != len(worker_branches):
        raise RuntimeError("every tracker.worker_branches entry must be an object with worktree")

    applied_edits = []
    for edit in tracker_edit_plan.get("prioritized", []):
        worktree = edit.get("target_worktree")
        if not isinstance(worktree, str) or not worktree:
            raise RuntimeError("tracker_edit_plan action is missing target_worktree")
        branch_plan = branch_plans.get(worktree)
        if branch_plan is None:
            raise RuntimeError(f"tracker is missing worker branch entry for {worktree}")
        mutations = edit.get("tracker_mutations")
        if not isinstance(mutations, list):
            raise RuntimeError(f"tracker_edit_plan tracker_mutations for {worktree} must be a list")
        mutation_result = apply_tracker_mutations_to_branch_plan(
            updated,
            branch_plan,
            worktree,
            mutations,
        )
        applied_edits.append(
            {
                "target_worktree": worktree,
                "paths": edit.get("paths", []),
                "applied_allowed_paths": mutation_result["applied_allowed_paths"],
                "applied_scope_fragments": mutation_result["applied_scope_fragments"],
            }
        )

    choice_required = tracker_edit_plan.get("choice_required", [])
    updated["last_tracker_edit_application"] = {
        "applied_at": applied_at,
        "applied_edit_count": len(applied_edits),
        "applied_worktrees": [item["target_worktree"] for item in applied_edits],
        "applied_path_count": sum(len(item["applied_allowed_paths"]) for item in applied_edits),
        "pending_choice_count": len(choice_required),
        "pending_choice_paths": sorted(
            {
                path
                for item in choice_required
                for path in item.get("paths", [])
                if isinstance(path, str)
            }
        ),
        "applied_edits": applied_edits,
    }
    return updated


def apply_tracker_choice_to_tracker(
    tracker: dict,
    choice_runbook: dict,
    choice_path: str,
    owner_worktree: str,
    applied_at: str,
) -> dict:
    if not isinstance(choice_path, str) or not choice_path.strip():
        raise RuntimeError("--apply-tracker-choice-path must be a non-empty string")
    if not isinstance(owner_worktree, str) or not owner_worktree.strip():
        raise RuntimeError("--apply-tracker-choice-owner must be a non-empty string")

    updated = deepcopy(tracker)
    worker_branches = updated.get("worker_branches")
    if not isinstance(worker_branches, list):
        raise RuntimeError("tracker.worker_branches must be a list")
    branch_plans = {
        item.get("worktree"): item
        for item in worker_branches
        if isinstance(item, dict) and item.get("worktree")
    }

    matching_choice = None
    for choice in choice_runbook.get("prioritized", []):
        if choice_path in choice.get("paths", []):
            matching_choice = choice
            break
    if matching_choice is None:
        raise RuntimeError(f"no pending tracker choice found for path: {choice_path}")

    selected_option = None
    for option in matching_choice.get("candidate_options", []):
        if option.get("target_worktree") == owner_worktree:
            selected_option = option
            break
    if selected_option is None:
        raise RuntimeError(
            f"{owner_worktree} is not a valid owner for pending choice path {choice_path}"
        )

    branch_plan = branch_plans.get(owner_worktree)
    if branch_plan is None:
        raise RuntimeError(f"tracker is missing worker branch entry for {owner_worktree}")

    mutation_result = apply_tracker_mutations_to_branch_plan(
        updated,
        branch_plan,
        owner_worktree,
        selected_option.get("tracker_mutations", []),
    )
    updated["last_tracker_choice_application"] = {
        "applied_at": applied_at,
        "path": choice_path,
        "owner": owner_worktree,
        "recommended_owner": matching_choice.get("recommended_owner"),
        "recommendation_reason": matching_choice.get("recommendation_reason"),
        "applied_allowed_paths": mutation_result["applied_allowed_paths"],
        "applied_scope_fragments": mutation_result["applied_scope_fragments"],
    }
    return updated


def build_tracker_edit_plan(
    decision_queue: dict,
    lane_proposals: dict,
    branch_plans: dict[str, dict],
) -> dict:
    grouped_edits: dict[str, dict] = {}
    choice_required = []

    for proposal in lane_proposals["prioritized"]:
        if proposal["proposal_type"] == "create_dedicated_lane":
            scope_fragments = [humanize_scope_fragment(path) for path in proposal["paths"]]
            choice_required.append(
                {
                    "action_type": "choose_owner_and_expand_lane",
                    "blocking_wave": proposal["blocking_wave"],
                    "paths": proposal["paths"],
                    "scope_fragments": scope_fragments,
                    "candidate_options": [
                        {
                            "target_worktree": worktree,
                            "current_scope": branch_plans.get(worktree, {}).get("proposed_scope"),
                            "add_allowed_paths": [
                                path
                                for path in proposal["paths"]
                                if path not in branch_plans.get(worktree, {}).get("allowed_paths", [])
                            ],
                            "tracker_mutations": build_tracker_mutations(
                                worktree,
                                branch_plans,
                                [
                                    path
                                    for path in proposal["paths"]
                                    if path not in branch_plans.get(worktree, {}).get("allowed_paths", [])
                                ],
                                scope_fragments,
                            ),
                            "patch_preview": build_tracker_patch_preview(
                                worktree,
                                branch_plans,
                                [
                                    path
                                    for path in proposal["paths"]
                                    if path not in branch_plans.get(worktree, {}).get("allowed_paths", [])
                                ],
                                scope_fragments,
                            ),
                        }
                        for worktree in proposal["candidate_workers"]
                    ],
                    "recommended_tracker_action": proposal["recommended_tracker_action"],
                }
            )
            continue

        worktree = proposal["target_worktree"]
        edit = grouped_edits.setdefault(
            worktree,
            {
                "target_worktree": worktree,
                "merge_wave": proposal["merge_wave"],
                "blocking_wave": proposal["blocking_wave"],
                "current_scope": proposal["current_scope"],
                "action_types": set(),
                "paths": [],
                "scope_fragments": [],
                "source_paths": [],
            },
        )
        edit["merge_wave"] = min(
            [wave for wave in [edit["merge_wave"], proposal["merge_wave"]] if wave is not None],
            default=proposal["merge_wave"],
        )
        edit["blocking_wave"] = min(
            [wave for wave in [edit["blocking_wave"], proposal["blocking_wave"]] if wave is not None],
            default=proposal["blocking_wave"],
        )
        edit["action_types"].add(proposal["proposal_type"])
        edit["paths"].extend(proposal["paths"])
        edit["scope_fragments"].extend(humanize_scope_fragment(path) for path in proposal["paths"])
        edit["source_paths"].extend(proposal["source_decisions"])

    for decision in decision_queue["prioritized"]:
        if decision["decision_type"] != "review_existing_lane" or not decision["suggested_owner"]:
            continue
        worktree = decision["suggested_owner"]
        edit = grouped_edits.setdefault(
            worktree,
            {
                "target_worktree": worktree,
                "merge_wave": branch_plans.get(worktree, {}).get("merge_wave"),
                "blocking_wave": decision["blocking_wave"],
                "current_scope": branch_plans.get(worktree, {}).get("proposed_scope"),
                "action_types": set(),
                "paths": [],
                "scope_fragments": [],
                "source_paths": [],
            },
        )
        edit["merge_wave"] = min(
            [wave for wave in [edit["merge_wave"], decision["suggested_owner_merge_wave"]] if wave is not None],
            default=decision["suggested_owner_merge_wave"],
        )
        edit["blocking_wave"] = min(
            [wave for wave in [edit["blocking_wave"], decision["blocking_wave"]] if wave is not None],
            default=decision["blocking_wave"],
        )
        edit["action_types"].add("review_existing_lane")
        edit["paths"].append(decision["path"])
        edit["scope_fragments"].append(humanize_scope_fragment(decision["path"]))
        edit["source_paths"].append(decision["path"])

    actionable = []
    for worktree, edit in sorted(grouped_edits.items()):
        unique_paths = unique_preserve_order(edit["paths"])
        scope_fragments = unique_preserve_order(edit["scope_fragments"])
        current_allowed_paths = branch_plans.get(worktree, {}).get("allowed_paths", [])
        add_allowed_paths = [path for path in unique_paths if path not in current_allowed_paths]
        action_types = sorted(edit["action_types"])
        actionable.append(
            {
                "target_worktree": worktree,
                "merge_wave": edit["merge_wave"],
                "blocking_wave": edit["blocking_wave"],
                "action_types": action_types,
                "current_scope": edit["current_scope"],
                "paths": unique_paths,
                "add_allowed_paths": add_allowed_paths,
                "scope_fragments": scope_fragments,
                "tracker_mutations": build_tracker_mutations(
                    worktree,
                    branch_plans,
                    add_allowed_paths,
                    scope_fragments,
                ),
                "patch_preview": build_tracker_patch_preview(
                    worktree,
                    branch_plans,
                    add_allowed_paths,
                    scope_fragments,
                ),
                "source_paths": unique_preserve_order(edit["source_paths"]),
                "recommended_tracker_action": (
                    f"Update {worktree} tracker scope and allowed_paths to cover {len(unique_paths)} path(s)."
                ),
            }
        )

    prioritized = sorted(
        actionable,
        key=lambda item: (
            item["blocking_wave"] if item["blocking_wave"] is not None else 999,
            min(
                {
                    "retask_as_expanded_lane": 0,
                    "expand_existing_lane": 1,
                    "review_existing_lane": 2,
                }.get(action, 3)
                for action in item["action_types"]
            ),
            -len(item["paths"]),
            item["target_worktree"],
        ),
    )
    return {
        "actionable_edits": actionable,
        "prioritized": prioritized,
        "choice_required": choice_required,
        "summary": {
            "actionable_edit_count": len(actionable),
            "choice_required_count": len(choice_required),
            "target_worktree_count": len(actionable),
            "path_addition_count": sum(len(item["add_allowed_paths"]) for item in actionable),
            "scope_fragment_count": sum(len(item["scope_fragments"]) for item in actionable),
        },
    }


def build_choice_runbook(
    tracker_edit_plan: dict,
    branch_actions: list[dict],
    branch_triage: dict,
    branch_lifecycle: dict,
    path_actions: list[dict],
    branch_plans: dict[str, dict],
) -> dict:
    branch_action_map = {item["worktree"]: item for item in branch_actions}
    triage_map = {item["worktree"]: item for item in branch_triage["branches"]}
    lifecycle_map = {item["worktree"]: item for item in branch_lifecycle["branches"]}
    path_action_map = {item["path"]: item for item in path_actions}

    def candidate_priority_tuple(item: dict) -> tuple:
        return (
            item["merge_wave"] if item["merge_wave"] is not None else 999,
            item["unowned_count"],
            item["lead_only_count"],
            item["release_count"],
            item["incoming_release_count"],
            -item["keep_count"],
            item["active_count"],
            item["target_worktree"],
        )

    prioritized = []
    for choice in tracker_edit_plan.get("choice_required", []):
        choice_paths = validate_non_empty_strings(choice.get("paths"), "choice_required.paths")
        active_workers = sorted(
            {
                worker
                for path in choice_paths
                for worker in path_action_map.get(path, {}).get("active_workers", [])
            }
        )
        candidates = []
        for option in choice.get("candidate_options", []):
            worktree = option.get("target_worktree")
            if not isinstance(worktree, str) or not worktree:
                raise RuntimeError("choice_required candidate option is missing target_worktree")
            branch_action = branch_action_map.get(worktree, {})
            triage = triage_map.get(worktree, {})
            lifecycle = lifecycle_map.get(worktree, {})
            merge_wave = branch_plans.get(worktree, {}).get("merge_wave")
            predicted_releasers = sorted(worker for worker in active_workers if worker != worktree)
            candidate = {
                "target_worktree": worktree,
                "merge_wave": merge_wave,
                "disposition": triage.get("disposition"),
                "lifecycle_state": lifecycle.get("lifecycle_state"),
                "active_count": len(branch_action.get("keep_paths", []))
                + len(branch_action.get("release_paths", []))
                + len(branch_action.get("ambiguous_paths", []))
                + len(branch_action.get("unowned_paths", [])),
                "keep_count": len(branch_action.get("keep_paths", [])),
                "release_count": len(branch_action.get("release_paths", [])),
                "unowned_count": len(branch_action.get("unowned_paths", [])),
                "lead_only_count": len(branch_action.get("lead_only_hits", [])),
                "incoming_release_count": len(triage.get("awaiting_incoming_releases", [])),
                "incoming_release_from": triage.get("awaiting_incoming_releases", []),
                "predicted_releasers": predicted_releasers,
                "predicted_releaser_count": len(predicted_releasers),
                "tracker_mutations": option.get("tracker_mutations", []),
                "patch_preview": option.get("patch_preview", {}),
                "current_scope": option.get("current_scope"),
                "add_allowed_paths": option.get("add_allowed_paths", []),
            }
            candidate["priority"] = candidate_priority_tuple(candidate)
            candidates.append(candidate)

        ranked_candidates = sorted(candidates, key=candidate_priority_tuple)
        recommended = ranked_candidates[0] if ranked_candidates else None
        recommendation_reason = None
        if len(ranked_candidates) >= 2:
            runner_up = ranked_candidates[1]
            recommendation_reason = (
                f"{recommended['target_worktree']} wins on wave/order pressure "
                f"(wave={recommended['merge_wave']}, unowned={recommended['unowned_count']}, "
                f"release={recommended['release_count']}, incoming={recommended['incoming_release_count']}) "
                f"vs {runner_up['target_worktree']} "
                f"(wave={runner_up['merge_wave']}, unowned={runner_up['unowned_count']}, "
                f"release={runner_up['release_count']}, incoming={runner_up['incoming_release_count']})."
            )
        prioritized.append(
            {
                "action_type": choice.get("action_type"),
                "blocking_wave": choice.get("blocking_wave"),
                "paths": choice_paths,
                "active_workers": active_workers,
                "scope_fragments": choice.get("scope_fragments", []),
                "candidate_options": ranked_candidates,
                "recommended_owner": recommended["target_worktree"] if recommended else None,
                "recommendation_reason": recommendation_reason,
                "recommended_tracker_action": choice.get("recommended_tracker_action"),
            }
        )

    prioritized.sort(
        key=lambda item: (
            item["blocking_wave"] if item["blocking_wave"] is not None else 999,
            len(item["paths"]),
            item["paths"][0] if item["paths"] else "",
        )
    )
    return {
        "choices": prioritized,
        "prioritized": prioritized,
        "summary": {
            "pending_choice_count": len(prioritized),
            "recommended_choice_count": sum(1 for item in prioritized if item["recommended_owner"]),
            "candidate_option_count": sum(len(item["candidate_options"]) for item in prioritized),
        },
    }


def build_cleanup_impact_queue(
    branch_actions: list[dict],
    owner_runbook: list[dict],
    conflict_runbook: dict,
    branch_plans: dict[str, dict],
) -> dict:
    owner_runbook_map = {item["owner"]: item for item in owner_runbook}
    conflict_map = {item["path"]: item for item in conflict_runbook["clusters"]}
    branch_wave_map = {
        worktree: plan.get("merge_wave")
        for worktree, plan in branch_plans.items()
    }
    items = []
    for action in branch_actions:
        if not (action["release_paths"] or action["ambiguous_paths"] or action["unowned_paths"]):
            continue
        owner_wait_reduction = sorted(
            owner
            for owner, runbook in owner_runbook_map.items()
            if action["worktree"] in runbook.get("awaiting_release_from", [])
        )
        owners_relieved = sorted(
            {
                item["owner"]
                for item in action["release_paths"]
                if item.get("owner")
            }
        )
        owner_waves = sorted(
            {
                0 if owner == "agent-01" else branch_wave_map.get(owner)
                for owner in owners_relieved + owner_wait_reduction
                if owner == "agent-01" or branch_wave_map.get(owner) is not None
            }
        )
        overlap_paths = []
        critical_overlap_paths = []
        high_overlap_paths = []
        blocking_waves = set(owner_waves)
        for release in action["release_paths"]:
            conflict = conflict_map.get(release["path"])
            if not conflict:
                continue
            overlap_paths.append(release["path"])
            if conflict["severity"] == "critical":
                critical_overlap_paths.append(release["path"])
            elif conflict["severity"] == "high":
                high_overlap_paths.append(release["path"])
            if conflict.get("blocking_wave") is not None:
                blocking_waves.add(conflict["blocking_wave"])

        blocking_wave_list = sorted(blocking_waves)
        lead_only_release_paths = sorted(
            {
                item["path"]
                for item in action["release_paths"]
                if item.get("owner") == "agent-01"
            }
        )
        item = {
            "worktree": action["worktree"],
            "branch": action["branch"],
            "merge_wave": branch_wave_map.get(action["worktree"]),
            "blocking_waves_relieved": blocking_wave_list,
            "earliest_blocking_wave_relieved": (
                blocking_wave_list[0] if blocking_wave_list else None
            ),
            "owners_relieved": owners_relieved,
            "owner_wait_reduction": owner_wait_reduction,
            "owner_wait_reduction_count": len(owner_wait_reduction),
            "release_path_count": len(action["release_paths"]),
            "lead_only_release_paths": lead_only_release_paths,
            "lead_only_release_count": len(lead_only_release_paths),
            "overlap_paths": sorted(set(overlap_paths)),
            "overlap_path_count": len(set(overlap_paths)),
            "critical_overlap_paths": sorted(set(critical_overlap_paths)),
            "critical_overlap_count": len(set(critical_overlap_paths)),
            "high_overlap_paths": sorted(set(high_overlap_paths)),
            "high_overlap_count": len(set(high_overlap_paths)),
            "ambiguous_path_count": len(action["ambiguous_paths"]),
            "unowned_path_count": len(action["unowned_paths"]),
            "cleanup_commands": action["cleanup_commands"],
            "post_cleanup_commands": action["post_cleanup_commands"],
            "command_batch": action["command_batch"],
        }
        item["priority"] = (
            item["earliest_blocking_wave_relieved"]
            if item["earliest_blocking_wave_relieved"] is not None
            else 999,
            -item["owner_wait_reduction_count"],
            -item["lead_only_release_count"],
            -item["critical_overlap_count"],
            -item["high_overlap_count"],
            -item["overlap_path_count"],
            -item["release_path_count"],
            item["worktree"],
        )
        item["recommended_reason"] = (
            f"Relieves {item['owner_wait_reduction_count']} owner queues, "
            f"{item['lead_only_release_count']} lead-only paths, "
            f"and {item['critical_overlap_count']} critical overlap paths."
        )
        items.append(item)

    prioritized = sorted(items, key=lambda item: item["priority"])
    return {
        "branches": items,
        "prioritized": prioritized,
        "summary": {
            "branch_count": len(items),
            "wave_zero_branch_count": sum(
                1 for item in items if item["earliest_blocking_wave_relieved"] == 0
            ),
            "branches_reducing_owner_waits": sum(
                1 for item in items if item["owner_wait_reduction_count"] > 0
            ),
            "branches_releasing_lead_only": sum(
                1 for item in items if item["lead_only_release_count"] > 0
            ),
            "branches_reducing_critical_overlaps": sum(
                1 for item in items if item["critical_overlap_count"] > 0
            ),
            "top_candidate": prioritized[0]["worktree"] if prioritized else None,
        },
    }


def simulate_cleanup_projection(
    path_actions: list[dict],
    branch_actions: list[dict],
    owner_runbook: list[dict],
    conflict_runbook: dict,
    selected_worktrees: list[str],
) -> dict:
    selected = set(selected_worktrees)
    severity_by_path = {
        item["path"]: item["severity"]
        for item in conflict_runbook.get("clusters", [])
    }
    projected_paths = []
    for item in path_actions:
        projected_active_workers = [
            worktree
            for worktree in item["active_workers"]
            if not (worktree in selected and worktree in item["release_from"])
        ]
        projected_release_from = [
            worktree
            for worktree in item["release_from"]
            if worktree in projected_active_workers
        ]
        projected_paths.append(
            {
                "path": item["path"],
                "owner": item["owner"],
                "owner_source": item["owner_source"],
                "active_workers": projected_active_workers,
                "release_from": projected_release_from,
            }
        )

    overlaps = [item for item in projected_paths if len(item["active_workers"]) > 1]
    owners_waiting = []
    lead_waiting_on = []
    for owner_item in owner_runbook:
        owner = owner_item["owner"]
        waiting_on = sorted(
            {
                worktree
                for owned_path in owner_item["owned_paths"]
                for path_item in projected_paths
                if path_item["path"] == owned_path["path"]
                for worktree in path_item["release_from"]
            }
        )
        if waiting_on:
            owners_waiting.append({"owner": owner, "waiting_on": waiting_on})
        if owner == "agent-01":
            lead_waiting_on = waiting_on

    remaining_branch_work = []
    projected_branch_states = []
    for action in branch_actions:
        cleanup_applied = action["worktree"] in selected
        projected_state = {
            "worktree": action["worktree"],
            "cleanup_applied": cleanup_applied,
            "release_path_count": 0 if cleanup_applied else len(action["release_paths"]),
            "lead_only_count": 0 if cleanup_applied else len(action["lead_only_hits"]),
            "ambiguous_path_count": len(action["ambiguous_paths"]),
            "unowned_path_count": len(action["unowned_paths"]),
        }
        projected_state["still_blocked"] = any(
            projected_state[key] > 0
            for key in [
                "release_path_count",
                "lead_only_count",
                "ambiguous_path_count",
                "unowned_path_count",
            ]
        )
        projected_branch_states.append(projected_state)
        if projected_state["still_blocked"]:
            remaining_branch_work.append(projected_state)

    return {
        "selected_worktrees": selected_worktrees,
        "paths_with_overlaps": len(overlaps),
        "ownerless_overlap_count": sum(1 for item in overlaps if item["owner"] is None),
        "critical_overlap_count": sum(
            1 for item in overlaps if severity_by_path.get(item["path"]) == "critical"
        ),
        "high_overlap_count": sum(
            1 for item in overlaps if severity_by_path.get(item["path"]) == "high"
        ),
        "owners_waiting_on_releases": len(owners_waiting),
        "lead_waiting_on_release_count": len(lead_waiting_on),
        "lead_waiting_on": lead_waiting_on,
        "owner_waits": owners_waiting,
        "projected_branch_states": projected_branch_states,
        "branches_needing_handoff": sum(
            1
            for item in remaining_branch_work
            if item["release_path_count"] or item["ambiguous_path_count"] or item["unowned_path_count"]
        ),
        "branches_blocked_by_lead_only": sum(
            1 for item in remaining_branch_work if item["lead_only_count"] > 0
        ),
    }


def build_cleanup_forecast(
    cleanup_impact_queue: dict,
    path_actions: list[dict],
    branch_actions: list[dict],
    owner_runbook: list[dict],
    conflict_runbook: dict,
) -> dict:
    prioritized = cleanup_impact_queue.get("prioritized", [])
    sequence = [item["worktree"] for item in prioritized]
    baseline = simulate_cleanup_projection(
        path_actions,
        branch_actions,
        owner_runbook,
        conflict_runbook,
        [],
    )
    if not sequence:
        return {
            "summary": {
                "sequence_length": 0,
                "milestone_count": 0,
            },
            "baseline": baseline,
            "milestones": [],
        }

    milestone_points = unique_preserve_order(
        [
            point
            for point in [1, 3, 5, 10, 18, len(sequence)]
            if point <= len(sequence)
        ]
    )
    milestones = []
    for point in milestone_points:
        applied = sequence[:point]
        projected = simulate_cleanup_projection(
            path_actions,
            branch_actions,
            owner_runbook,
            conflict_runbook,
            applied,
        )
        milestones.append(
            {
                "batch_size": point,
                "applied_worktrees": applied,
                "top_worktree": applied[-1],
                "projected": projected,
                "delta": {
                    "paths_with_overlaps_reduced_by": baseline["paths_with_overlaps"] - projected["paths_with_overlaps"],
                    "critical_overlap_count_reduced_by": baseline["critical_overlap_count"]
                    - projected["critical_overlap_count"],
                    "high_overlap_count_reduced_by": baseline["high_overlap_count"]
                    - projected["high_overlap_count"],
                    "owners_waiting_on_releases_reduced_by": baseline["owners_waiting_on_releases"]
                    - projected["owners_waiting_on_releases"],
                    "lead_waiting_on_release_count_reduced_by": baseline["lead_waiting_on_release_count"]
                    - projected["lead_waiting_on_release_count"],
                    "branches_needing_handoff_reduced_by": baseline["branches_needing_handoff"]
                    - projected["branches_needing_handoff"],
                    "branches_blocked_by_lead_only_reduced_by": baseline["branches_blocked_by_lead_only"]
                    - projected["branches_blocked_by_lead_only"],
                },
            }
        )

    return {
        "summary": {
            "sequence_length": len(sequence),
            "milestone_count": len(milestones),
            "top_candidate": sequence[0],
            "largest_modeled_batch": milestones[-1]["batch_size"] if milestones else 0,
        },
        "baseline": baseline,
        "milestones": milestones,
    }


def build_wave_unlock_forecast(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "waves": []}

    remediation = payload["tracker_validation"]["remediation"]
    cleanup_sequence = [
        item["worktree"] for item in remediation["cleanup_impact_queue"].get("prioritized", [])
    ]
    checkpoints = [0] + list(range(1, len(cleanup_sequence) + 1))
    simulations = {}
    for checkpoint in checkpoints:
        applied = cleanup_sequence[:checkpoint]
        simulations[checkpoint] = simulate_cleanup_projection(
            remediation["path_actions"],
            remediation["branch_actions"],
            remediation["owner_runbook"],
            remediation["conflict_runbook"],
            applied,
        )

    waves = []
    lead_drift_present = bool(payload["summary"]["lead_vs_main_file_drift"])
    for wave in tracker.get("merge_order", []):
        wave_no = wave.get("wave")
        owner = wave.get("owner")
        wave_workers = list(wave.get("workers", []))
        if owner and owner != "agent-01" and owner not in wave_workers:
            wave_workers.append(owner)

        cleanup_ready_after = None
        cleanup_ready_after_worktree = None
        wave_checkpoints = []
        for checkpoint in checkpoints:
            simulation = simulations[checkpoint]
            owner_wait_map = {
                item["owner"]: item["waiting_on"]
                for item in simulation.get("owner_waits", [])
            }
            branch_state_map = {
                item["worktree"]: item
                for item in simulation.get("projected_branch_states", [])
            }
            overlapping_wave_paths = sorted(
                item["path"]
                for item in simulation.get("projected_paths", [])
                if item["path"] in wave.get("paths", []) and len(item["active_workers"]) > 1
            )
            blocked_workers = sorted(
                worktree
                for worktree in wave_workers
                if branch_state_map.get(worktree, {}).get("still_blocked")
            )
            owner_waiting_on = owner_wait_map.get(owner, [])
            manual_blockers = []
            if owner == "agent-01" and lead_drift_present:
                manual_blockers.append("lead_drift")
            cleanup_ready = not blocked_workers and not owner_waiting_on and not overlapping_wave_paths
            if cleanup_ready and cleanup_ready_after is None:
                cleanup_ready_after = checkpoint
                if checkpoint > 0:
                    cleanup_ready_after_worktree = cleanup_sequence[checkpoint - 1]
            wave_checkpoints.append(
                {
                    "batch_size": checkpoint,
                    "applied_worktrees": cleanup_sequence[:checkpoint],
                    "cleanup_ready": cleanup_ready,
                    "fully_ready": cleanup_ready and not manual_blockers,
                    "blocked_workers": blocked_workers,
                    "owner_waiting_on": owner_waiting_on,
                    "overlapping_wave_paths": overlapping_wave_paths,
                    "manual_blockers": manual_blockers,
                }
            )

        waves.append(
            {
                "wave": wave_no,
                "owner": owner,
                "goal": wave.get("goal"),
                "paths": wave.get("paths", []),
                "workers": wave_workers,
                "cleanup_ready_after_batch": cleanup_ready_after,
                "cleanup_ready_after_worktree": cleanup_ready_after_worktree,
                "fully_ready_after_batch": next(
                    (item["batch_size"] for item in wave_checkpoints if item["fully_ready"]),
                    None,
                ),
                "currently_manual_blockers": (
                    ["lead_drift"] if owner == "agent-01" and lead_drift_present else []
                ),
                "checkpoints": wave_checkpoints,
            }
        )

    return {
        "summary": {
            "wave_count": len(waves),
            "cleanup_ready_wave_count": sum(
                1 for item in waves if item["cleanup_ready_after_batch"] is not None
            ),
            "fully_ready_wave_count": sum(
                1 for item in waves if item["fully_ready_after_batch"] is not None
            ),
            "first_cleanup_ready_wave": next(
                (item["wave"] for item in waves if item["cleanup_ready_after_batch"] is not None),
                None,
            ),
            "first_fully_ready_wave": next(
                (item["wave"] for item in waves if item["fully_ready_after_batch"] is not None),
                None,
            ),
        },
        "waves": waves,
    }


def build_wave_critical_path(wave_unlock_forecast: dict) -> dict:
    prioritized = []
    for wave in wave_unlock_forecast.get("waves", []):
        checkpoints = wave.get("checkpoints", [])
        current = checkpoints[0] if checkpoints else {}
        cleanup_ready_checkpoint = next(
            (item for item in checkpoints if item.get("cleanup_ready")),
            None,
        )
        fully_ready_checkpoint = next(
            (item for item in checkpoints if item.get("fully_ready")),
            None,
        )
        cleanup_prefix = cleanup_ready_checkpoint.get("applied_worktrees", []) if cleanup_ready_checkpoint else []
        fully_ready_prefix = fully_ready_checkpoint.get("applied_worktrees", []) if fully_ready_checkpoint else []
        status = "blocked"
        if fully_ready_checkpoint:
            status = "fully_ready_reachable"
        elif cleanup_ready_checkpoint:
            status = "cleanup_ready_with_manual_gate"
        prioritized.append(
            {
                "wave": wave.get("wave"),
                "owner": wave.get("owner"),
                "goal": wave.get("goal"),
                "status": status,
                "cleanup_ready_after_batch": wave.get("cleanup_ready_after_batch"),
                "cleanup_ready_after_worktree": wave.get("cleanup_ready_after_worktree"),
                "cleanup_prefix": cleanup_prefix,
                "cleanup_prefix_length": len(cleanup_prefix),
                "fully_ready_after_batch": wave.get("fully_ready_after_batch"),
                "fully_ready_after_worktree": (
                    fully_ready_prefix[-1] if fully_ready_prefix else None
                ),
                "fully_ready_prefix": fully_ready_prefix,
                "fully_ready_prefix_length": len(fully_ready_prefix),
                "current_blockers": {
                    "blocked_workers": current.get("blocked_workers", []),
                    "owner_waiting_on": current.get("owner_waiting_on", []),
                    "overlapping_wave_paths": current.get("overlapping_wave_paths", []),
                    "manual_blockers": current.get("manual_blockers", []),
                },
                "post_cleanup_blockers": (
                    {
                        "blocked_workers": cleanup_ready_checkpoint.get("blocked_workers", []),
                        "owner_waiting_on": cleanup_ready_checkpoint.get("owner_waiting_on", []),
                        "overlapping_wave_paths": cleanup_ready_checkpoint.get("overlapping_wave_paths", []),
                        "manual_blockers": cleanup_ready_checkpoint.get("manual_blockers", []),
                    }
                    if cleanup_ready_checkpoint
                    else None
                ),
                "recommended_next_action": (
                    "Run the cleanup prefix, then clear the remaining manual gate."
                    if cleanup_ready_checkpoint and not fully_ready_checkpoint
                    else (
                        "Run the cleanup prefix to make the wave fully ready."
                        if fully_ready_checkpoint
                        else "Continue down the ranked cleanup queue; this wave is not yet projected to unlock."
                    )
                ),
            }
        )

    prioritized.sort(
        key=lambda item: (
            item["cleanup_ready_after_batch"]
            if item["cleanup_ready_after_batch"] is not None
            else 999,
            item["wave"] if item["wave"] is not None else 999,
        )
    )
    return {
        "waves": prioritized,
        "prioritized": prioritized,
        "summary": {
            "wave_count": len(prioritized),
            "cleanup_ready_with_manual_gate_count": sum(
                1 for item in prioritized if item["status"] == "cleanup_ready_with_manual_gate"
            ),
            "fully_ready_reachable_count": sum(
                1 for item in prioritized if item["status"] == "fully_ready_reachable"
            ),
            "blocked_count": sum(1 for item in prioritized if item["status"] == "blocked"),
            "first_manual_gate_wave": next(
                (
                    item["wave"]
                    for item in prioritized
                    if item["status"] == "cleanup_ready_with_manual_gate"
                ),
                None,
            ),
            "first_fully_ready_wave": next(
                (
                    item["wave"]
                    for item in prioritized
                    if item["status"] == "fully_ready_reachable"
                ),
                None,
            ),
        },
    }


def build_unlock_ladder(wave_critical_path: dict) -> dict:
    waves = wave_critical_path.get("prioritized", [])
    event_points = sorted(
        {
            batch_size
            for item in waves
            for batch_size in [
                item.get("cleanup_ready_after_batch"),
                item.get("fully_ready_after_batch"),
            ]
            if batch_size is not None
        }
    )
    stages = []
    seen_cleanup_ready: set[int] = set()
    seen_fully_ready: set[int] = set()
    for batch_size in event_points:
        newly_cleanup_ready = [
            item["wave"]
            for item in waves
            if item.get("cleanup_ready_after_batch") == batch_size
            and item["wave"] not in seen_cleanup_ready
        ]
        newly_fully_ready = [
            item["wave"]
            for item in waves
            if item.get("fully_ready_after_batch") == batch_size
            and item["wave"] not in seen_fully_ready
        ]
        for wave in newly_cleanup_ready:
            seen_cleanup_ready.add(wave)
        for wave in newly_fully_ready:
            seen_fully_ready.add(wave)

        cleanup_ready_with_manual_gates = [
            item["wave"]
            for item in waves
            if item.get("cleanup_ready_after_batch") is not None
            and item["cleanup_ready_after_batch"] <= batch_size
            and item.get("fully_ready_after_batch") is None
        ]
        cleanup_ready_not_fully_ready = [
            item["wave"]
            for item in waves
            if item.get("cleanup_ready_after_batch") is not None
            and item["cleanup_ready_after_batch"] <= batch_size
            and (
                item.get("fully_ready_after_batch") is None
                or item["fully_ready_after_batch"] > batch_size
            )
        ]
        fully_ready_waves = [
            item["wave"]
            for item in waves
            if item.get("fully_ready_after_batch") is not None
            and item["fully_ready_after_batch"] <= batch_size
        ]
        triggering_worktrees = sorted(
            {
                worktree
                for item in waves
                for field in [
                    "cleanup_ready_after_worktree",
                    "fully_ready_after_worktree",
                ]
                if (
                    item.get(
                        "cleanup_ready_after_batch"
                        if field == "cleanup_ready_after_worktree"
                        else "fully_ready_after_batch"
                    )
                    == batch_size
                )
                for worktree in [item.get(field)]
                if worktree
            }
        )
        stages.append(
            {
                "batch_size": batch_size,
                "triggering_worktrees": triggering_worktrees,
                "newly_cleanup_ready_waves": newly_cleanup_ready,
                "newly_fully_ready_waves": newly_fully_ready,
                "cleanup_ready_not_fully_ready_waves": cleanup_ready_not_fully_ready,
                "cleanup_ready_with_manual_gates": cleanup_ready_with_manual_gates,
                "fully_ready_waves": fully_ready_waves,
            }
        )

    return {
        "stages": stages,
        "summary": {
            "stage_count": len(stages),
            "first_stage_batch": stages[0]["batch_size"] if stages else None,
            "last_stage_batch": stages[-1]["batch_size"] if stages else None,
            "first_cleanup_ready_stage": next(
                (
                    stage["batch_size"]
                    for stage in stages
                    if stage["newly_cleanup_ready_waves"]
                ),
                None,
            ),
            "first_fully_ready_stage": next(
                (
                    stage["batch_size"]
                    for stage in stages
                    if stage["newly_fully_ready_waves"]
                ),
                None,
            ),
        },
    }


def build_stage_command_runbook(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "stages": [], "prioritized": []}

    remediation = payload["tracker_validation"]["remediation"]
    unlock_ladder = payload["tracker_validation"]["unlock_ladder"]
    wave_command_runbook = payload["tracker_validation"]["wave_command_runbook"]
    branch_batch_map = {
        item["worktree"]: item for item in remediation.get("branch_batches", [])
    }
    wave_runbook_map = {
        item["wave"]: item for item in wave_command_runbook.get("prioritized", [])
    }
    cleanup_sequence = [
        item["worktree"]
        for item in remediation.get("cleanup_impact_queue", {}).get("prioritized", [])
    ]

    stages = []
    previous_batch_size = 0
    cumulative_command_count = 0
    for stage_index, stage in enumerate(unlock_ladder.get("stages", []), start=1):
        batch_size = stage.get("batch_size", 0)
        incremental_worktrees = cleanup_sequence[previous_batch_size:batch_size]
        cumulative_worktrees = cleanup_sequence[:batch_size]
        incremental_batches = []
        incremental_command_count = 0

        for sequence, worktree in enumerate(incremental_worktrees, start=1):
            batch = branch_batch_map.get(worktree)
            if not batch:
                continue
            command_batch = batch.get("command_batch", [])
            incremental_batches.append(
                {
                    "sequence": sequence,
                    "worktree": worktree,
                    "branch": batch.get("branch"),
                    "release_count": batch.get("release_count", 0),
                    "lead_only_count": batch.get("lead_only_count", 0),
                    "unowned_count": batch.get("unowned_count", 0),
                    "command_count": len(command_batch),
                    "command_batch": command_batch,
                }
            )
            incremental_command_count += len(command_batch)

        cumulative_command_count += incremental_command_count
        newly_ready_waves = sorted(
            set(stage.get("newly_cleanup_ready_waves", []))
            | set(stage.get("newly_fully_ready_waves", []))
        )
        wave_effects = []
        remaining_manual_gates = []
        for wave_no in newly_ready_waves:
            runbook = wave_runbook_map.get(wave_no, {})
            post_cleanup_manual_steps = runbook.get("post_cleanup_manual_steps", [])
            wave_effects.append(
                {
                    "wave": wave_no,
                    "owner": runbook.get("owner"),
                    "goal": runbook.get("goal"),
                    "status": (
                        "fully_ready"
                        if wave_no in stage.get("fully_ready_waves", [])
                        else "cleanup_ready_with_manual_gate"
                    ),
                    "newly_cleanup_ready": wave_no in stage.get("newly_cleanup_ready_waves", []),
                    "newly_fully_ready": wave_no in stage.get("newly_fully_ready_waves", []),
                    "cleanup_ready_after_batch": runbook.get("cleanup_ready_after_batch"),
                    "cleanup_ready_after_worktree": runbook.get("cleanup_ready_after_worktree"),
                    "post_cleanup_manual_steps": post_cleanup_manual_steps,
                }
            )

        for wave_no in stage.get("cleanup_ready_with_manual_gates", []):
            runbook = wave_runbook_map.get(wave_no, {})
            for step in runbook.get("post_cleanup_manual_steps", []):
                remaining_manual_gates.append(
                    {
                        "wave": wave_no,
                        "owner": runbook.get("owner"),
                        "goal": runbook.get("goal"),
                        "blocker": step.get("blocker"),
                        "worktree": step.get("worktree"),
                        "operation_type": step.get("operation_type"),
                        "reason": step.get("reason"),
                        "paths": step.get("paths", []),
                        "blocking_local_changes": step.get("blocking_local_changes", []),
                        "command_batch": step.get("command_batch", []),
                        "command_count": step.get("command_count", 0),
                    }
                )

        stages.append(
            {
                "stage": stage_index,
                "batch_size": batch_size,
                "previous_batch_size": previous_batch_size,
                "incremental_batch_size": max(batch_size - previous_batch_size, 0),
                "triggering_worktrees": stage.get("triggering_worktrees", []),
                "incremental_worktrees": incremental_worktrees,
                "cumulative_worktrees": cumulative_worktrees,
                "incremental_batches": incremental_batches,
                "incremental_command_count": incremental_command_count,
                "cumulative_command_count": cumulative_command_count,
                "newly_cleanup_ready_waves": stage.get("newly_cleanup_ready_waves", []),
                "newly_fully_ready_waves": stage.get("newly_fully_ready_waves", []),
                "cleanup_ready_not_fully_ready_waves": stage.get(
                    "cleanup_ready_not_fully_ready_waves", []
                ),
                "cleanup_ready_with_manual_gates": stage.get(
                    "cleanup_ready_with_manual_gates", []
                ),
                "fully_ready_waves": stage.get("fully_ready_waves", []),
                "wave_effects": wave_effects,
                "remaining_manual_gates": remaining_manual_gates,
                "remaining_manual_gate_count": len(remaining_manual_gates),
                "recommended_next_action": (
                    "Run the incremental cleanup batches, then clear the remaining manual gates."
                    if remaining_manual_gates
                    else "Run the incremental cleanup batches; this stage produces fully ready waves."
                ),
            }
        )
        previous_batch_size = batch_size

    return {
        "stages": stages,
        "prioritized": stages,
        "summary": {
            "stage_count": len(stages),
            "total_incremental_batches": sum(
                item.get("incremental_batch_size", 0) for item in stages
            ),
            "total_incremental_commands": sum(
                item.get("incremental_command_count", 0) for item in stages
            ),
            "first_stage_batch": stages[0]["batch_size"] if stages else None,
            "first_manual_gate_stage": next(
                (
                    item["stage"]
                    for item in stages
                    if item.get("remaining_manual_gate_count", 0) > 0
                ),
                None,
            ),
            "first_fully_ready_stage": next(
                (
                    item["stage"]
                    for item in stages
                    if item.get("newly_fully_ready_waves")
                ),
                None,
            ),
        },
    }


def build_unlock_frontier(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "next_stage": None, "next_fully_ready_stage": None, "next_manual_gate_stage": None}

    validation_summary = payload["tracker_validation"]["summary"]
    execution_summary = payload["tracker_validation"]["execution_queue"]["summary"]
    stage_runbook = payload["tracker_validation"]["stage_command_runbook"]
    stages = stage_runbook.get("prioritized", [])

    def compact_stage(stage: dict | None) -> dict | None:
        if not stage:
            return None
        return {
            "stage": stage.get("stage"),
            "batch_size": stage.get("batch_size"),
            "incremental_batch_size": stage.get("incremental_batch_size"),
            "incremental_command_count": stage.get("incremental_command_count"),
            "triggering_worktrees": stage.get("triggering_worktrees", []),
            "incremental_worktrees": stage.get("incremental_worktrees", []),
            "newly_cleanup_ready_waves": stage.get("newly_cleanup_ready_waves", []),
            "newly_fully_ready_waves": stage.get("newly_fully_ready_waves", []),
            "cleanup_ready_with_manual_gates": stage.get(
                "cleanup_ready_with_manual_gates", []
            ),
            "remaining_manual_gate_count": stage.get("remaining_manual_gate_count", 0),
            "remaining_manual_gates": stage.get("remaining_manual_gates", []),
            "recommended_next_action": stage.get("recommended_next_action"),
        }

    next_stage = stages[0] if stages else None
    next_fully_ready_stage = next(
        (stage for stage in stages if stage.get("newly_fully_ready_waves")),
        None,
    )
    next_manual_gate_stage = next(
        (stage for stage in stages if stage.get("remaining_manual_gate_count", 0) > 0),
        None,
    )
    progression = [
        {
            "stage": stage.get("stage"),
            "batch_size": stage.get("batch_size"),
            "incremental_worktree_count": len(stage.get("incremental_worktrees", [])),
            "incremental_command_count": stage.get("incremental_command_count", 0),
            "triggering_worktrees": stage.get("triggering_worktrees", []),
            "newly_cleanup_ready_waves": stage.get("newly_cleanup_ready_waves", []),
            "newly_fully_ready_waves": stage.get("newly_fully_ready_waves", []),
            "remaining_manual_gate_count": stage.get("remaining_manual_gate_count", 0),
        }
        for stage in stages
    ]

    return {
        "summary": {
            "tracked_workers": validation_summary.get("tracked_workers"),
            "violation_count": validation_summary.get("violation_count"),
            "dirty_worker_count": validation_summary.get("dirty_worker_count"),
            "next_stage": next_stage.get("stage") if next_stage else None,
            "next_stage_worktree_count": len(next_stage.get("incremental_worktrees", []))
            if next_stage
            else 0,
            "next_stage_command_count": next_stage.get("incremental_command_count", 0)
            if next_stage
            else 0,
            "next_fully_ready_stage": next_fully_ready_stage.get("stage")
            if next_fully_ready_stage
            else None,
            "next_manual_gate_stage": next_manual_gate_stage.get("stage")
            if next_manual_gate_stage
            else None,
            "next_manual_gate_blocker_count": next_manual_gate_stage.get(
                "remaining_manual_gate_count", 0
            )
            if next_manual_gate_stage
            else 0,
            "execution_operation_count": execution_summary.get("operation_count"),
            "execution_manual_operation_count": execution_summary.get(
                "manual_operation_count"
            ),
        },
        "next_stage": compact_stage(next_stage),
        "next_fully_ready_stage": compact_stage(next_fully_ready_stage),
        "next_manual_gate_stage": compact_stage(next_manual_gate_stage),
        "progression": progression,
    }


def build_next_unlock_batch(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "current_stage": None, "follow_up_stage": None}

    unlock_frontier = payload["tracker_validation"]["unlock_frontier"]
    stage_runbook = payload["tracker_validation"]["stage_command_runbook"]
    stages = stage_runbook.get("prioritized", [])
    next_stage_number = unlock_frontier.get("summary", {}).get("next_stage")
    next_stage = next(
        (stage for stage in stages if stage.get("stage") == next_stage_number),
        None,
    )
    follow_up_stage = None
    if next_stage:
        follow_up_stage = next(
            (
                stage
                for stage in stages
                if stage.get("stage", 0) > next_stage.get("stage", 0)
            ),
            None,
        )

    def summarize_wave_effects(stage: dict | None) -> list[dict]:
        if not stage:
            return []
        return [
            {
                "wave": item.get("wave"),
                "owner": item.get("owner"),
                "goal": item.get("goal"),
                "status": item.get("status"),
                "newly_cleanup_ready": item.get("newly_cleanup_ready"),
                "newly_fully_ready": item.get("newly_fully_ready"),
            }
            for item in stage.get("wave_effects", [])
        ]

    def flatten_stage(stage: dict | None) -> dict | None:
        if not stage:
            return None
        command_batch = []
        for batch in stage.get("incremental_batches", []):
            command_batch.extend(batch.get("command_batch", []))
        command_batch = unique_preserve_order(command_batch)
        return {
            "stage": stage.get("stage"),
            "batch_size": stage.get("batch_size"),
            "incremental_batch_size": stage.get("incremental_batch_size"),
            "incremental_worktrees": stage.get("incremental_worktrees", []),
            "triggering_worktrees": stage.get("triggering_worktrees", []),
            "incremental_batches": stage.get("incremental_batches", []),
            "incremental_command_count": stage.get("incremental_command_count", 0),
            "flattened_command_batch": command_batch,
            "flattened_command_count": len(command_batch),
            "wave_effects": summarize_wave_effects(stage),
            "remaining_manual_gates": stage.get("remaining_manual_gates", []),
            "remaining_manual_gate_count": stage.get("remaining_manual_gate_count", 0),
            "recommended_next_action": stage.get("recommended_next_action"),
        }

    current_stage = flatten_stage(next_stage)
    follow_up = flatten_stage(follow_up_stage)
    return {
        "summary": {
            "current_stage": current_stage.get("stage") if current_stage else None,
            "current_stage_worktree_count": len(current_stage.get("incremental_worktrees", []))
            if current_stage
            else 0,
            "current_stage_flattened_command_count": current_stage.get(
                "flattened_command_count", 0
            )
            if current_stage
            else 0,
            "current_stage_newly_ready_waves": [
                item["wave"] for item in current_stage.get("wave_effects", [])
            ]
            if current_stage
            else [],
            "follow_up_stage": follow_up.get("stage") if follow_up else None,
            "follow_up_manual_gate_count": follow_up.get("remaining_manual_gate_count", 0)
            if follow_up
            else 0,
        },
        "current_stage": current_stage,
        "follow_up_stage": follow_up,
    }


def build_next_unlock_brief(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "brief": None}

    validation_summary = payload["tracker_validation"]["summary"]
    next_unlock_batch = payload["tracker_validation"]["next_unlock_batch"]
    current_stage = next_unlock_batch.get("current_stage")
    follow_up_stage = next_unlock_batch.get("follow_up_stage")
    current_wave_effects = current_stage.get("wave_effects", []) if current_stage else []
    follow_up_wave_effects = follow_up_stage.get("wave_effects", []) if follow_up_stage else []
    follow_up_manual_gates = (
        follow_up_stage.get("remaining_manual_gates", []) if follow_up_stage else []
    )

    brief = {
        "current_state": {
            "tracked_workers": validation_summary.get("tracked_workers"),
            "violation_count": validation_summary.get("violation_count"),
            "high_severity_count": validation_summary.get("high_severity_count"),
            "dirty_worker_count": validation_summary.get("dirty_worker_count"),
            "merge_ready_workers": validation_summary.get("merge_ready_workers"),
        },
        "run_now": {
            "stage": current_stage.get("stage") if current_stage else None,
            "worktrees": current_stage.get("incremental_worktrees", []) if current_stage else [],
            "worktree_count": len(current_stage.get("incremental_worktrees", []))
            if current_stage
            else 0,
            "command_count": current_stage.get("flattened_command_count", 0)
            if current_stage
            else 0,
            "triggering_worktrees": current_stage.get("triggering_worktrees", [])
            if current_stage
            else [],
            "command_batch": current_stage.get("flattened_command_batch", [])
            if current_stage
            else [],
            "recommended_action": current_stage.get("recommended_next_action")
            if current_stage
            else None,
        },
        "payoff": {
            "waves_newly_ready": [item["wave"] for item in current_wave_effects],
            "wave_effects": current_wave_effects,
            "first_fully_ready_wave": next(
                (
                    item["wave"]
                    for item in current_wave_effects
                    if item.get("newly_fully_ready")
                ),
                None,
            ),
        },
        "follow_up": {
            "stage": follow_up_stage.get("stage") if follow_up_stage else None,
            "worktrees": follow_up_stage.get("incremental_worktrees", [])
            if follow_up_stage
            else [],
            "worktree_count": len(follow_up_stage.get("incremental_worktrees", []))
            if follow_up_stage
            else 0,
            "command_count": follow_up_stage.get("flattened_command_count", 0)
            if follow_up_stage
            else 0,
            "wave_effects": follow_up_wave_effects,
            "remaining_manual_gates": follow_up_manual_gates,
            "remaining_manual_gate_count": len(follow_up_manual_gates),
        },
        "recommended_sequence": [
            "Run the current-stage cleanup pack to unlock the earliest fully ready wave.",
            "Run the follow-up stage to make wave 0 cleanup-ready.",
            "Clear the remaining lead-drift gate on agent-01 before attempting reserved-file merges.",
        ],
    }
    return {
        "summary": {
            "current_stage": brief["run_now"]["stage"],
            "current_stage_worktree_count": brief["run_now"]["worktree_count"],
            "current_stage_command_count": brief["run_now"]["command_count"],
            "current_stage_newly_ready_waves": brief["payoff"]["waves_newly_ready"],
            "follow_up_stage": brief["follow_up"]["stage"],
            "follow_up_worktree_count": brief["follow_up"]["worktree_count"],
            "follow_up_manual_gate_count": brief["follow_up"]["remaining_manual_gate_count"],
        },
        "brief": brief,
    }


def build_objective_audit(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "requirements": []}

    validation_summary = payload["tracker_validation"]["summary"]
    remediation_summary = payload["tracker_validation"]["remediation"]["summary"]
    merge_summary = payload["tracker_validation"]["merge_gates"]["summary"]
    execution_summary = payload["tracker_validation"]["execution_queue"]["summary"]
    checklist = tracker.get("checklist", [])
    checklist_items = {item.get("item"): item for item in checklist}
    merge_order = tracker.get("merge_order", [])
    tracker_stale = not validation_summary.get("tracker_matches_live_state", False)
    compatibility_lanes_active = any(
        matches_any(path, ["ui-tui/**", "gateway/**", "tools/send_message_tool.py", "toolsets.py", "tests/**"])
        for entry in payload["workers"]
        for path in entry.get("active_files", [])
    )

    requirements = []

    requirements.append(
        {
            "requirement": "Track all 21 worker branches",
            "status": "complete"
            if validation_summary.get("tracked_workers") == 21
            else "incomplete",
            "proof_strength": "direct",
            "evidence": (
                f"tracked_workers={validation_summary.get('tracked_workers')}; "
                f"dirty_worker_count={validation_summary.get('dirty_worker_count')}."
            ),
            "gaps": [],
        }
    )

    requirements.append(
        {
            "requirement": "Define merge order",
            "status": "complete"
            if merge_order and merge_summary.get("wave_count") == len(merge_order)
            else "incomplete",
            "proof_strength": "direct",
            "evidence": (
                f"merge_order_count={len(merge_order)}; "
                f"merge_gate_wave_count={merge_summary.get('wave_count')}."
            ),
            "gaps": []
            if merge_order and merge_summary.get("wave_count") == len(merge_order)
            else ["merge order is missing or does not match the live wave gate count"],
        }
    )

    conflict_gaps = []
    if remediation_summary.get("paths_with_overlaps", 0) > 0:
        conflict_gaps.append(
            f"paths_with_overlaps={remediation_summary.get('paths_with_overlaps')}"
        )
    if remediation_summary.get("critical_overlap_count", 0) > 0:
        conflict_gaps.append(
            f"critical_overlap_count={remediation_summary.get('critical_overlap_count')}"
        )
    if remediation_summary.get("ownerless_overlap_count", 0) > 0:
        conflict_gaps.append(
            f"ownerless_overlap_count={remediation_summary.get('ownerless_overlap_count')}"
        )
    if remediation_summary.get("ownership_decision_count", 0) > 0:
        conflict_gaps.append(
            f"ownership_decision_count={remediation_summary.get('ownership_decision_count')}"
        )
    if remediation_summary.get("tracker_edit_action_count", 0) > 0:
        conflict_gaps.append(
            f"tracker_edit_action_count={remediation_summary.get('tracker_edit_action_count')}"
        )
    if remediation_summary.get("lane_proposal_count", 0) > 0:
        conflict_gaps.append(
            f"lane_proposal_count={remediation_summary.get('lane_proposal_count')}"
        )
    if remediation_summary.get("unowned_path_count", 0) > 0:
        conflict_gaps.append(
            f"unowned_path_count={remediation_summary.get('unowned_path_count')}"
        )
    requirements.append(
        {
            "requirement": "Prevent shared-file conflicts",
            "status": "complete" if not conflict_gaps else "in_progress",
            "proof_strength": "direct",
            "evidence": (
                f"paths_with_overlaps={remediation_summary.get('paths_with_overlaps')}; "
                f"critical_overlap_count={remediation_summary.get('critical_overlap_count')}; "
                f"ownership_decision_count={remediation_summary.get('ownership_decision_count')}; "
                f"tracker_edit_action_count={remediation_summary.get('tracker_edit_action_count')}."
            ),
            "gaps": conflict_gaps,
        }
    )

    checklist_item = checklist_items.get("Keep an integration checklist", {})
    checklist_gaps = []
    if not checklist:
        checklist_gaps.append("tracker checklist is missing")
    if tracker_stale:
        checklist_gaps.append("tracker snapshot is stale relative to live state")
    requirements.append(
        {
            "requirement": "Keep an integration checklist",
            "status": "stale" if checklist_gaps else "complete",
            "proof_strength": "direct",
            "evidence": checklist_item.get(
                "evidence",
                f"checklist_count={len(checklist)}; tracker_matches_live_state={validation_summary.get('tracker_matches_live_state')}.",
            ),
            "gaps": checklist_gaps,
        }
    )

    blocker_gaps = []
    if execution_summary.get("operation_count", 0) == 0:
        blocker_gaps.append("execution queue has no tracked operations")
    if validation_summary.get("violation_count", 0) == 0 and merge_summary.get("blocked_waves", 0) == 0:
        blocker_gaps.append("no active blockers are currently being surfaced")
    requirements.append(
        {
            "requirement": "Continuously identify blockers",
            "status": "operational" if not blocker_gaps else "incomplete",
            "proof_strength": "direct",
            "evidence": (
                f"violation_count={validation_summary.get('violation_count')}; "
                f"blocked_waves={merge_summary.get('blocked_waves')}; "
                f"execution_operation_count={execution_summary.get('operation_count')}."
            ),
            "gaps": blocker_gaps,
        }
    )

    rebase_gaps = []
    if payload["summary"].get("lead_vs_main_file_drift"):
        rebase_gaps.append(
            f"lead drift files={payload['summary'].get('lead_vs_main_file_drift')}"
        )
    requirements.append(
        {
            "requirement": "Rebase the integration stack onto main",
            "status": "blocked" if rebase_gaps else "complete",
            "proof_strength": "direct",
            "evidence": (
                f"lead_head={payload['summary'].get('lead_head')}; "
                f"main_head={payload['summary'].get('main_head')}."
            ),
            "gaps": rebase_gaps,
        }
    )

    port_gaps = []
    if validation_summary.get("dirty_worker_count", 0) > 0:
        port_gaps.append(f"dirty_worker_count={validation_summary.get('dirty_worker_count')}")
    if merge_summary.get("blocked_waves", 0) > 0:
        port_gaps.append(f"blocked_waves={merge_summary.get('blocked_waves')}")
    if remediation_summary.get("paths_with_overlaps", 0) > 0:
        port_gaps.append(f"paths_with_overlaps={remediation_summary.get('paths_with_overlaps')}")
    if compatibility_lanes_active:
        port_gaps.append("Python/TS compatibility lanes are still active")
    requirements.append(
        {
            "requirement": "Complete the Python/TS-to-Rust port",
            "status": "in_progress" if port_gaps else "complete",
            "proof_strength": "direct",
            "evidence": (
                f"dirty_worker_count={validation_summary.get('dirty_worker_count')}; "
                f"blocked_waves={merge_summary.get('blocked_waves')}; "
                f"paths_with_overlaps={remediation_summary.get('paths_with_overlaps')}."
            ),
            "gaps": port_gaps,
        }
    )

    return {
        "summary": {
            "requirement_count": len(requirements),
            "complete_count": sum(1 for item in requirements if item["status"] == "complete"),
            "operational_count": sum(1 for item in requirements if item["status"] == "operational"),
            "in_progress_count": sum(1 for item in requirements if item["status"] == "in_progress"),
            "blocked_count": sum(1 for item in requirements if item["status"] == "blocked"),
            "stale_count": sum(1 for item in requirements if item["status"] == "stale"),
            "tracker_sync_required": tracker_stale,
            "objective_complete": all(item["status"] == "complete" for item in requirements),
        },
        "requirements": requirements,
    }


def build_objective_resolution_runbook(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "requirements": []}

    audit = payload["tracker_validation"]["objective_audit"]
    remediation_summary = payload["tracker_validation"]["remediation"]["summary"]
    next_unlock_brief = payload["tracker_validation"]["next_unlock_brief"]["brief"]
    sync_required = audit.get("summary", {}).get("tracker_sync_required", False)
    commands = {
        "sync": tracker.get("sync_command"),
        "objective_audit": tracker.get("objective_audit_command"),
        "conflict_runbook": tracker.get("conflict_runbook_command"),
        "decision_queue": tracker.get("decision_queue_command"),
        "lane_proposals": tracker.get("lane_proposals_command"),
        "tracker_edit_plan": tracker.get("tracker_edit_plan_command"),
        "next_unlock_brief": tracker.get("next_unlock_brief_command"),
        "next_unlock_batch": tracker.get("next_unlock_batch_command"),
        "merge_gates": tracker.get("merge_gates_command"),
        "wave_command_runbook": tracker.get("wave_command_runbook_command"),
        "cleanup_impact": tracker.get("cleanup_impact_command"),
        "cleanup_forecast": tracker.get("cleanup_forecast_command"),
        "unlock_frontier": tracker.get("unlock_frontier_command"),
    }

    runbook = []
    for item in audit.get("requirements", []):
        requirement = item.get("requirement")
        status = item.get("status")
        entry = {
            "requirement": requirement,
            "status": status,
            "gaps": item.get("gaps", []),
            "evidence": item.get("evidence"),
            "sync_required": sync_required,
            "prerequisite_commands": [commands["sync"]] if sync_required else [],
            "commands": [],
            "recommended_next_action": None,
        }

        if requirement == "Prevent shared-file conflicts":
            entry["commands"] = [
                commands["conflict_runbook"],
                commands["decision_queue"],
                commands["lane_proposals"],
                commands["tracker_edit_plan"],
                commands["cleanup_impact"],
            ]
            entry["recommended_next_action"] = (
                "Resolve ownership and lane gaps first, then execute the cleanup queue against the highest-impact overlap clusters."
            )
            entry["live_context"] = {
                "paths_with_overlaps": remediation_summary.get("paths_with_overlaps"),
                "critical_overlap_count": remediation_summary.get("critical_overlap_count"),
                "ownership_decision_count": remediation_summary.get("ownership_decision_count"),
                "tracker_edit_action_count": remediation_summary.get("tracker_edit_action_count"),
                "unowned_path_count": remediation_summary.get("unowned_path_count"),
            }
        elif requirement == "Keep an integration checklist":
            entry["commands"] = [commands["sync"], commands["objective_audit"]]
            entry["recommended_next_action"] = (
                "Refresh the tracker before using any checklist evidence or execution protocol step."
            )
            entry["live_context"] = {
                "live_state_fingerprint": payload["summary"].get("state_fingerprint"),
                "tracker_matches_live_state": payload["tracker_validation"]["summary"].get(
                    "tracker_matches_live_state"
                ),
            }
        elif requirement == "Continuously identify blockers":
            entry["commands"] = [
                commands["objective_audit"],
                commands["merge_gates"],
                commands["unlock_frontier"],
            ]
            entry["recommended_next_action"] = (
                "Use the audit for objective-level status, then use merge gates and unlock frontier for the current operational blocker."
            )
            entry["live_context"] = {
                "violation_count": payload["tracker_validation"]["summary"].get(
                    "violation_count"
                ),
                "blocked_waves": payload["tracker_validation"]["merge_gates"]["summary"].get(
                    "blocked_waves"
                ),
            }
        elif requirement == "Rebase the integration stack onto main":
            entry["commands"] = [
                commands["next_unlock_brief"],
                commands["next_unlock_batch"],
                commands["wave_command_runbook"],
            ]
            entry["recommended_next_action"] = (
                "Run the current cleanup stage and its follow-up stage, then clear the lead rebase gate on agent-01."
            )
            entry["live_context"] = {
                "run_now_worktree_count": next_unlock_brief.get("run_now", {}).get(
                    "worktree_count"
                ),
                "run_now_command_count": next_unlock_brief.get("run_now", {}).get(
                    "command_count"
                ),
                "follow_up_worktree_count": next_unlock_brief.get("follow_up", {}).get(
                    "worktree_count"
                ),
                "remaining_manual_gates": next_unlock_brief.get("follow_up", {}).get(
                    "remaining_manual_gates", []
                ),
            }
        elif requirement == "Complete the Python/TS-to-Rust port":
            entry["commands"] = [
                commands["objective_audit"],
                commands["next_unlock_brief"],
                commands["cleanup_forecast"],
                commands["merge_gates"],
            ]
            entry["recommended_next_action"] = (
                "Reduce overlap clusters and blocked waves first; the port is only complete once dirty workers, blocked waves, and compatibility-lane drift all reach zero."
            )
            entry["live_context"] = {
                "dirty_worker_count": payload["tracker_validation"]["summary"].get(
                    "dirty_worker_count"
                ),
                "blocked_waves": payload["tracker_validation"]["merge_gates"]["summary"].get(
                    "blocked_waves"
                ),
                "paths_with_overlaps": remediation_summary.get("paths_with_overlaps"),
                "compatibility_lanes_active": any(
                    matches_any(
                        path,
                        [
                            "ui-tui/**",
                            "gateway/**",
                            "tools/send_message_tool.py",
                            "toolsets.py",
                            "tests/**",
                        ],
                    )
                    for entry_worker in payload["workers"]
                    for path in entry_worker.get("active_files", [])
                ),
            }
        else:
            entry["commands"] = [commands["objective_audit"]]
            entry["recommended_next_action"] = "No additional runbook action required."

        entry["commands"] = [command for command in entry["commands"] if command]
        runbook.append(entry)

    return {
        "summary": {
            "requirement_count": len(runbook),
            "actionable_requirement_count": sum(
                1
                for item in runbook
                if item.get("status") not in {"complete", "operational"}
            ),
            "sync_required": sync_required,
            "top_requirement": next(
                (
                    item["requirement"]
                    for item in runbook
                    if item.get("status") in {"stale", "blocked", "in_progress"}
                ),
                None,
            ),
        },
        "requirements": runbook,
    }


def build_wave_command_runbook(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "waves": []}

    remediation = payload["tracker_validation"]["remediation"]
    wave_critical_path = payload["tracker_validation"]["wave_critical_path"]
    execution_queue = payload["tracker_validation"]["execution_queue"]
    branch_batch_map = {
        item["worktree"]: item for item in remediation.get("branch_batches", [])
    }
    execution_wave_map = {
        item["wave"]: item for item in execution_queue.get("waves", [])
    }

    waves = []
    for item in wave_critical_path.get("prioritized", []):
        post_cleanup_blockers = item.get("post_cleanup_blockers") or {}
        cleanup_batches = []
        total_cleanup_command_count = 0
        for index, worktree in enumerate(item.get("cleanup_prefix", []), start=1):
            batch = branch_batch_map.get(worktree)
            if not batch:
                continue
            command_batch = batch.get("command_batch", [])
            cleanup_batches.append(
                {
                    "sequence": index,
                    "worktree": worktree,
                    "branch": batch.get("branch"),
                    "release_count": batch.get("release_count", 0),
                    "lead_only_count": batch.get("lead_only_count", 0),
                    "unowned_count": batch.get("unowned_count", 0),
                    "command_count": len(command_batch),
                    "command_batch": command_batch,
                }
            )
            total_cleanup_command_count += len(command_batch)

        wave_execution = execution_wave_map.get(item.get("wave"), {})
        manual_operations = wave_execution.get("manual_operations", [])
        post_cleanup_manual_steps = []
        if post_cleanup_blockers.get("manual_blockers"):
            for blocker in post_cleanup_blockers["manual_blockers"]:
                if blocker == "lead_drift":
                    lead_sync = next(
                        (
                            operation
                            for operation in manual_operations
                            if operation.get("operation_type") == "lead_sync"
                        ),
                        None,
                    )
                    if lead_sync:
                        post_cleanup_manual_steps.append(
                            {
                                "blocker": blocker,
                                "worktree": lead_sync.get("worktree"),
                                "operation_type": lead_sync.get("operation_type"),
                                "reason": lead_sync.get("reason"),
                                "paths": lead_sync.get("paths", []),
                                "blocking_local_changes": lead_sync.get("blocking_local_changes", []),
                                "command_batch": lead_sync.get("command_batch", []),
                                "command_count": lead_sync.get("command_count", 0),
                            }
                        )
                        continue
                post_cleanup_manual_steps.append(
                    {
                        "blocker": blocker,
                        "worktree": None,
                        "operation_type": "manual_gate",
                        "reason": blocker,
                        "paths": [],
                        "blocking_local_changes": [],
                        "command_batch": [],
                        "command_count": 0,
                    }
                )

        waves.append(
            {
                "wave": item.get("wave"),
                "owner": item.get("owner"),
                "goal": item.get("goal"),
                "status": item.get("status"),
                "cleanup_ready_after_batch": item.get("cleanup_ready_after_batch"),
                "cleanup_ready_after_worktree": item.get("cleanup_ready_after_worktree"),
                "cleanup_batches": cleanup_batches,
                "cleanup_batch_count": len(cleanup_batches),
                "cleanup_command_count": total_cleanup_command_count,
                "post_cleanup_manual_steps": post_cleanup_manual_steps,
                "post_cleanup_manual_step_count": len(post_cleanup_manual_steps),
                "current_blockers": item.get("current_blockers"),
                "post_cleanup_blockers": post_cleanup_blockers,
                "recommended_next_action": item.get("recommended_next_action"),
            }
        )

    return {
        "waves": waves,
        "prioritized": waves,
        "summary": {
            "wave_count": len(waves),
            "cleanup_ready_wave_count": sum(
                1 for item in waves if item.get("cleanup_ready_after_batch") is not None
            ),
            "manual_gate_wave_count": sum(
                1 for item in waves if item.get("post_cleanup_manual_step_count", 0) > 0
            ),
            "total_cleanup_batches": sum(item.get("cleanup_batch_count", 0) for item in waves),
            "top_wave": waves[0]["wave"] if waves else None,
        },
    }


def build_remediation_plan(
    payload: dict,
    branch_plans: dict[str, dict],
    lead_only: set[str],
    ownership_overrides: list[dict],
    hotspot_paths: set[str],
    worktree_root: str,
) -> dict:
    workers = {entry["worktree"]: entry for entry in payload["workers"]}
    path_activity: dict[str, list[str]] = defaultdict(list)
    for worktree, entry in workers.items():
        for path in entry["active_files"]:
            path_activity[path].append(worktree)

    path_actions = []
    branch_actions = []
    branch_action_map: dict[str, dict] = {}
    owner_cache: dict[str, dict | None] = {}

    for path, active_workers in sorted(path_activity.items()):
        owner_info = owner_cache.setdefault(
            path,
            resolve_owner(path, branch_plans, lead_only, ownership_overrides),
        )
        owner = owner_info.get("owner") if owner_info else None
        keep_in = [worker for worker in active_workers if worker == owner] if owner else []
        release_from = [worker for worker in active_workers if worker != owner] if owner else active_workers
        path_actions.append(
            {
                "path": path,
                "active_workers": sorted(active_workers),
                "owner": owner,
                "owner_reason": owner_info.get("reason") if owner_info else "No owner resolved from tracker.",
                "owner_source": owner_info.get("source") if owner_info else "unowned",
                "candidates": owner_info.get("candidates", []) if owner_info else [],
                "keep_in": sorted(keep_in),
                "release_from": sorted(release_from),
            }
        )

    for worktree, entry in sorted(workers.items()):
        keep_paths = []
        release_paths = []
        ambiguous_paths = []
        unowned_paths = []
        cleanup_commands = []
        for path in entry["active_files"]:
            owner_info = owner_cache.get(path)
            owner = owner_info.get("owner") if owner_info else None
            if owner == worktree:
                keep_paths.append(path)
            elif owner:
                status_code = entry["status_by_path"].get(path, "")
                command = cleanup_command_for_path(worktree_root, worktree, path, status_code)
                release_paths.append(
                    {
                        "path": path,
                        "owner": owner,
                        "reason": owner_info.get("reason", ""),
                        "source": owner_info.get("source", ""),
                        "cleanup_command": command,
                        "status_code": status_code,
                    }
                )
                cleanup_commands.append(command)
            elif owner_info and owner_info.get("candidates"):
                ambiguous_paths.append(
                    {
                        "path": path,
                        "candidates": owner_info["candidates"],
                        "reason": owner_info.get("reason", ""),
                    }
                )
            else:
                unowned_paths.append(path)

        branch_action = (
            {
                "worktree": worktree,
                "branch": entry["branch"],
                "status": entry["status"],
                "behind_main": entry["behind_main"],
                "upstream_configured": bool(entry["upstream"]),
                "keep_paths": keep_paths,
                "release_paths": release_paths,
                "ambiguous_paths": ambiguous_paths,
                "unowned_paths": unowned_paths,
                "cleanup_commands": cleanup_commands,
                "lead_only_hits": [path for path in entry["active_files"] if path in lead_only],
                "needs_rebase": entry["behind_main"] > 0,
                "needs_upstream": not bool(entry["upstream"]),
                "ready_after_cleanup": not release_paths and not ambiguous_paths and not unowned_paths,
                "rebase_command": (
                    f"git -C {Path(worktree_root, worktree)} rebase main"
                    if entry["behind_main"] > 0
                    else None
                ),
                "publish_command": (
                    f"git -C {Path(worktree_root, worktree)} push -u origin {entry['branch']}"
                    if not entry["upstream"]
                    else None
                ),
            }
        )
        branch_action["cleanup_commands"] = unique_preserve_order(branch_action["cleanup_commands"])
        branch_action["post_cleanup_commands"] = [
            command
            for command in [branch_action["rebase_command"], branch_action["publish_command"]]
            if command
        ]
        branch_action["command_batch"] = unique_preserve_order(
            branch_action["cleanup_commands"] + branch_action["post_cleanup_commands"]
        )
        branch_actions.append(branch_action)
        branch_action_map[worktree] = branch_action

    owner_runbook: list[dict] = []
    owners = sorted(
        {
            item["owner"]
            for item in path_actions
            if item["owner"]
        }
    )
    for owner in owners:
        owned_paths = [item for item in path_actions if item["owner"] == owner]
        awaiting_release_from = sorted(
            {
                worktree
                for item in owned_paths
                for worktree in item["release_from"]
            }
        )
        owner_branch = branch_action_map.get(owner)
        owner_runbook.append(
            {
                "owner": owner,
                "owned_paths": [
                    {
                        "path": item["path"],
                        "active_workers": item["active_workers"],
                        "release_from": item["release_from"],
                    }
                    for item in owned_paths
                ],
                "awaiting_release_from": awaiting_release_from,
                "owner_branch_ready_after_cleanup": owner_branch["ready_after_cleanup"]
                if owner_branch
                else owner == "agent-01",
                "owner_branch_release_paths": owner_branch["release_paths"] if owner_branch else [],
                "owner_branch_unowned_paths": owner_branch["unowned_paths"] if owner_branch else [],
                "owner_branch_lead_only_hits": owner_branch["lead_only_hits"] if owner_branch else [],
                "owner_rebase_command": owner_branch["rebase_command"] if owner_branch else None,
                "owner_publish_command": owner_branch["publish_command"] if owner_branch else None,
                "owner_command_batch": owner_branch["command_batch"] if owner_branch else [],
            }
        )

    branch_triage = build_branch_triage(branch_actions, owner_runbook)
    branch_lifecycle = build_branch_lifecycle(branch_triage, owner_runbook)
    conflict_runbook = build_conflict_runbook(
        path_actions,
        branch_actions,
        branch_plans,
        hotspot_paths,
    )
    lane_gap_report = build_lane_gap_report(branch_actions, branch_plans, path_actions)
    tracker_realignments = build_tracker_realignments(
        branch_lifecycle,
        lane_gap_report,
        branch_plans,
    )
    decision_queue = build_decision_queue(
        lane_gap_report,
        tracker_realignments,
        branch_triage,
        branch_lifecycle,
        conflict_runbook,
        branch_actions,
        branch_plans,
    )
    lane_proposals = build_lane_proposals(decision_queue, branch_plans)
    tracker_edit_plan = build_tracker_edit_plan(
        decision_queue,
        lane_proposals,
        branch_plans,
    )
    choice_runbook = build_choice_runbook(
        tracker_edit_plan,
        branch_actions,
        branch_triage,
        branch_lifecycle,
        path_actions,
        branch_plans,
    )
    cleanup_impact_queue = build_cleanup_impact_queue(
        branch_actions,
        owner_runbook,
        conflict_runbook,
        branch_plans,
    )
    cleanup_forecast = build_cleanup_forecast(
        cleanup_impact_queue,
        path_actions,
        branch_actions,
        owner_runbook,
        conflict_runbook,
    )

    branch_batches = [
        {
            "worktree": item["worktree"],
            "branch": item["branch"],
            "cleanup_commands": item["cleanup_commands"],
            "post_cleanup_commands": item["post_cleanup_commands"],
            "command_batch": item["command_batch"],
            "release_count": len(item["release_paths"]),
            "lead_only_count": len(item["lead_only_hits"]),
            "unowned_count": len(item["unowned_paths"]),
        }
        for item in branch_actions
    ]

    prioritized_actions = [
        {
            "priority": 1,
            "owner": "agent-01",
            "action": "Rebase the lead integration branch onto main and absorb all lead-only paths.",
            "paths": sorted(
                {
                    path
                    for path, workers_on_path in path_activity.items()
                    if path in lead_only and workers_on_path
                }
            ),
        },
        {
            "priority": 2,
            "owner": "agent-01",
            "action": "Reduce each overlap cluster to a single owner before merging any worker branch.",
            "paths": [item["path"] for item in path_actions if len(item["active_workers"]) > 1],
        },
        {
            "priority": 3,
            "owner": "agent-01",
            "action": "Retask or stop workers with files outside their declared lane.",
            "workers": [
                item["worktree"]
                for item in branch_actions
                if item["release_paths"] or item["ambiguous_paths"] or item["unowned_paths"]
            ],
        },
        {
            "priority": 4,
            "owner": "agent-01",
            "action": "Rebase and publish only after each branch has released non-owned paths.",
            "workers": [item["worktree"] for item in branch_actions if item["needs_rebase"]],
        },
    ]

    return {
        "path_actions": path_actions,
        "branch_actions": branch_actions,
        "branch_batches": branch_batches,
        "branch_triage": branch_triage,
        "branch_lifecycle": branch_lifecycle,
        "conflict_runbook": conflict_runbook,
        "lane_gap_report": lane_gap_report,
        "tracker_realignments": tracker_realignments,
        "decision_queue": decision_queue,
        "lane_proposals": lane_proposals,
        "tracker_edit_plan": tracker_edit_plan,
        "choice_runbook": choice_runbook,
        "cleanup_impact_queue": cleanup_impact_queue,
        "cleanup_forecast": cleanup_forecast,
        "owner_runbook": owner_runbook,
        "prioritized_actions": prioritized_actions,
        "summary": {
            "paths_with_overlaps": sum(1 for item in path_actions if len(item["active_workers"]) > 1),
            "branches_needing_handoff": sum(
                1
                for item in branch_actions
                if item["release_paths"] or item["ambiguous_paths"] or item["unowned_paths"]
            ),
            "branches_ready_after_cleanup": sum(
                1
                for item in branch_actions
                if item["ready_after_cleanup"] and not item["lead_only_hits"]
            ),
            "branches_blocked_by_lead_only": sum(1 for item in branch_actions if item["lead_only_hits"]),
            "branches_blocked_by_handoffs": sum(1 for item in branch_actions if item["release_paths"]),
            "owners_waiting_on_releases": sum(1 for item in owner_runbook if item["awaiting_release_from"]),
            "release_only_branches": branch_triage["summary"]["release_only"],
            "retask_or_stop_branches": branch_triage["summary"]["retask_or_stop"],
            "salvage_in_lane_branches": branch_triage["summary"]["salvage_in_lane"],
            "salvage_with_reassignment_branches": branch_triage["summary"]["salvage_with_reassignment"],
            "retire_after_release_branches": branch_lifecycle["summary"]["retire_after_release"],
            "receive_then_retain_branches": branch_lifecycle["summary"]["receive_then_retain"],
            "critical_overlap_count": conflict_runbook["summary"]["critical_count"],
            "ownerless_overlap_count": conflict_runbook["summary"]["ownerless_overlap_count"],
            "ownership_decision_count": decision_queue["summary"]["decision_count"],
            "lane_proposal_count": lane_proposals["summary"]["proposal_count"],
            "tracker_edit_action_count": tracker_edit_plan["summary"]["actionable_edit_count"],
            "pending_choice_count": choice_runbook["summary"]["pending_choice_count"],
            "cleanup_impact_branch_count": cleanup_impact_queue["summary"]["branch_count"],
            "cleanup_impact_top_candidate": cleanup_impact_queue["summary"]["top_candidate"],
            "cleanup_forecast_milestone_count": cleanup_forecast["summary"]["milestone_count"],
            "unowned_path_count": lane_gap_report["summary"]["unowned_path_count"],
            "new_lane_candidates": lane_gap_report["summary"]["new_lane_candidates"],
            "allowed_path_addition_count": tracker_realignments["summary"]["allowed_path_addition_count"],
            "review_candidate_count": tracker_realignments["summary"]["review_candidate_count"],
        },
    }


def cleanup_command_for_path(worktree_root: str, worktree: str, path: str, status_code: str) -> str:
    target = Path(worktree_root, worktree)
    if status_code == "??":
        return f"git -C {target} clean -f -- {path}"
    return f"git -C {target} restore --source=HEAD --staged --worktree -- {path}"


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def select_payload(data: object, selector: str | None) -> object:
    if not selector:
        return data
    current = data
    for part in selector.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index]
            continue
        raise KeyError(f"selector path not found: {selector}")
    return current


def build_merge_gates(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "waves": []}

    remediation = payload["tracker_validation"]["remediation"]
    branch_map = {item["worktree"]: item for item in remediation["branch_actions"]}
    owner_map = {item["owner"]: item for item in remediation["owner_runbook"]}
    waves = []

    for wave in tracker.get("merge_order", []):
        wave_no = wave.get("wave")
        wave_workers = list(wave.get("workers", []))
        owner = wave.get("owner")
        if owner and owner != "agent-01" and owner not in wave_workers:
            wave_workers.append(owner)

        worker_actions = [branch_map[worker] for worker in wave_workers if worker in branch_map]
        blocked_by = []
        cleanup_commands = []
        branch_batches = []
        for action in worker_actions:
            if action["lead_only_hits"]:
                blocked_by.append(
                    {
                        "type": "lead_only",
                        "worktree": action["worktree"],
                        "paths": action["lead_only_hits"],
                    }
                )
            if action["release_paths"]:
                blocked_by.append(
                    {
                        "type": "handoff",
                        "worktree": action["worktree"],
                        "paths": [item["path"] for item in action["release_paths"]],
                    }
                )
                cleanup_commands.extend(action["cleanup_commands"])
            if action["unowned_paths"]:
                blocked_by.append(
                    {
                        "type": "unowned",
                        "worktree": action["worktree"],
                        "paths": action["unowned_paths"],
                    }
                )
            if action["ambiguous_paths"]:
                blocked_by.append(
                    {
                        "type": "ambiguous",
                        "worktree": action["worktree"],
                        "paths": [item["path"] for item in action["ambiguous_paths"]],
                    }
                )
            branch_batches.append(
                {
                    "worktree": action["worktree"],
                    "cleanup_commands": action["cleanup_commands"],
                    "post_cleanup_commands": action["post_cleanup_commands"],
                    "command_batch": action["command_batch"],
                }
            )

        owner_runbook = owner_map.get(owner) if owner else None
        if owner == "agent-01" and payload["summary"]["lead_vs_main_file_drift"]:
            blocked_by.append(
                {
                    "type": "lead_drift",
                    "worktree": "agent-01",
                    "paths": payload["summary"]["lead_vs_main_file_drift"],
                }
            )
        if owner_runbook and owner_runbook["awaiting_release_from"]:
            blocked_by.append(
                {
                    "type": "owner_waiting",
                    "worktree": owner,
                    "paths": [item["path"] for item in owner_runbook["owned_paths"] if item["release_from"]],
                    "waiting_on": owner_runbook["awaiting_release_from"],
                }
            )

        ready_workers = [
            action["worktree"]
            for action in worker_actions
            if action["ready_after_cleanup"] and not action["lead_only_hits"]
        ]
        status = "ready" if not blocked_by else "blocked"
        cleanup_commands = unique_preserve_order(cleanup_commands)
        waves.append(
            {
                "wave": wave_no,
                "goal": wave.get("goal"),
                "owner": owner,
                "paths": wave.get("paths", []),
                "workers": wave_workers,
                "status": status,
                "ready_workers": ready_workers,
                "blocked_workers": [action["worktree"] for action in worker_actions if action["worktree"] not in ready_workers],
                "blocked_by": blocked_by,
                "cleanup_commands": cleanup_commands,
                "branch_batches": branch_batches,
                "owner_waiting_on": owner_runbook["awaiting_release_from"] if owner_runbook else [],
                "owner_rebase_command": owner_runbook["owner_rebase_command"] if owner_runbook else None,
                "owner_publish_command": owner_runbook["owner_publish_command"] if owner_runbook else None,
            }
        )

    return {
        "summary": {
            "wave_count": len(waves),
            "ready_waves": sum(1 for wave in waves if wave["status"] == "ready"),
            "blocked_waves": sum(1 for wave in waves if wave["status"] == "blocked"),
            "first_blocked_wave": next((wave["wave"] for wave in waves if wave["status"] == "blocked"), None),
        },
        "waves": waves,
    }


def build_execution_queue(payload: dict, tracker: dict | None) -> dict:
    if not tracker:
        return {"summary": {}, "next_batch": None, "waves": [], "operations": []}

    remediation = payload["tracker_validation"]["remediation"]
    merge_gates = payload["tracker_validation"]["merge_gates"]
    branch_map = {item["worktree"]: item for item in remediation["branch_actions"]}
    owner_map = {item["owner"]: item for item in remediation["owner_runbook"]}
    lead_path = Path(payload["worktree_root"], payload["lead"]["worktree"])
    lead = payload["lead"]
    main_ref = payload["summary"]["main_ref"]
    waves = []
    operations = []
    next_order = 1

    for wave in merge_gates.get("waves", []):
        owner = wave.get("owner")
        owner_runbook = owner_map.get(owner) if owner else None
        worker_sequence: list[str] = []
        if owner_runbook:
            worker_sequence.extend(owner_runbook["awaiting_release_from"])
        for worker in wave.get("workers", []):
            if worker not in worker_sequence:
                worker_sequence.append(worker)

        wave_operations = []

        if owner == "agent-01" and payload["summary"]["lead_vs_main_file_drift"]:
            command_batch = [f"git -C {lead_path} rebase {main_ref}"]
            ready_to_run = lead["dirty_file_count"] == 0
            operation = {
                "order": next_order,
                "wave": wave["wave"],
                "worktree": "agent-01",
                "branch": lead["branch"],
                "operation_type": "lead_sync",
                "reason": "Lead branch is behind main and must absorb main drift before worker merges.",
                "paths": payload["summary"]["lead_vs_main_file_drift"],
                "blocking_local_changes": lead["dirty_files"],
                "ready_to_run": ready_to_run,
                "command_batch": command_batch,
                "command_count": len(command_batch),
            }
            wave_operations.append(operation)
            operations.append(operation)
            next_order += 1

        for worktree in worker_sequence:
            action = branch_map.get(worktree)
            if not action:
                continue
            reason_parts = []
            if owner and owner_runbook and worktree in owner_runbook["awaiting_release_from"]:
                reason_parts.append(f"release owner-blocking paths for {owner}")
            if action["release_paths"]:
                reason_parts.append("drop non-owned paths")
            if action["lead_only_hits"]:
                reason_parts.append("release lead-only paths")
            if action["ambiguous_paths"]:
                reason_parts.append("resolve ambiguous ownership")
            if action["unowned_paths"]:
                reason_parts.append("assign unowned paths")
            command_batch = action["command_batch"]
            operation = {
                "order": next_order,
                "wave": wave["wave"],
                "worktree": worktree,
                "branch": action["branch"],
                "operation_type": (
                    "release_to_owner"
                    if owner and owner_runbook and worktree in owner_runbook["awaiting_release_from"]
                    else "branch_cleanup"
                ),
                "reason": "; ".join(reason_parts) or "cleanup branch before merge",
                "owner": owner,
                "release_paths": [item["path"] for item in action["release_paths"]],
                "lead_only_hits": action["lead_only_hits"],
                "ambiguous_paths": [item["path"] for item in action["ambiguous_paths"]],
                "unowned_paths": action["unowned_paths"],
                "ready_to_run": bool(command_batch),
                "command_batch": command_batch,
                "command_count": len(command_batch),
            }
            wave_operations.append(operation)
            operations.append(operation)
            next_order += 1

        if owner_runbook:
            command_batch = owner_runbook["owner_command_batch"]
            waiting_on = owner_runbook["awaiting_release_from"]
            operation = {
                "order": next_order,
                "wave": wave["wave"],
                "worktree": owner,
                "branch": branch_map[owner]["branch"] if owner in branch_map else lead["branch"],
                "operation_type": "owner_finalize",
                "reason": (
                    f"receive released paths and finalize {owner} after dependencies clear"
                    if waiting_on
                    else f"{owner} can finalize owned paths"
                ),
                "owned_paths": [item["path"] for item in owner_runbook["owned_paths"]],
                "waiting_on": waiting_on,
                "ready_to_run": bool(command_batch) and not waiting_on,
                "command_batch": command_batch,
                "command_count": len(command_batch),
            }
            wave_operations.append(operation)
            operations.append(operation)
            next_order += 1

        executable_operations = [item for item in wave_operations if item["ready_to_run"]]
        manual_operations = [item for item in wave_operations if not item["ready_to_run"]]
        waves.append(
            {
                "wave": wave["wave"],
                "goal": wave.get("goal"),
                "status": wave.get("status"),
                "owner": owner,
                "operations": wave_operations,
                "executable_operations": executable_operations,
                "manual_operations": manual_operations,
                "command_count": sum(item["command_count"] for item in wave_operations),
            }
        )

    next_batch = next(
        (
            wave
            for wave in waves
            if wave["executable_operations"] or wave["manual_operations"]
        ),
        None,
    )

    return {
        "summary": {
            "wave_count": len(waves),
            "operation_count": len(operations),
            "executable_operation_count": sum(1 for item in operations if item["ready_to_run"]),
            "manual_operation_count": sum(1 for item in operations if not item["ready_to_run"]),
            "first_actionable_wave": next((wave["wave"] for wave in waves if wave["operations"]), None),
            "lead_sync_required": bool(payload["summary"]["lead_vs_main_file_drift"]),
        },
        "next_batch": next_batch,
        "waves": waves,
        "operations": operations,
    }


def build_tracker_blockers(payload: dict) -> list[dict]:
    overlaps = sorted(
        payload["shared_file_overlaps"],
        key=lambda item: (-len(item["worktrees"]), item["path"]),
    )
    lead_only = {
        item["path"]
        for item in payload["tracker_validation"]["remediation"]["path_actions"]
        if item["owner"] == "agent-01" and item["owner_source"] == "lead_only" and item["release_from"]
    }
    remediation_summary = payload["tracker_validation"]["remediation"]["summary"]
    merge_summary = payload["tracker_validation"]["merge_gates"]["summary"]
    validation_summary = payload["tracker_validation"]["summary"]
    local_only_workers = [
        entry
        for entry in payload["workers"]
        if not entry.get("upstream")
        and entry.get("status") == "dirty"
        and entry.get("committed_file_count_vs_main", 0) == 0
    ]
    local_only_summary = ", ".join(
        f"{entry['worktree']}@{entry['head']} ({entry['dirty_file_count']} dirty files)"
        for entry in local_only_workers[:6]
    )
    top_conflicts = ", ".join(
        f"{item['path'].split('/')[-1]} has {len(item['worktrees'])} workers"
        for item in overlaps[:6]
    )
    blockers = [
        {
            "severity": "high",
            "summary": (
                f"All {payload['summary']['worker_count']} worker branches are dirty."
                if payload["summary"]["dirty_workers"] == payload["summary"]["worker_count"]
                else f"{payload['summary']['dirty_workers']} of {payload['summary']['worker_count']} worker branches are dirty."
            ),
            "evidence": f"scripts/rust_port_status.py reports dirty_workers={payload['summary']['dirty_workers']}.",
            "next_action": "Stop new edits and reduce each overlap cluster to one owner before merging.",
        },
        {
            "severity": "high",
            "summary": "Reserved lead-only files are already being edited on worker branches.",
            "evidence": (
                ", ".join(sorted(lead_only))
                if lead_only
                else "No lead-only paths are currently held outside agent-01."
            ),
            "next_action": "Move those edits into the lead branch or drop them from workers.",
        },
        {
            "severity": "high",
            "summary": (
                f"The lead baseline is {payload['lead']['behind_main']} commits behind main."
                if payload["lead"]["behind_main"] > 0
                else "The lead baseline is aligned with main."
            ),
            "evidence": (
                f"main_head is {payload['summary']['main_head']} while the lead remains {payload['summary']['lead_head']}."
                if payload["lead"]["behind_main"] > 0
                else f"lead_head and main_head are both {payload['summary']['main_head']}."
            ),
            "next_action": "Rebase agent-01 first, then restack or rebase workers onto that refreshed baseline.",
        },
        {
            "severity": "high",
            "summary": "The worst conflict clusters are active right now.",
            "evidence": top_conflicts or "No shared-file overlaps are currently active.",
            "next_action": "Use live_conflict_clusters owners as the first serialization pass.",
        },
        {
            "severity": "high" if local_only_workers else "medium",
            "summary": (
                "Some worker branches still exist only as dirty local worktrees with no committed branch delta."
                if local_only_workers
                else (
                    "No worker branch has an upstream configured."
                    if payload["summary"]["workers_with_upstream"] == 0
                    else "Some worker branches still lack upstreams."
                )
            ),
            "evidence": (
                f"Unmergeable local-only branches: {local_only_summary}."
                if local_only_workers
                else f"workers_with_upstream={payload['summary']['workers_with_upstream']} in the status script summary."
            ),
            "next_action": (
                "Commit and push those local-only branches, or explicitly discard them, before treating their lanes as merge candidates."
                if local_only_workers
                else "Push the surviving branches after scopes are corrected so integration state is externally visible."
            ),
        },
        {
            "severity": "medium",
            "summary": "The validator still reports many scope violations because the live edits do not match the planned lanes.",
            "evidence": (
                f"tracker_validation.summary reports violation_count={validation_summary['violation_count']} "
                f"with high_severity_count={validation_summary['high_severity_count']}."
            ),
            "next_action": "Either retask the branches to their current files or move the files to the designated owners.",
        },
        {
            "severity": "medium",
            "summary": "Most owner branches are still waiting on releases from other workers.",
            "evidence": (
                "tracker_validation.remediation.summary reports "
                f"owners_waiting_on_releases={remediation_summary['owners_waiting_on_releases']}."
            ),
            "next_action": "Use owner_runbook to clear incoming release dependencies before attempting rebases.",
        },
        {
            "severity": "medium",
            "summary": "Every planned merge wave is currently blocked."
            if merge_summary["blocked_waves"] == merge_summary["wave_count"]
            else "Some planned merge waves are still blocked.",
            "evidence": (
                "tracker_validation.merge_gates.summary reports "
                f"blocked_waves={merge_summary['blocked_waves']} and ready_waves={merge_summary['ready_waves']}."
            ),
            "next_action": "Do not start later-wave merges while earlier-wave cleanup commands remain outstanding.",
        },
        {
            "severity": "medium",
            "summary": "The cleanup queue now exists, but every worker branch still requires at least one command batch before it can be restacked.",
            "evidence": (
                "tracker_validation.remediation.summary reports "
                f"branches_needing_handoff={remediation_summary['branches_needing_handoff']} "
                f"and branches_ready_after_cleanup={remediation_summary['branches_ready_after_cleanup']}."
            ),
            "next_action": "Use branch_batches or execution_queue.next_batch to execute cleanup in wave order.",
        },
    ]
    if any(
        matches_any(path, ["ui-tui/**", "gateway/**", "tools/send_message_tool.py", "toolsets.py", "tests/**"])
        for entry in payload["workers"]
        for path in entry["active_files"]
    ):
        blockers.append(
            {
                "severity": "medium",
                "summary": "Python and TypeScript compatibility work is still separate from the Rust runtime.",
                "evidence": "ui-tui, gateway, toolsets, or Python delivery/test files are still active on worker branches.",
                "next_action": "Keep those lanes late in the merge order and do not treat the Rust port as complete until they converge.",
            }
        )
    return blockers


def apply_high_confidence_realignments(tracker: dict, payload: dict) -> list[dict]:
    branch_plans = {
        item["worktree"]: item
        for item in tracker.get("worker_branches", [])
        if item.get("worktree")
    }
    additions = []
    for item in payload["tracker_validation"]["remediation"]["tracker_realignments"]["allowed_path_additions"]:
        owner = item["owner"]
        path = item["path"]
        plan = branch_plans.get(owner)
        if not plan:
            continue
        allowed_paths = plan.setdefault("allowed_paths", [])
        if path in allowed_paths:
            continue
        allowed_paths.append(path)
        additions.append(
            {
                "owner": owner,
                "path": path,
                "confidence": item["confidence"],
                "reason": item["reason"],
            }
        )
    return additions


def sync_tracker_state(tracker: dict, payload: dict) -> dict:
    synced = deepcopy(tracker)
    auto_applied_realignments = apply_high_confidence_realignments(synced, payload)
    generated_date = payload["generated_at"].split("T", 1)[0]
    synced["snapshot_date"] = generated_date
    synced["last_synced_at"] = payload["generated_at"]
    synced["state_fingerprint"] = payload["summary"]["state_fingerprint"]
    synced["refresh_command"] = "python3 scripts/rust_port_status.py"
    synced["summary_command"] = "python3 scripts/rust_port_status.py --select tracker_validation.summary"
    synced["validate_command"] = "python3 scripts/rust_port_status.py --select tracker_validation"
    synced["objective_audit_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.objective_audit"
    )
    synced["objective_resolution_runbook_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.objective_resolution_runbook"
    )
    synced["remediation_command"] = "python3 scripts/rust_port_status.py --select tracker_validation.remediation"
    synced["triage_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.branch_triage"
    )
    synced["lifecycle_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.branch_lifecycle"
    )
    synced["lane_gap_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.lane_gap_report"
    )
    synced["tracker_realignments_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.tracker_realignments"
    )
    synced["conflict_runbook_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.conflict_runbook"
    )
    synced["decision_queue_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.decision_queue"
    )
    synced["lane_proposals_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.lane_proposals"
    )
    synced["tracker_edit_plan_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.tracker_edit_plan"
    )
    synced["choice_runbook_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.choice_runbook"
    )
    synced["cleanup_impact_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.cleanup_impact_queue"
    )
    synced["cleanup_forecast_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.cleanup_forecast"
    )
    synced["apply_tracker_edit_plan_command"] = (
        "python3 scripts/rust_port_status.py --apply-tracker-edit-plan"
    )
    synced["apply_tracker_choice_command"] = (
        "python3 scripts/rust_port_status.py --apply-tracker-choice-path <path> --apply-tracker-choice-owner <worktree>"
    )
    synced["sync_drift_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.sync_drift"
    )
    synced["published_head_coverage_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.published_head_coverage"
    )
    synced["branch_batches_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.branch_batches"
    )
    synced["owner_runbook_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.remediation.owner_runbook"
    )
    synced["merge_gates_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.merge_gates"
    )
    synced["wave_unlock_forecast_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.wave_unlock_forecast"
    )
    synced["wave_critical_path_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.wave_critical_path"
    )
    synced["unlock_ladder_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.unlock_ladder"
    )
    synced["unlock_frontier_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.unlock_frontier"
    )
    synced["next_unlock_batch_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.next_unlock_batch"
    )
    synced["next_unlock_brief_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.next_unlock_brief"
    )
    synced["stage_command_runbook_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.stage_command_runbook"
    )
    synced["wave_command_runbook_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.wave_command_runbook"
    )
    synced["execution_queue_command"] = (
        "python3 scripts/rust_port_status.py --select tracker_validation.execution_queue"
    )
    synced["sync_command"] = "python3 scripts/rust_port_status.py --sync-tracker"
    prior_realignments = synced.get("auto_applied_realignments", [])
    seen_realignments = {
        (item.get("owner"), item.get("path"))
        for item in prior_realignments
    }
    merged_realignments = list(prior_realignments)
    for item in auto_applied_realignments:
        key = (item.get("owner"), item.get("path"))
        if key in seen_realignments:
            continue
        merged_realignments.append(item)
        seen_realignments.add(key)
    synced["auto_applied_realignments"] = merged_realignments

    scope = synced.setdefault("scope", {})
    scope["lead_worktree"] = payload["lead"]["worktree"]
    scope["lead_branch"] = payload["lead"]["branch"]
    scope["worker_branch_count"] = payload["summary"]["worker_count"]
    scope["baseline_head"] = payload["lead"]["head"]
    scope["baseline_subject"] = payload["lead"]["subject"]
    scope["main_head"] = payload["summary"]["main_head"]
    scope["main_subject"] = payload["summary"]["main_subject"]
    scope["main_drift_files"] = payload["summary"]["lead_vs_main_file_drift"]
    scope["state_fingerprint"] = payload["summary"]["state_fingerprint"]

    live_workers = {entry["worktree"]: entry for entry in payload["workers"]}
    for worker in synced.get("worker_branches", []):
        live = live_workers.get(worker.get("worktree"))
        if not live:
            continue
        worker["branch"] = live["branch"]
        worker["head"] = live["head"]
        worker["subject"] = live["subject"]
        worker["status"] = live["status"]
        worker["observed_dirty_files"] = live["dirty_files"]

    synced["live_conflict_clusters"] = build_live_conflict_clusters(payload)

    validation_summary = payload["tracker_validation"]["summary"]
    remediation_summary = payload["tracker_validation"]["remediation"]["summary"]
    triage_summary = payload["tracker_validation"]["remediation"]["branch_triage"]["summary"]
    lifecycle_summary = payload["tracker_validation"]["remediation"]["branch_lifecycle"]["summary"]
    lane_gap_summary = payload["tracker_validation"]["remediation"]["lane_gap_report"]["summary"]
    realignment_summary = payload["tracker_validation"]["remediation"]["tracker_realignments"]["summary"]
    execution_summary = payload["tracker_validation"]["execution_queue"]["summary"]
    for item in synced.get("checklist", []):
        label = item.get("item")
        if label == "Track all 21 worker branches":
            item["status"] = "done" if payload["summary"]["worker_count"] == 21 else "in_progress"
            item["evidence"] = (
                f"worker_branches tracks {payload['summary']['worker_count']} worktrees; "
                f"{payload['summary']['dirty_workers']} are currently dirty."
            )
        elif label == "Define merge order":
            item["status"] = "done" if synced.get("merge_order") else "in_progress"
            item["evidence"] = (
                f"merge_order defines {len(synced.get('merge_order', []))} waves; "
                f"merge_gates reports {payload['tracker_validation']['merge_gates']['summary']['blocked_waves']} blocked."
            )
        elif label == "Prevent shared-file conflicts":
            item["status"] = (
                "done" if remediation_summary["paths_with_overlaps"] == 0 else "in_progress"
            )
            item["evidence"] = (
                "The validator emits conflict_runbook, owner_runbook, branch_batches, merge_gates, and execution_queue; "
                f"paths_with_overlaps={remediation_summary['paths_with_overlaps']}; "
                f"critical_overlap_count={remediation_summary['critical_overlap_count']}; "
                f"ownership_decision_count={remediation_summary['ownership_decision_count']}; "
                f"lane_proposal_count={remediation_summary['lane_proposal_count']}; "
                f"tracker_edit_action_count={remediation_summary['tracker_edit_action_count']}; "
                f"release_only={triage_summary['release_only']}; "
                f"retask_or_stop={triage_summary['retask_or_stop']}; "
                f"retire_after_release={lifecycle_summary['retire_after_release']}; "
                f"unowned_path_count={lane_gap_summary['unowned_path_count']}; "
                f"allowed_path_additions={realignment_summary['allowed_path_addition_count']}."
            )
        elif label == "Keep an integration checklist":
            item["status"] = "done"
            item["evidence"] = "This tracker can now be refreshed in place with python3 scripts/rust_port_status.py --sync-tracker."
        elif label == "Continuously identify blockers":
            item["status"] = "in_progress"
            item["evidence"] = (
                f"tracker_validation.summary reports violation_count={validation_summary['violation_count']} and "
                f"execution_queue.summary reports operation_count={execution_summary['operation_count']}."
            )
        elif label == "Rebase the integration stack onto main":
            item["status"] = "blocked" if payload["summary"]["lead_vs_main_file_drift"] else "in_progress"
            item["evidence"] = (
                f"lead_head={payload['summary']['lead_head']} vs main_head={payload['summary']['main_head']}; "
                f"drift files={payload['summary']['lead_vs_main_file_drift']}."
            )
        elif label == "Complete the Python/TS-to-Rust port":
            item["status"] = "in_progress"
            item["evidence"] = (
                f"dirty_workers={payload['summary']['dirty_workers']}, "
                f"blocked_waves={payload['tracker_validation']['merge_gates']['summary']['blocked_waves']}, "
                "and compatibility lanes are still active."
            )

    synced["execution_protocol"] = [
        {
            "step": 1,
            "command": synced["sync_command"],
            "purpose": "Refresh the tracker JSON from live branch state before using any static evidence in this file.",
        },
        {
            "step": 2,
            "command": synced["summary_command"],
            "purpose": "Confirm current violation counts and whether any branch is merge-ready.",
        },
        {
            "step": 3,
            "command": synced["sync_drift_command"],
            "purpose": "See exactly which branches, overlap clusters, and baseline refs changed since the last tracker sync.",
        },
        {
            "step": 4,
            "command": synced["published_head_coverage_command"],
            "purpose": "Separate already-integrated published worker heads from residual local dirty work so post-merge blockers are not mistaken for missing merges.",
        },
        {
            "step": 5,
            "command": synced["objective_audit_command"],
            "purpose": "Audit the live state against the actual integration objective so requirement status does not depend on stale tracker prose.",
        },
        {
            "step": 6,
            "command": synced["objective_resolution_runbook_command"],
            "purpose": "Map each incomplete objective requirement to the exact next tracker commands and current live blockers needed to move it forward.",
        },
        {
            "step": 7,
            "command": "python3 scripts/rust_port_status.py --select tracker_validation.execution_queue.next_batch",
            "purpose": "Get the exact next branch-operation batch for the first blocked merge wave, including owner dependency releases.",
        },
        {
            "step": 8,
            "command": synced["triage_command"],
            "purpose": "Classify branches as salvageable, release-only, or retask/stop before executing cleanup work.",
        },
        {
            "step": 9,
            "command": synced["lifecycle_command"],
            "purpose": "Decide which branches should retire after cleanup versus remain alive to receive or keep lane-owned files.",
        },
        {
            "step": 10,
            "command": synced["lane_gap_command"],
            "purpose": "Resolve unowned paths by checking which ones fit existing lanes versus which ones need a new or expanded lane.",
        },
        {
            "step": 11,
            "command": synced["tracker_realignments_command"],
            "purpose": "Convert lane-gap findings into concrete tracker edits such as allowed_paths additions, owner reviews, or new-lane decisions.",
        },
        {
            "step": 12,
            "command": synced["conflict_runbook_command"],
            "purpose": "Clear shared-file conflicts path-by-path using owner-specific releaser commands and wave ordering.",
        },
        {
            "step": 13,
            "command": synced["decision_queue_command"],
            "purpose": "Resolve unresolved owner reviews and new-lane decisions in one prioritized queue before calling branches aligned.",
        },
        {
            "step": 14,
            "command": synced["lane_proposals_command"],
            "purpose": "Turn unresolved new-lane decisions into grouped lane-expansion or dedicated-lane proposals.",
        },
        {
            "step": 15,
            "command": synced["tracker_edit_plan_command"],
            "purpose": "Convert proposals and owner reviews into exact tracker mutations for allowed_paths and scope updates.",
        },
        {
            "step": 16,
            "command": synced["choice_runbook_command"],
            "purpose": "Review remaining ambiguous ownership choices with live branch pressure and wave context.",
        },
        {
            "step": 17,
            "command": synced["apply_tracker_edit_plan_command"],
            "purpose": "Apply all non-ambiguous tracker scope mutations so later cleanup uses updated lane ownership.",
        },
        {
            "step": 18,
            "command": synced["apply_tracker_choice_command"],
            "purpose": "Apply a selected owner for a remaining ambiguous lane choice, then refresh the tracker snapshot.",
        },
        {
            "step": 19,
            "command": synced["cleanup_impact_command"],
            "purpose": "Rank branch cleanup batches by early-wave unblock value, owner wait reduction, and critical overlap relief.",
        },
        {
            "step": 20,
            "command": synced["cleanup_forecast_command"],
            "purpose": "Project how much the top cleanup batches reduce overlaps, lead-only blockers, and owner wait lists before running them.",
        },
        {
            "step": 21,
            "command": "python3 scripts/rust_port_status.py --select tracker_validation.remediation.prioritized_actions",
            "purpose": "Get the current ordered cleanup queue for the integration lead.",
        },
        {
            "step": 22,
            "command": "python3 scripts/rust_port_status.py --select tracker_validation.remediation.branch_batches",
            "purpose": "Use compact per-branch command batches when executing the cleanup queue.",
        },
        {
            "step": 23,
            "command": "python3 scripts/rust_port_status.py --select tracker_validation.remediation.owner_runbook",
            "purpose": "Check which owner branches are waiting on releases and which owned paths are ready to receive.",
        },
        {
            "step": 24,
            "command": "python3 scripts/rust_port_status.py --select tracker_validation.merge_gates",
            "purpose": "Check which merge wave is blocked, why it is blocked, and which cleanup commands belong to that wave.",
        },
        {
            "step": 25,
            "command": synced["wave_unlock_forecast_command"],
            "purpose": "See when each merge wave becomes cleanup-ready under the current prioritized cleanup sequence and what manual blockers remain.",
        },
        {
            "step": 26,
            "command": synced["wave_critical_path_command"],
            "purpose": "Get the exact cleanup prefix and remaining manual gates required to unlock each merge wave.",
        },
        {
            "step": 27,
            "command": synced["unlock_ladder_command"],
            "purpose": "See the global stage transitions where cleanup batches newly make waves cleanup-ready or fully ready.",
        },
        {
            "step": 28,
            "command": synced["unlock_frontier_command"],
            "purpose": "Lift the current unlock frontier into one view: what to run now, which stage first yields a fully ready wave, and where the next manual gate appears.",
        },
        {
            "step": 29,
            "command": synced["next_unlock_batch_command"],
            "purpose": "Flatten the immediate unlock transition into one ordered command pack and preview the follow-up stage or manual gate.",
        },
        {
            "step": 30,
            "command": synced["next_unlock_brief_command"],
            "purpose": "Compress the next unlock batch into one operator brief: current commands, immediate payoff, follow-up wave effect, and the next manual gate.",
        },
        {
            "step": 31,
            "command": synced["stage_command_runbook_command"],
            "purpose": "Split the global unlock ladder into incremental branch command packs and carry forward the remaining manual gates at each stage.",
        },
        {
            "step": 32,
            "command": synced["wave_command_runbook_command"],
            "purpose": "Expand each wave critical path into ordered branch command batches plus the remaining manual gate commands.",
        },
    ]
    synced["blockers"] = build_tracker_blockers(payload)
    return synced


def write_tracker(path: Path, tracker: dict) -> None:
    path.write_text(json.dumps(tracker, indent=2) + "\n")


def sync_tracker_file(
    tracker_path: Path,
    tracker: dict,
    payload: dict,
    worktree_root: str,
    repo_root: Path,
) -> dict:
    tracker = sync_tracker_state(tracker, payload)
    write_tracker(tracker_path, tracker)
    populate_tracker_views(payload, tracker_path, tracker, worktree_root, repo_root)
    tracker = sync_tracker_state(tracker, payload)
    write_tracker(tracker_path, tracker)
    populate_tracker_views(payload, tracker_path, tracker, worktree_root, repo_root)
    return tracker


def populate_tracker_views(
    payload: dict,
    tracker_path: Path | None,
    tracker: dict | None,
    worktree_root: str,
    repo_root: Path,
) -> None:
    payload["tracker_validation"] = validate_tracker(payload, tracker_path, tracker, worktree_root)
    payload["tracker_validation"]["published_head_coverage"] = build_published_head_coverage(
        payload,
        repo_root,
    )
    payload["tracker_validation"]["merge_gates"] = build_merge_gates(payload, tracker)
    payload["tracker_validation"]["wave_unlock_forecast"] = build_wave_unlock_forecast(payload, tracker)
    payload["tracker_validation"]["wave_critical_path"] = build_wave_critical_path(
        payload["tracker_validation"]["wave_unlock_forecast"]
    )
    payload["tracker_validation"]["unlock_ladder"] = build_unlock_ladder(
        payload["tracker_validation"]["wave_critical_path"]
    )
    payload["tracker_validation"]["execution_queue"] = build_execution_queue(payload, tracker)
    payload["tracker_validation"]["wave_command_runbook"] = build_wave_command_runbook(
        payload,
        tracker,
    )
    payload["tracker_validation"]["stage_command_runbook"] = build_stage_command_runbook(
        payload,
        tracker,
    )
    payload["tracker_validation"]["unlock_frontier"] = build_unlock_frontier(
        payload,
        tracker,
    )
    payload["tracker_validation"]["next_unlock_batch"] = build_next_unlock_batch(
        payload,
        tracker,
    )
    payload["tracker_validation"]["next_unlock_brief"] = build_next_unlock_brief(
        payload,
        tracker,
    )
    payload["tracker_validation"]["objective_audit"] = build_objective_audit(
        payload,
        tracker,
    )
    payload["tracker_validation"]["objective_resolution_runbook"] = build_objective_resolution_runbook(
        payload,
        tracker,
    )


def validate_tracker(
    payload: dict, tracker_path: Path | None, tracker: dict | None, worktree_root: str
) -> dict:
    if tracker_path is None:
        return {
            "tracker_path": None,
            "loaded": False,
            "status": "not_requested",
            "summary": {},
            "violations": [],
            "branch_assessments": [],
        }
    if tracker is None:
        return {
            "tracker_path": str(tracker_path),
            "loaded": False,
            "status": "missing",
            "summary": {"violation_count": 1},
            "violations": [
                {
                    "severity": "high",
                    "code": "missing_tracker",
                    "detail": f"Tracker file not found: {tracker_path}",
                }
            ],
            "branch_assessments": [],
        }

    branch_plans = {
        entry["worktree"]: entry for entry in tracker.get("worker_branches", []) if "worktree" in entry
    }
    scope_patterns = tracker.get("scope_patterns", {})
    guardrails = tracker.get("shared_file_guardrails", {})
    lead_only = set(guardrails.get("lead_only", []))
    hotspot_paths = {entry["path"] for entry in guardrails.get("hotspots", []) if "path" in entry}
    ownership_overrides = tracker.get("live_conflict_clusters", [])
    violations: list[dict] = []
    assessments: list[dict] = []
    workers = {entry["worktree"]: entry for entry in payload["workers"]}

    if len(branch_plans) != len(workers):
        violations.append(
            {
                "severity": "high",
                "code": "tracker_worker_count_mismatch",
                "detail": (
                    f"Tracker has {len(branch_plans)} worker entries but live state has {len(workers)}."
                ),
            }
        )

    for worktree, entry in sorted(workers.items()):
        plan = branch_plans.get(worktree)
        active_files = entry["active_files"]
        patterns = plan.get("allowed_paths") or scope_patterns.get(worktree, [])
        has_scope_patterns = bool(patterns)
        unexpected_files: list[str] = []
        lead_only_hits = [path for path in active_files if path in lead_only]
        if plan is None:
            violations.append(
                {
                    "severity": "high",
                    "code": "untracked_worker_branch",
                    "worktree": worktree,
                    "detail": "Live worker branch is missing from tracker.worker_branches.",
                }
            )
            assessments.append(
                {
                    "worktree": worktree,
                    "status": entry["status"],
                    "tracked": False,
                    "scope_ok": False,
                    "guardrails_ok": not lead_only_hits,
                    "merge_ready": False,
                }
            )
            continue

        if plan.get("status") != entry["status"]:
            violations.append(
                {
                    "severity": "medium",
                    "code": "tracker_status_stale",
                    "worktree": worktree,
                    "detail": (
                        f"Tracker status is {plan.get('status')!r} but live status is {entry['status']!r}."
                    ),
                }
            )

        if active_files and not has_scope_patterns:
            violations.append(
                {
                    "severity": "medium",
                    "code": "missing_scope_patterns",
                    "worktree": worktree,
                    "detail": "Dirty worker has no scope_patterns entry for automated validation.",
                }
            )
        elif active_files:
            unexpected_files = [path for path in active_files if not matches_any(path, patterns)]
            if unexpected_files:
                violations.append(
                    {
                        "severity": "high",
                        "code": "scope_violation",
                        "worktree": worktree,
                        "detail": "Dirty files fall outside the planned ownership patterns.",
                        "paths": unexpected_files,
                    }
                )

        if lead_only_hits:
            violations.append(
                {
                    "severity": "high",
                    "code": "lead_only_violation",
                    "worktree": worktree,
                    "detail": "Worker is editing a reserved lead-only integration file.",
                    "paths": lead_only_hits,
                }
            )

        assessments.append(
            {
                "worktree": worktree,
                "status": entry["status"],
                "tracked": True,
                "merge_wave": plan.get("merge_wave"),
                "scope_ok": has_scope_patterns and not unexpected_files,
                "guardrails_ok": not lead_only_hits,
                "merge_ready": entry["status"] != "dirty"
                or (has_scope_patterns and not unexpected_files and not lead_only_hits),
            }
        )

    for overlap in payload["shared_file_overlaps"]:
        path = overlap["path"]
        severity = "high" if path in lead_only or path in hotspot_paths else "medium"
        code = "hotspot_overlap" if path in hotspot_paths else "shared_file_overlap"
        violations.append(
            {
                "severity": severity,
                "code": code,
                "detail": f"Multiple workers are editing {path}.",
                "path": path,
                "worktrees": overlap["worktrees"],
            }
        )

    if payload["summary"]["workers_with_upstream"] == 0:
        violations.append(
            {
                "severity": "medium",
                "code": "no_worker_upstreams",
                "detail": "No worker branch has an upstream configured.",
            }
        )

    if payload["summary"]["lead_vs_main_file_drift"]:
        violations.append(
            {
                "severity": "high",
                "code": "lead_behind_main",
                "detail": "Lead baseline is behind main.",
                "paths": payload["summary"]["lead_vs_main_file_drift"],
            }
        )

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    highest = max((severity_rank.get(item["severity"], 0) for item in violations), default=0)
    status = "ok" if not violations else {1: "warn", 2: "warn", 3: "fail"}[highest]
    remediation = build_remediation_plan(
        payload,
        branch_plans,
        lead_only,
        ownership_overrides,
        hotspot_paths,
        worktree_root,
    )
    sync_drift = build_sync_drift(payload, tracker)
    tracker_fingerprint = tracker.get("state_fingerprint")
    live_fingerprint = payload["summary"]["state_fingerprint"]
    return {
        "tracker_path": str(tracker_path),
        "loaded": True,
        "status": status,
        "summary": {
            "tracked_workers": len(branch_plans),
            "violation_count": len(violations),
            "high_severity_count": sum(1 for item in violations if item["severity"] == "high"),
            "medium_severity_count": sum(1 for item in violations if item["severity"] == "medium"),
            "dirty_worker_count": payload["summary"]["dirty_workers"],
            "merge_ready_workers": sum(1 for item in assessments if item["merge_ready"]),
            "live_state_fingerprint": live_fingerprint,
            "tracker_state_fingerprint": tracker_fingerprint,
            "tracker_matches_live_state": tracker_fingerprint == live_fingerprint,
        },
        "violations": violations,
        "branch_assessments": assessments,
        "remediation": remediation,
        "sync_drift": sync_drift,
    }


def main() -> int:
    args = parse_args()
    if bool(args.apply_tracker_choice_path) != bool(args.apply_tracker_choice_owner):
        print(
            "--apply-tracker-choice-path and --apply-tracker-choice-owner must be provided together",
            file=sys.stderr,
        )
        return 2
    if args.apply_tracker_edit_plan and args.apply_tracker_choice_path:
        print(
            "--apply-tracker-edit-plan cannot be combined with --apply-tracker-choice-* in one run",
            file=sys.stderr,
        )
        return 2
    root = Path(args.worktree_root).resolve()
    if not root.exists():
        print(f"worktree root does not exist: {root}", file=sys.stderr)
        return 2

    worktrees = discover_worktrees(root)
    if not worktrees:
        print(f"no agent-* worktrees found under {root}", file=sys.stderr)
        return 2

    lead = root / args.lead_worktree
    if lead not in worktrees:
        print(f"lead worktree not found: {lead}", file=sys.stderr)
        return 2

    workers = [path for path in worktrees if path != lead]
    if args.expected_workers < 0:
        print("--expected-workers must be non-negative", file=sys.stderr)
        return 2
    if len(workers) != args.expected_workers:
        print(
            f"expected {args.expected_workers} workers, found {len(workers)} under {root}",
            file=sys.stderr,
        )
        return 2

    lead_head = git(["rev-parse", "--short", "HEAD"], lead)
    lead_entry = collect_worktree_status(lead, args.main_ref, lead_head)
    worker_entries = [
        collect_worktree_status(path, args.main_ref, lead_head) for path in workers
    ]
    main_head = git(["rev-parse", "--short", args.main_ref], lead)
    main_subject = git(["log", "-1", "--format=%s", args.main_ref], lead)
    main_drift_files = [
        line
        for line in git(["diff", "--name-only", f"{lead_head}..{args.main_ref}"], lead).splitlines()
        if line
    ]
    tracker_path, tracker = load_tracker(args.tracker)
    if (
        args.sync_tracker
        or args.apply_tracker_edit_plan
        or args.apply_tracker_choice_path
    ) and (tracker_path is None or tracker is None):
        print(
            "--sync-tracker, --apply-tracker-edit-plan, and --apply-tracker-choice-* require an existing JSON tracker file",
            file=sys.stderr,
        )
        return 2
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "worktree_root": str(root),
        "lead": lead_entry,
        "workers": worker_entries,
        "summary": {
            "worker_count": len(worker_entries),
            "dirty_workers": sum(1 for entry in worker_entries if entry["status"] == "dirty"),
            "diverged_workers": sum(1 for entry in worker_entries if entry["status"] == "diverged"),
            "workers_with_upstream": sum(1 for entry in worker_entries if entry["upstream"]),
            "main_ref": args.main_ref,
            "main_head": main_head,
            "main_subject": main_subject,
            "lead_head": lead_head,
            "lead_vs_main_file_drift": main_drift_files,
            "state_fingerprint": compute_state_fingerprint(
                lead_entry,
                worker_entries,
                main_head,
                main_drift_files,
            ),
        },
        "shared_file_overlaps": build_overlap_map(worker_entries),
    }
    repo_root = Path.cwd()
    populate_tracker_views(payload, tracker_path, tracker, str(root), repo_root)
    if args.sync_tracker:
        tracker = sync_tracker_file(tracker_path, tracker, payload, str(root), repo_root)
        populate_tracker_views(payload, tracker_path, tracker, str(root), repo_root)
    if args.apply_tracker_edit_plan:
        tracker = apply_tracker_edit_plan_to_tracker(
            tracker,
            payload["tracker_validation"]["remediation"]["tracker_edit_plan"],
            payload["generated_at"],
        )
        write_tracker(tracker_path, tracker)
        populate_tracker_views(payload, tracker_path, tracker, str(root), repo_root)
        tracker = sync_tracker_file(tracker_path, tracker, payload, str(root), repo_root)
        populate_tracker_views(payload, tracker_path, tracker, str(root), repo_root)
    if args.apply_tracker_choice_path:
        tracker = apply_tracker_choice_to_tracker(
            tracker,
            payload["tracker_validation"]["remediation"]["choice_runbook"],
            args.apply_tracker_choice_path,
            args.apply_tracker_choice_owner,
            payload["generated_at"],
        )
        write_tracker(tracker_path, tracker)
        populate_tracker_views(payload, tracker_path, tracker, str(root), repo_root)
        tracker = sync_tracker_file(tracker_path, tracker, payload, str(root), repo_root)
        populate_tracker_views(payload, tracker_path, tracker, str(root), repo_root)
    try:
        selected = select_payload(payload, args.select)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(selected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
