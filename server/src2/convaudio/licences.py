"""Licence policy enforcement and auditing."""

from __future__ import annotations

import fnmatch
import importlib.metadata
from pathlib import Path
from typing import Any

import yaml

from convaudio.errors import LicenceViolation

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "licences.yaml"


def load_licence_policy(policy_path: Path | str | None = None) -> dict[str, Any]:
    """Load licence policy YAML file."""
    path = Path(policy_path) if policy_path else DEFAULT_POLICY_PATH
    if not path.exists():
        # Fallback to current working directory or search parent directories
        alt_paths = [
            Path.cwd() / "licences.yaml",
            Path.cwd() / "src2" / "licences.yaml",
            Path(__file__).resolve().parent / "licences.yaml",
        ]
        for alt in alt_paths:
            if alt.exists():
                path = alt
                break
    if not path.exists():
        raise FileNotFoundError(f"Licence policy file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("licences.yaml must contain a top-level dictionary")
    return data


def _normalize_licence_str(lic: str) -> str:
    return lic.strip()


def normalize_licence_spdx(lic: str) -> str:
    """Normalize common licence strings and classifiers to canonical SPDX IDs."""
    raw = lic.strip()
    norm = raw.upper()

    # Extract from classifier if present
    if "::" in norm:
        norm = norm.split("::")[-1].strip()

    if "MIT" in norm:
        return "MIT"
    if "APACHE" in norm and ("2" in norm or "2.0" in norm):
        return "Apache-2.0"
    if "APACHE SOFTWARE" in norm or norm == "APACHE":
        return "Apache-2.0"
    if "BSD" in norm:
        if "2" in norm or "SIMPLIFIED" in norm:
            return "BSD-2-Clause"
        if "4" in norm or "ORIGINAL" in norm:
            return "BSD-4-Clause"
        return "BSD-3-Clause"
    if "ISC" in norm:
        return "ISC"
    if "CC-BY-4.0" in norm or "CREATIVE COMMONS ATTRIBUTION 4.0" in norm:
        return "CC-BY-4.0"
    if norm in ("N/A", "NONE"):
        return "N/A"

    return raw


def is_licence_denied(licence: str, denied_patterns: list[str]) -> bool:
    """Check if licence string matches any denied pattern."""
    norm = licence.strip().upper()
    for pattern in denied_patterns:
        pat_upper = pattern.upper()
        if fnmatch.fnmatch(norm, pat_upper):
            return True
        if "NC" in pat_upper and "NC" in norm:
            return True
        if "GPL" in pat_upper and "GPL" in norm and "LGPL" not in pat_upper:
            return True
    return False


def is_licence_allowed(licence: str, policy: dict[str, Any]) -> bool:
    """Check if licence is explicitly allowed (or allowed with attribution)."""
    norm = licence.strip()
    denied_patterns: list[str] = policy.get("denied_patterns", [])

    if is_licence_denied(norm, denied_patterns):
        return False

    spdx = normalize_licence_spdx(norm)
    if is_licence_denied(spdx, denied_patterns):
        return False

    allowed: list[str] = policy.get("allowed", [])
    allowed_attr: list[str] = policy.get("allowed_with_attribution", [])

    all_allowed = {a.strip().upper() for a in allowed} | {a.strip().upper() for a in allowed_attr}
    return spdx.upper() in all_allowed


def assert_licence_allowed(
    code_licence: str, weights_licence: str, policy: dict[str, Any] | None = None
) -> None:
    """Verify that both code and weights licences satisfy the allowlist policy.

    Raises LicenceViolation on deny-list or unknown values.
    """
    if policy is None:
        policy = load_licence_policy()

    denied_patterns = policy.get("denied_patterns", [])

    # Check code licence
    if is_licence_denied(code_licence, denied_patterns) or not is_licence_allowed(code_licence, policy):
        raise LicenceViolation(
            f"Code licence '{code_licence}' violates licence policy (denied or not allowed)."
        )

    # Check weights licence
    if is_licence_denied(weights_licence, denied_patterns) or not is_licence_allowed(weights_licence, policy):
        raise LicenceViolation(
            f"Weights licence '{weights_licence}' violates licence policy (denied or not allowed)."
        )


def get_convaudio_runtime_distributions() -> list[importlib.metadata.Distribution]:
    """Resolve transitive runtime dependencies of convaudio."""
    import re
    seen = {"convaudio"}
    queue = ["convaudio"]
    while queue:
        cur = queue.pop()
        try:
            reqs = importlib.metadata.requires(cur) or []
        except Exception:
            continue
        for req in reqs:
            if "extra ==" in req:
                continue
            m = re.match(r"^([a-zA-Z0-9_\-]+)", req)
            if m:
                dep = m.group(1).lower().replace("_", "-")
                if dep not in seen:
                    seen.add(dep)
                    queue.append(dep)
    dists: list[importlib.metadata.Distribution] = []
    for d in importlib.metadata.distributions():
        d_name = (d.metadata["Name"] if "Name" in d.metadata else "").lower().replace("_", "-")
        if d_name in seen:
            dists.append(d)
    return dists


def check_installed_distributions(
    policy: dict[str, Any] | None = None,
    extra_manifest: list[dict[str, str]] | None = None,
    all_dists: bool = False,
) -> tuple[bool, list[str]]:
    """Scan installed distributions and declared weights manifest.

    Returns (is_clean, list_of_violations).
    """
    if policy is None:
        policy = load_licence_policy()

    violations: list[str] = []
    approved_components = policy.get("approved_components", {})

    # Check extra manifest first (test injections or model declarations)
    if extra_manifest:
        for entry in extra_manifest:
            name = entry.get("name", "unknown")
            code_lic = entry.get("code_licence", "UNKNOWN")
            weights_lic = entry.get("weights_licence", "UNKNOWN")
            try:
                assert_licence_allowed(code_lic, weights_lic, policy)
            except LicenceViolation as e:
                violations.append(f"Manifest entry '{name}': {e}")

    try:
        if all_dists:
            dists = list(importlib.metadata.distributions())
        else:
            dists = get_convaudio_runtime_distributions()
            if not dists:
                dists = list(importlib.metadata.distributions())
    except Exception:
        dists = []

    for dist in dists:
        dist_name = (dist.metadata["Name"] if "Name" in dist.metadata else "unknown").lower()
        lic = dist.metadata["License"] if "License" in dist.metadata else ""

        # Check against approved components map
        matched_approved = None
        for app_name, app_spec in approved_components.items():
            if app_name.lower() == dist_name or app_name.lower().replace("-", "_") == dist_name.replace("-", "_"):
                matched_approved = app_spec
                break

        if matched_approved:
            code_lic = matched_approved.get("code_licence", "UNKNOWN")
            weights_lic = matched_approved.get("weights_licence", "UNKNOWN")
            try:
                assert_licence_allowed(code_lic, weights_lic, policy)
            except LicenceViolation as e:
                violations.append(f"{dist_name}: {e}")
            continue

        # Look for classifiers
        classifiers = dist.metadata.get_all("Classifier") or []
        lic_classifiers = [c for c in classifiers if c.startswith("License ::")]

        # Determine effective licence candidate
        candidate = lic.strip() if (lic and lic.strip() and lic.strip() != "UNKNOWN") else ""
        if not candidate and lic_classifiers:
            candidate = lic_classifiers[0]

        if candidate:
            try:
                if not is_licence_allowed(candidate, policy):
                    violations.append(f"Distribution '{dist_name}' has disallowed licence '{candidate[:60]}'")
            except Exception as e:
                violations.append(f"Distribution '{dist_name}': {e}")

    return len(violations) == 0, violations
