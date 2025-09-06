#!/usr/bin/env python3
"""
GitHub Multi‑Agent Operator

Planner Agent → produces a linear plan made of [TOOL:<NAME>] <json-args> lines
Executor Agent → parses the plan and invokes concrete Tools (GitHub API wrappers)
Tools → thin functions that call GitHub REST API endpoints

Capabilities implemented:
- CREATE_REPO
- UPSERT_README (create/update README.md)
- RENAME_REPO
- DELETE_REPO (guarded by confirm=true)
- CREATE_GIST

Design features:
- Safe by default: destructive actions (DELETE_REPO) require explicit confirm=true.
- Dry‑run mode to preview API calls.
- Owner/org defaults from env or CLI flags.
- Planner via OpenAI API (optional). Fallback: trivial rule‑based parser for simple phrases.

Usage (examples):
  export GITHUB_TOKEN=ghp_your_token_here
  export GITHUB_OWNER=your_github_username  # or pass --owner on CLI

  # 1) Let the planner create a plan and run it:
  python github_operator.py --prompt "Create a public repo 'storyloom' with a README saying 'Hello World'"

  # 2) Dry‑run to see the calls without executing:
  python github_operator.py --prompt "rename repo storyloom to storyloom-pro" --dry-run

  # 3) Execute an explicit plan file (skip planner):
  cat > plan.txt <<EOF
  [TOOL:CREATE_REPO] {"name": "demo-repo", "private": false, "description": "demo"}
  [TOOL:UPSERT_README] {"repo": "demo-repo", "content": "# Demo\nHello from the operator."}
  [TOOL:RENAME_REPO] {"repo": "demo-repo", "new_name": "demo-repo-2"}
  [TOOL:DELETE_REPO] {"repo": "demo-repo-2", "confirm": true}
  [TOOL:CREATE_GIST] {"description": "Demo Gist", "files": {"file1.txt": {"content": "Hello World"}}}
  EOF
  python github_operator.py --plan-file plan.txt

Notes:
- Token scopes: For personal repos, a classic token with `repo` scope works. Fine‑grained tokens: enable repo:read/write, and delete repoes if you plan to delete.
- The UPSERT_README tool base64‑encodes content and will update if README.md exists (fetches sha); otherwise it creates it.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

API_URL = "https://api.github.com"

# ------------------------------
# Utilities
# ------------------------------

class OperatorError(Exception):
    pass


def b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


@dataclass
class Context:
    token: str
    owner: str
    org: Optional[str] = None
    dry_run: bool = False

    @property
    def auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "github-operator/1.0",
        }


# ------------------------------
# GitHub API Tools
# ------------------------------

class GitHubClient:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    # Helper to resolve owner/org creation URL
    def _create_repo_url(self) -> str:
        if self.ctx.org:
            return f"{API_URL}/orgs/{self.ctx.org}/repos"
        return f"{API_URL}/user/repos"

    def create_repo(
        self,
        name: str,
        private: bool = False,
        description: Optional[str] = None,
        auto_init: bool = False,
        gitignore_template: Optional[str] = None,
        license_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = self._create_repo_url()
        payload: Dict[str, Any] = {
            "name": name,
            "private": private,
            "auto_init": auto_init,
        }
        if description:
            payload["description"] = description
        if gitignore_template:
            payload["gitignore_template"] = gitignore_template
        if license_template:
            payload["license_template"] = license_template

        if self.ctx.dry_run:
            return {"dry_run": True, "action": "create_repo", "payload": payload}

        r = requests.post(url, headers=self.ctx.auth_headers, json=payload)
        if r.status_code >= 400:
            raise OperatorError(f"Create repo failed: {r.status_code} {r.text}")
        return r.json()

    def delete_repo(self, repo: str, owner: Optional[str] = None) -> Dict[str, Any]:
        owner = owner or self.ctx.owner
        url = f"{API_URL}/repos/{owner}/{repo}"
        if self.ctx.dry_run:
            return {"dry_run": True, "action": "delete_repo", "url": url}
        r = requests.delete(url, headers=self.ctx.auth_headers)
        if r.status_code not in (204, 202):
            raise OperatorError(f"Delete repo failed: {r.status_code} {r.text}")
        return {"ok": True}

    def rename_repo(self, repo: str, new_name: str, owner: Optional[str] = None) -> Dict[str, Any]:
        owner = owner or self.ctx.owner
        url = f"{API_URL}/repos/{owner}/{repo}"
        payload = {"name": new_name}
        if self.ctx.dry_run:
            return {"dry_run": True, "action": "rename_repo", "url": url, "payload": payload}
        r = requests.patch(url, headers=self.ctx.auth_headers, json=payload)
        if r.status_code >= 400:
            raise OperatorError(f"Rename repo failed: {r.status_code} {r.text}")
        return r.json()

    def get_file(self, repo: str, path: str, owner: Optional[str] = None) -> Optional[Dict[str, Any]]:
        owner = owner or self.ctx.owner
        url = f"{API_URL}/repos/{owner}/{repo}/contents/{path}"
        r = requests.get(url, headers=self.ctx.auth_headers)
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise OperatorError(f"Get file failed: {r.status_code} {r.text}")
        return r.json()

    def upsert_file(
        self,
        repo: str,
        path: str,
        content: str,
        message: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        owner = owner or self.ctx.owner
        url = f"{API_URL}/repos/{owner}/{repo}/contents/{path}"
        existing = None
        try:
            existing = self.get_file(repo, path, owner)
        except OperatorError as e:
            # Treat unknown errors as fatal
            raise e

        payload: Dict[str, Any] = {
            "message": message or f"Upsert {path}",
            "content": b64(content),
        }
        if existing and "sha" in existing:
            payload["sha"] = existing["sha"]

        if self.ctx.dry_run:
            return {
                "dry_run": True,
                "action": "upsert_file",
                "url": url,
                "payload": {k: v for k, v in payload.items() if k != "content"} | {"content_preview": content[:80]},
            }
        r = requests.put(url, headers=self.ctx.auth_headers, json=payload)
        if r.status_code >= 400:
            raise OperatorError(f"Upsert file failed: {r.status_code} {r.text}")
        return r.json()
    
    def create_gist(
        self, description: str, files: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        url = f"{API_URL}/gists"
        payload = {"description": description, "files": files}
        if self.ctx.dry_run:
            return {"dry_run": True, "action": "create_gist", "url": url, "payload": payload}
        r = requests.post(url, headers=self.ctx.auth_headers, json=payload)
        if r.status_code >= 400:
            raise OperatorError(f"Create gist failed: {r.status_code} {r.text}")
        return r.json()


# ------------------------------
# Plan format & parsing
# ------------------------------

PLAN_NOTE = """
PLAN FORMAT (strict): one action per line. Lines must start with
  [TOOL:CREATE_REPO] {json}
  [TOOL:UPSERT_README] {json}
  [TOOL:RENAME_REPO] {json}
  [TOOL:DELETE_REPO] {json}
  [TOOL:CREATE_GIST] {json}

JSON ARGUMENTS (schemas):
- CREATE_REPO: {"name": str, "private"?: bool, "description"?: str, "auto_init"?: bool}
- UPSERT_README: {"repo": str, "content": str, "message"?: str, "path"?: str}
  (path defaults to "README.md")
- RENAME_REPO: {"repo": str, "new_name": str}
- DELETE_REPO: {"repo": str, "confirm": bool}
    (confirm must be true to proceed)
- CREATE_GIST: {"description": str, "files": {"filename": {"content": str}}}
Optional common fields: "owner": str (defaults to context.owner)
""".strip()

TOOL_LINE_RE = re.compile(r"^\s*\[TOOL:(?P<name>[A-Z_]+)\]\s*(?P<json>\{.*\})\s*$")

@dataclass
class Step:
    name: str
    args: Dict[str, Any]


def parse_plan(plan_text: str) -> List[Step]:
    steps: List[Step] = []
    for ln in plan_text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = TOOL_LINE_RE.match(ln)
        if not m:
            raise OperatorError(f"Invalid plan line: {ln}\nExpected format: [TOOL:NAME] {{json}}")
        name = m.group("name").upper()
        try:
            args = json.loads(m.group("json"))
        except json.JSONDecodeError as e:
            raise OperatorError(f"Invalid JSON on line: {ln}\nError: {e}")
        steps.append(Step(name=name, args=args))
    if not steps:
        raise OperatorError("Empty plan: no tool lines found")
    return steps


# ------------------------------
# Executor
# ------------------------------

class Executor:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.gh = GitHubClient(ctx)

    def run(self, steps: List[Step]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i, step in enumerate(steps, 1):
            try:
                res = self._run_step(step)
                results.append({"step": i, "tool": step.name, "ok": True, "result": res})
            except Exception as e:
                results.append({"step": i, "tool": step.name, "ok": False, "error": str(e)})
                break  # stop on first failure; adjust if you prefer best-effort
        return results

    def _run_step(self, step: Step) -> Dict[str, Any]:
        name = step.name
        a = step.args
        owner = a.get("owner") or self.ctx.owner

        if name == "CREATE_REPO":
            return self.gh.create_repo(
                name=a["name"],
                private=bool(a.get("private", False)),
                description=a.get("description"),
                auto_init=bool(a.get("auto_init", False)),
                gitignore_template=a.get("gitignore_template"),
                license_template=a.get("license_template"),
            )

        if name == "UPSERT_README":
            repo = a["repo"]
            path = a.get("path", "README.md")
            content = a["content"]
            message = a.get("message", f"Update {path}")
            return self.gh.upsert_file(repo=repo, path=path, content=content, message=message, owner=owner)

        if name == "RENAME_REPO":
            return self.gh.rename_repo(repo=a["repo"], new_name=a["new_name"], owner=owner)

        if name == "DELETE_REPO":
            if not a.get("confirm", False):
                raise OperatorError("DELETE_REPO requires confirm=true")
            return self.gh.delete_repo(repo=a["repo"], owner=owner)
        if name == "CREATE_GIST":
            return self.gh.create_gist(description=a["description"], files=a["files"])

        raise OperatorError(f"Unknown tool: {name}")


# ------------------------------
# Planner (LLM optional) & prompts
# ------------------------------

PLANNER_SYSTEM_PROMPT = (
    "You are a strict planner that outputs only actionable tool calls, one per line.\n"
    + PLAN_NOTE
    + "\nRules:\n"
      "- Output ONLY tool lines; no commentary.\n"
      "- Prefer minimal steps.\n"
      "- For README text, include the exact content string in JSON (use \n for newlines).\n"
      "- Use DELETE_REPO only if the user clearly asks deletion. Always set confirm=true.\n"
)


def call_openai_planner(user_prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """Optional: use OpenAI to produce a plan. Returns None if API key missing."""
    api_key = 'YOUR_OPEN_AI_KEY'
    if not api_key:
        return None
    try:
        import requests as _req

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        resp = _req.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise OperatorError(f"Planner API error: {e}")


# A tiny fallback interpreter for a few simple phrasings (best-effort)

def naive_fallback_plan(user_prompt: str) -> str:
    up = user_prompt.lower()
    # create repo
    m = re.search(r"create (?:a|the)?\s*repo(?:sitory)?\s*(?:called|named)?\s*'?(?P<name>[\w.-]+)'?", up)
    if m:
        name = m.group("name")
        # README content
        readme = None
        m2 = re.search(r"readme (?:saying|with|containing) '?(?P<msg>[^']+)'?", up)
        if m2:
            readme = m2.group("msg").strip()
        lines = [f"[TOOL:CREATE_REPO] {{\"name\": \"{name}\", \"private\": false}}"]
        if readme:
            content = json.dumps(readme)[1:-1]  # escape
            lines.append(f"[TOOL:UPSERT_README] {{\"repo\": \"{name}\", \"content\": \"{content}\"}}")
        return "\n".join(lines)

    # rename repo
    m = re.search(r"rename repo '?(?P<src>[\w.-]+)'? to '?(?P<dst>[\w.-]+)'?", up)
    if m:
        return f"[TOOL:RENAME_REPO] {{\"repo\": \"{m.group('src')}\", \"new_name\": \"{m.group('dst')}\"}}"

    # delete repo
    m = re.search(r"delete repo '?(?P<name>[\w.-]+)'?", up)
    if m:
        return f"[TOOL:DELETE_REPO] {{\"repo\": \"{m.group('name')}\", \"confirm\": true}}"

    # fallback failure
    raise OperatorError("Fallback planner couldn't interpret the prompt. Provide an explicit plan or set OPENAI_API_KEY.")


# ------------------------------
# CLI glue
# ------------------------------

def run_from_prompt(ctx: Context, prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
    plan = call_openai_planner(prompt) or naive_fallback_plan(prompt)
    steps = parse_plan(plan)
    ex = Executor(ctx)
    results = ex.run(steps)
    return plan, results


def run_from_plan_text(ctx: Context, plan_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    steps = parse_plan(plan_text)
    ex = Executor(ctx)
    results = ex.run(steps)
    return plan_text, results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GitHub Multi‑Agent Operator")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Natural language request (uses planner)")
    src.add_argument("--plan-file", help="Path to a plan file with [TOOL:...] lines")
    p.add_argument("--owner", help="GitHub owner/username (default $GITHUB_OWNER)")
    p.add_argument("--org", help="GitHub organization (if creating in org)")
    p.add_argument("--dry-run", action="store_true", help="Print actions without executing API calls")

    args = p.parse_args(argv)

    token = "<YOUR_GITHUB_PAT>" # The token should be a classic token with the delete, rename .. etc options enabled while creating the personal auth token.
    if not token:
        print("ERROR: Set GITHUB_TOKEN in environment.", file=sys.stderr)
        return 2

    owner = args.owner or os.getenv("GITHUB_OWNER")
    if not owner:
        print("ERROR: Provide --owner or set GITHUB_OWNER in environment.", file=sys.stderr)
        return 2

    ctx = Context(token=token, owner=owner, org=args.org, dry_run=args.dry_run)

    try:
        if args.prompt:
            plan, results = run_from_prompt(ctx, args.prompt)
        else:
            with open(args.plan_file, "r", encoding="utf-8") as f:
                plan_text = f.read()
            plan, results = run_from_plan_text(ctx, plan_text)
    except OperatorError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1

    # Pretty print
    print("\n=== PLAN ===")
    print(plan.strip())
    print("\n=== EXECUTION RESULTS ===")
    for r in results:
        status = "OK" if r.get("ok") else "FAIL" if not r.get("ok") else "OK"
        print(f"Step {r['step']:02d} [{r['tool']}]: {status}")
        if r.get("ok"):
            # Show minimal payload
            res = r.get("result")
            if isinstance(res, dict):
                subset_keys = ["html_url", "name", "full_name", "ok", "dry_run", "action", "url"]
                summary = {k: v for k, v in res.items() if k in subset_keys}
                if summary:
                    print("  ", summary)
        else:
            print("   Error:", r.get("error"))

    # Exit code indicates overall success
    last = results[-1] if results else {"ok": False}
    return 0 if last.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
