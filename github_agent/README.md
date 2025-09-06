# Github Agent

The above code has three important segments:
* Planner: An LLM agent that interprets the prompt from the user and drafts out a plan based on what the user wants to do.
* Executor: A python function that runs commands based on the plan drafter by the planner.
* Tools: Functions built to performs necessary actions whenever they are called by the executor.

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
