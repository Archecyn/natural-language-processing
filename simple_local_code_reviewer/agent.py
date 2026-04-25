"""
Azure DevOps AI Code Review Agent
=================================
Workflow is identical to the GitHub variant, but all interactions are
performed against Azure DevOps Git repositories via the REST API.
"""

import os
import sys
import logging
from typing import Literal

from azure_devops_client import AzureDevOpsClient
from llm_client import LLMClient
from code_analyser import CodeAnalyser
from pr_creator import PRCreator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Supported review modes ────────────────────────────────────────────────────
ReviewMode = Literal["errors", "optimise", "both"]


class AzureDevOpsAIAgent:
    """Top-level orchestrator for the Azure DevOps code‑review workflow."""

    def __init__(
        self,
        azure_token: str,
        repo_full_name: str,             # e.g. "org/project/repo"
        mode: ReviewMode = "both",
        base_branch: str = "main",
        pr_branch: str = "ai-code-review",
        max_files: int = 20,
        file_extensions: list[str] | None = None,
        folder: str | None = None,      # e.g. "src" or "app/components"
        model_size: str | None = None,  # e.g. "3b", "7b", "14b", "32b"
    ):
        self.mode = mode
        self.base_branch = base_branch
        self.pr_branch = pr_branch
        self.max_files = max_files
        self.file_extensions = file_extensions or [
            ".py", ".js", ".ts", ".jsx", ".tsx", ".ipynb",
            ".go", ".rs", ".java", ".rb", ".php",
            ".cs", ".cpp", ".c", ".swift", ".kt",
            ".sql", ".sh", ".yaml", ".yml", ".json",
        ]
        self.folder = folder
        self.model_size = model_size

        log.info("Initialising Azure DevOps client …")
        self.azdo = AzureDevOpsClient(azure_token, repo_full_name)

        log.info("Loading local LLM (this may take a moment) …")
        self.llm = LLMClient(model_size=self.model_size)

        self.analyser = CodeAnalyser(self.llm, mode)
        self.pr_creator = PRCreator(self.azdo)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> str:
        """Execute the full agentic workflow and return the PR URL."""

        log.info("=" * 60)
        log.info("Azure DevOps AI Agent — starting run")
        log.info(f"  Repo : {self.azdo.repo_full_name}")
        log.info(f"  Mode : {self.mode}")
        log.info(f"  Base : {self.base_branch}  →  PR branch: {self.pr_branch}")
        if self.folder:
            log.info(f"  Folder: {self.folder}")
        log.info("=" * 60)

        # 1. Fetch file list from Azure DevOps
        log.info("Step 1/4 — Fetching repository file tree …")
        files = self.azdo.list_code_files(
            branch=self.base_branch,
            extensions=self.file_extensions,
            max_files=self.max_files,
            folder=self.folder,
        )
        if not files:
            raise RuntimeError("No code files found in the repository.")
        log.info(f"  Found {len(files)} code file(s).")

        # 2. Analyse & rewrite files with the LLM
        log.info("Step 2/4 — Analysing files with local LLM …")
        changes: dict[str, str] = {}          # path → new_content
        summaries: dict[str, str] = {}        # path → human-readable summary

        for i, file_path in enumerate(files, 1):
            log.info(f"  [{i}/{len(files)}] {file_path}")
            original = self.azdo.get_file_content(file_path, branch=self.base_branch)

            result = self.analyser.analyse(file_path, original)

            if result is None:
                log.info("    → No changes suggested.")
                continue

            new_content, summary = result
            if new_content.strip() == original.strip():
                log.info("    → LLM returned identical content, skipping.")
                continue

            changes[file_path] = new_content
            summaries[file_path] = summary
            log.info(f"    → Changes recorded. ({len(summary)} char summary)")

        if not changes:
            log.info("No files were modified — nothing to PR.")
            return "(no PR created — no changes found)"

        log.info(f"Step 3/4 — {len(changes)} file(s) modified.")

        # 3. Push changes to a new branch & open PR
        log.info("Step 4/4 — Creating branch and opening Pull Request …")
        pr_url = self.pr_creator.create_pr(
            base_branch=self.base_branch,
            pr_branch=self.pr_branch,
            mode=self.mode,
            changes=changes,
            summaries=summaries,
        )

        log.info("=" * 60)
        log.info(f"✅  Pull Request opened: {pr_url}")
        log.info("=" * 60)
        return pr_url


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Azure DevOps AI Code Review Agent using local Qwen2.5-Coder"
    )
    parser.add_argument(
        "repo",
        help="Azure DevOps repo in organization/project/repo format, e.g. myorg/myproj/myrepo",
    )
    parser.add_argument(
        "--mode",
        choices=["errors", "optimise", "both"],
        default="both",
        help="Review mode (default: both)",
    )
    parser.add_argument("--base-branch", default="main", help="Branch to review (default: main)")
    parser.add_argument("--pr-branch", default="ai-code-review", help="New branch name for the PR")
    parser.add_argument("--max-files", type=int, default=20, help="Max files to analyse (0 = unlimited)")
    parser.add_argument("--folder", help="Focus on a specific folder within the repo, e.g. 'src' or 'app/components'")
    parser.add_argument("--file-extensions", help="Comma-separated file extensions to analyse, e.g. '.py,.js,.sql' (default: all supported types)")
    parser.add_argument("--model-size", choices=["3b", "7b", "14b", "32b"], default="7b", help="Model size to use (default: 7b)")
    parser.add_argument(
        "--token",
        default=os.environ.get("AZDO_TOKEN"),
        help="Azure DevOps Personal Access Token (or set AZDO_TOKEN env var)",
    )
    args = parser.parse_args()

    if not args.token:
        print("ERROR: Azure DevOps token required. Use --token or set AZDO_TOKEN env var.")
        sys.exit(1)

    file_extensions = None
    if args.file_extensions:
        file_extensions = [ext.strip() for ext in args.file_extensions.split(',')]

    agent = AzureDevOpsAIAgent(
        azure_token=args.token,
        repo_full_name=args.repo,
        mode=args.mode,
        base_branch=args.base_branch,
        pr_branch=args.pr_branch,
        max_files=args.max_files,
        folder=args.folder,
        file_extensions=file_extensions,
        model_size=args.model_size,
    )
    pr_url = agent.run()
    print(f"\nPR URL: {pr_url}")


if __name__ == "__main__":
    main()
