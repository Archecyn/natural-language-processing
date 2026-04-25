"""
azure_devops_client.py
──────────────────────
The client requires a personal access token with at least **Code (read &
write)** permissions. The repository is identified by a string of the form
``organization/project/repo`` (the values you see in the URL when browsing
a repo on dev.azure.com).
"""

import base64
import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

AZDO_API = "https://dev.azure.com"
API_VERSION = "7.0"  # stable version for Git APIs


class AzureDevOpsClient:
    """Interacts with a single Azure DevOps Git repository via REST.

    The ``repo`` argument should be ``organization/project/repo``.
    Internally we build a base URL that looks like:
    ``https://dev.azure.com/{org}/{project}/_apis/git/repositories/{repo}``.
    """

    def __init__(self, token: str, repo_full_name: str):
        self.token = token
        self.repo_full_name = repo_full_name
        parts = repo_full_name.split("/")
        if len(parts) != 3:
            raise ValueError(
                "repo_full_name must be 'organization/project/repo'"
            )
        org, project, repo = parts
        self.base_url = (
            f"{AZDO_API}/{org}/{project}/_apis/git/repositories/{repo}"
        )
        self.session = requests.Session()
        self.session.auth = ("", token)
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        # validate immediately
        self._validate()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}?api-version={API_VERSION}"

    def _get(self, path: str, **kwargs) -> dict | list:
        url = self._url(path)
        print(f"[azure_devops_client] GET -> {url}")
        resp = self.session.get(url, **kwargs)
        self._raise(resp)
        return resp.json()

    def _post(self, path: str, json: dict | list) -> dict:
        url = self._url(path)
        print(f"[azure_devops_client] POST -> {url}")
        resp = self.session.post(url, json=json)
        self._raise(resp)
        return resp.json()

    def _raise(self, resp: requests.Response):
        if not resp.ok:
            try:
                msg = resp.json().get("message", resp.text)
            except Exception:
                msg = resp.text
            raise RuntimeError(f"Azure DevOps API {resp.status_code}: {msg}")

    def _validate(self):
        # Fetch repository metadata to ensure the token and repo are valid
        data = self._get("")
        log.info(
            f"Connected to repo: {data['name']} (default branch: {data['defaultBranch']})"
        )

    # ── Public API (GitHub-like) ──────────────────────────────────────────────

    def list_code_files(
        self,
        branch: str = "main",
        extensions: list[str] | None = None,
        max_files: int = 20,
        folder: str | None = None,
    ) -> list[str]:
        """Return list of file paths in the repo (filtered by extension / folder)."""
        # Azure DevOps returns items under the ``items`` endpoint.
        params = {
            "scopePath": folder or "/",
            "recursionLevel": "Full",
            "versionDescriptor.version": branch,
            "includeContentMetadata": "false",
        }
        resp = self.session.get(
            f"{self.base_url}/items?api-version={API_VERSION}",
            params=params,
        )
        self._raise(resp)
        data = resp.json()
        items = data.get("value", [])
        extensions = extensions or [".py"]
        paths = [
            item["path"].lstrip("/")
            for item in items
            if not item.get("isFolder", False)
            and any(item["path"].lower().endswith(ext) for ext in extensions)
            and not self._is_vendor_path(item["path"])
        ]
        if folder:
            folder = folder.strip("/")
            if folder:
                paths = [p for p in paths if p.startswith(f"{folder}/") or p == folder]
        log.debug(f"Total matching files in tree: {len(paths)}")
        if max_files == 0:
            return paths
        return paths[:max_files]

    @staticmethod
    def _is_vendor_path(path: str) -> bool:
        skip = {
            "node_modules", "vendor", ".venv", "venv", "__pycache__",
            "dist", "build", ".git", "migrations", "static",
        }
        parts = set(path.split("/"))
        return bool(parts & skip)

    def get_file_content(self, path: str, branch: str = "main") -> str:
        """Fetch raw text content of a file from a specific branch."""
        params = {
            "path": "/" + path,
            "versionDescriptor.version": branch,
            "includeContent": "true",
        }
        resp = self.session.get(
            f"{self.base_url}/items?api-version={API_VERSION}",
            params=params,
        )
        if resp.status_code == 404:
            raise RuntimeError(f"File not found: {path}")
        self._raise(resp)
        data = resp.json()
        return data.get("content", "")

    def get_branch_sha(self, branch: str) -> str:
        """Return the commit SHA of the head of *branch*."""
        data = self._get(f"/refs?filter=heads/{branch}")
        values = data.get("value", [])
        if not values:
            raise RuntimeError(f"Branch '{branch}' not found")
        return values[0]["objectId"]

    def create_branch(self, new_branch: str, from_branch: str) -> None:
        """Create a new branch pointing at the tip of *from_branch*."""
        # delete existing ref if present (idempotent)
        try:
            self.session.delete(
                f"{self.base_url}/refs?filter=heads/{new_branch}&api-version={API_VERSION}"
            )
        except Exception:
            pass
        sha = self.get_branch_sha(from_branch)
        body = [
            {
                "name": f"refs/heads/{new_branch}",
                "oldObjectId": "0000000000000000000000000000000000000000",
                "newObjectId": sha,
            }
        ]
        self._post("/refs", json=body)
        log.info(f"Branch '{new_branch}' created from '{from_branch}' @ {sha[:7]}")

    def file_exists(self, path: str, branch: str) -> bool:
        try:
            self.get_file_content(path, branch)
            return True
        except RuntimeError:
            return False

    def commit_file(
        self,
        path: str,
        content: str,
        message: str,
        branch: str,
    ) -> None:
        """Create or update a file on *branch* with a commit."""
        base_sha = self.get_branch_sha(branch)
        change_type = "edit" if self.file_exists(path, branch) else "add"
        change = {
            "changeType": change_type,
            "item": {"path": "/" + path},
            "newContent": {"content": content, "contentType": "rawtext"},
        }
        body = {
            "refUpdates": [
                {"name": f"refs/heads/{branch}", "oldObjectId": base_sha}
            ],
            "commits": [
                {"comment": message, "changes": [change]}
            ],
        }
        self._post("/pushes", json=body)
        # throttle to avoid secondary limits
        time.sleep(0.5)

    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
    ) -> str:
        """Open a PR and return its web URL."""
        payload = {
            "sourceRefName": f"refs/heads/{head_branch}",
            "targetRefName": f"refs/heads/{base_branch}",
            "title": title,
            "description": body,
        }
        data = self._post("/pullrequests", json=payload)
        # ``url`` is the REST URL; use ``remoteUrl`` for the web link
        return data.get("remoteUrl", data.get("url", ""))
