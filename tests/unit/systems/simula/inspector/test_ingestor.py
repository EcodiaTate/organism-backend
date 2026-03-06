"""
Unit tests for Inspector TargetIngestor (Phase 3).

Tests attack surface detection across Python (AST-based), JavaScript/TypeScript
(regex-based), and Solidity (regex-based) codebases. Uses real temporary files
written to disk to validate the scanning pipeline end-to-end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from systems.simula.inspector.ingestor import TargetIngestor, _should_skip_path
from systems.simula.inspector.types import AttackSurface, AttackSurfaceType
from systems.simula.inspector.workspace import TargetWorkspace

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixture: build a workspace with source files ────────────────────────────


def _write_file(root: Path, rel_path: str, content: str) -> Path:
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return full


def _make_ingestor(tmp_path: Path) -> TargetIngestor:
    ws = TargetWorkspace(root=tmp_path, workspace_type="external_repo")
    return TargetIngestor(workspace=ws)


# ── Path Skipping ───────────────────────────────────────────────────────────


class TestPathSkipping:
    @pytest.mark.parametrize("path", [
        "node_modules/express/index.js",
        ".git/objects/abc",
        "__pycache__/foo.pyc",
        ".venv/lib/python3.12/site.py",
        "venv/lib/site.py",
        "dist/bundle.js",
        "build/output.js",
        ".next/server/page.js",
    ])
    def test_skip_non_source_dirs(self, path: str):
        assert _should_skip_path(path) is True

    @pytest.mark.parametrize("path", [
        "src/app.py",
        "routes/users.js",
        "contracts/Token.sol",
        "lib/middleware.ts",
    ])
    def test_allow_source_dirs(self, path: str):
        assert _should_skip_path(path) is False

    def test_skip_lock_files(self):
        assert _should_skip_path("package-lock.json") is True
        assert _should_skip_path("yarn.lock") is True
        assert _should_skip_path("pnpm-lock.yaml") is True


# ── Python Attack Surface Detection ─────────────────────────────────────────


class TestPythonSurfaceDetection:
    @pytest.mark.asyncio
    async def test_detects_fastapi_route_decorators(self, tmp_path: Path):
        _write_file(tmp_path, "app/routes.py", '''\
from fastapi import APIRouter

router = APIRouter()

@router.get("/api/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

@router.post("/api/users")
async def create_user(name: str):
    return {"name": name}
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        names = {s.entry_point for s in surfaces}
        assert "get_user" in names
        assert "create_user" in names

        get_surface = next(s for s in surfaces if s.entry_point == "get_user")
        assert get_surface.surface_type == AttackSurfaceType.API_ENDPOINT
        assert get_surface.http_method == "GET"
        assert get_surface.route_pattern == "/api/users/{user_id}"
        assert get_surface.file_path == "app/routes.py"
        assert get_surface.line_number is not None

    @pytest.mark.asyncio
    async def test_detects_flask_route_decorator(self, tmp_path: Path):
        _write_file(tmp_path, "app.py", '''\
from flask import Flask
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    return "OK"

@app.delete("/api/item/{item_id}")
def delete_item(item_id):
    pass
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        names = {s.entry_point for s in surfaces}
        assert "health_check" in names
        assert "delete_item" in names

    @pytest.mark.asyncio
    async def test_detects_handler_name_patterns(self, tmp_path: Path):
        _write_file(tmp_path, "handlers.py", '''\
def middleware(request, next):
    return next(request)

def authenticate(request):
    token = request.headers.get("Authorization")
    return verify(token)

def handle_upload(file_data):
    save_to_disk(file_data)

def on_message(ws, message):
    process(message)

def resolve_user(info, id):
    return User.get(id)

def execute_query(sql, params):
    cursor.execute(sql, params)
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        types = {s.entry_point: s.surface_type for s in surfaces}
        assert types.get("middleware") == AttackSurfaceType.MIDDLEWARE
        assert types.get("authenticate") == AttackSurfaceType.AUTH_HANDLER
        assert types.get("handle_upload") == AttackSurfaceType.FILE_UPLOAD
        assert types.get("on_message") == AttackSurfaceType.EVENT_HANDLER
        assert types.get("resolve_user") == AttackSurfaceType.GRAPHQL_RESOLVER
        assert types.get("execute_query") == AttackSurfaceType.DATABASE_QUERY

    @pytest.mark.asyncio
    async def test_skips_private_functions(self, tmp_path: Path):
        _write_file(tmp_path, "internal.py", '''\
def _private_helper():
    pass

def __dunder_method__():
    pass

def public_handler():
    pass
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        names = {s.entry_point for s in surfaces}
        assert "_private_helper" not in names
        # Public handlers should still be found if they match patterns
        # __dunder_method__ starts with __ not single _ so is not skipped

    @pytest.mark.asyncio
    async def test_handles_syntax_error_gracefully(self, tmp_path: Path):
        _write_file(tmp_path, "broken.py", '''\
def foo(
    # Missing closing paren and colon — SyntaxError
''')
        ingestor = _make_ingestor(tmp_path)
        # Should not raise — returns empty list for this file
        await ingestor.map_attack_surfaces()
        # No crash is the test


# ── JavaScript/TypeScript Surface Detection ──────────────────────────────────


class TestJavaScriptSurfaceDetection:
    @pytest.mark.asyncio
    async def test_detects_express_routes(self, tmp_path: Path):
        _write_file(tmp_path, "routes.js", '''\
const express = require('express');
const router = express.Router();

router.get('/api/users', async (req, res) => {
    const users = await db.getUsers();
    res.json(users);
});

router.post('/api/users', async (req, res) => {
    const user = await db.createUser(req.body);
    res.json(user);
});

app.delete('/api/users/:id', (req, res) => {
    db.deleteUser(req.params.id);
});
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        assert len(surfaces) >= 3
        methods = {s.http_method for s in surfaces}
        assert "GET" in methods
        assert "POST" in methods
        assert "DELETE" in methods

        routes = {s.route_pattern for s in surfaces}
        assert "/api/users" in routes
        assert "/api/users/:id" in routes

    @pytest.mark.asyncio
    async def test_detects_exported_functions(self, tmp_path: Path):
        _write_file(tmp_path, "handlers.ts", '''\
export function processPayment(amount: number) {
    return stripe.charge(amount);
}

export async function validateToken(token: string) {
    return jwt.verify(token);
}
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        names = {s.entry_point for s in surfaces}
        assert "processPayment" in names
        assert "validateToken" in names

        for s in surfaces:
            assert s.surface_type == AttackSurfaceType.FUNCTION_EXPORT

    @pytest.mark.asyncio
    async def test_detects_default_export_handler(self, tmp_path: Path):
        _write_file(tmp_path, "page.tsx", '''\
export default function HomePage() {
    return <div>Home</div>;
}
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        names = {s.entry_point for s in surfaces}
        assert any("HomePage" in name for name in names)


# ── Solidity Surface Detection ──────────────────────────────────────────────


class TestSoliditySurfaceDetection:
    @pytest.mark.asyncio
    async def test_detects_public_functions(self, tmp_path: Path):
        _write_file(tmp_path, "contracts/Token.sol", '''\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Token {
    mapping(address => uint256) public balances;

    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount);
        payable(msg.sender).transfer(amount);
        balances[msg.sender] -= amount;
    }

    function _internalHelper() internal pure returns (bool) {
        return true;
    }
}
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        names = {s.entry_point for s in surfaces}
        assert "transfer" in names
        assert "withdraw" in names
        # Internal functions should not be detected by regex
        # (the regex matches "public" or "external" specifically)

        for s in surfaces:
            assert s.surface_type == AttackSurfaceType.SMART_CONTRACT_PUBLIC

    @pytest.mark.asyncio
    async def test_detects_public_state_variables(self, tmp_path: Path):
        _write_file(tmp_path, "contracts/Vault.sol", '''\
pragma solidity ^0.8.0;

contract Vault {
    uint256 public totalDeposits;
    address public owner;
    mapping(address => uint256) public balances;
}
''')
        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        names = {s.entry_point for s in surfaces}
        assert "totalDeposits" in names
        assert "owner" in names


# ── Codebase Graph Building ─────────────────────────────────────────────────


class TestCodebaseGraph:
    @pytest.mark.asyncio
    async def test_builds_import_graph(self, tmp_path: Path):
        _write_file(tmp_path, "app/__init__.py", "")
        _write_file(tmp_path, "app/main.py", '''\
from app.routes import setup_routes
from app.db import get_session
import logging
''')
        _write_file(tmp_path, "app/routes.py", '''\
from app.models import User
from flask import Blueprint
''')
        _write_file(tmp_path, "app/db.py", '''\
import sqlalchemy
''')

        ingestor = _make_ingestor(tmp_path)
        graph = await ingestor.build_codebase_graph()

        assert "app/main.py" in graph
        assert "app.routes" in graph["app/main.py"]
        assert "app.db" in graph["app/main.py"]
        assert "logging" in graph["app/main.py"]

        assert "app/routes.py" in graph
        assert "app.models" in graph["app/routes.py"]

    @pytest.mark.asyncio
    async def test_skips_oversized_files(self, tmp_path: Path):
        # Write a file larger than the 512KB limit
        _write_file(tmp_path, "huge.py", "x = 1\n" * 200_000)

        ingestor = _make_ingestor(tmp_path)
        graph = await ingestor.build_codebase_graph()

        assert "huge.py" not in graph


# ── Context Code Extraction ──────────────────────────────────────────────────


class TestContextExtraction:
    @pytest.mark.asyncio
    async def test_python_context_returns_function_body(self, tmp_path: Path):
        _write_file(tmp_path, "app.py", '''\
import os

def helper():
    return 42

def get_user(user_id: int):
    """Get a user by ID."""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(404)
    return user

def another():
    pass
''')

        ingestor = _make_ingestor(tmp_path)
        surface = AttackSurface(
            entry_point="get_user",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="app.py",
            line_number=6,
        )

        context = await ingestor.extract_context_code(surface)
        assert "def get_user" in context
        assert "db.query" in context

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_empty(self, tmp_path: Path):
        ingestor = _make_ingestor(tmp_path)
        surface = AttackSurface(
            entry_point="missing",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="does_not_exist.py",
        )
        context = await ingestor.extract_context_code(surface)
        assert context == ""

    @pytest.mark.asyncio
    async def test_js_context_returns_line_window(self, tmp_path: Path):
        lines = [f"// line {i}" for i in range(100)]
        lines[50] = "export function targetFunc() {"
        _write_file(tmp_path, "code.js", "\n".join(lines))

        ingestor = _make_ingestor(tmp_path)
        surface = AttackSurface(
            entry_point="targetFunc",
            surface_type=AttackSurfaceType.FUNCTION_EXPORT,
            file_path="code.js",
            line_number=51,
        )

        context = await ingestor.extract_context_code(surface)
        assert "targetFunc" in context


# ── Mixed-Language Codebase ──────────────────────────────────────────────────


class TestMixedLanguageCodebase:
    @pytest.mark.asyncio
    async def test_detects_surfaces_across_languages(self, tmp_path: Path):
        _write_file(tmp_path, "api/routes.py", '''\
from fastapi import APIRouter
router = APIRouter()

@router.get("/api/health")
def health():
    return {"status": "ok"}
''')
        _write_file(tmp_path, "frontend/api.ts", '''\
export function fetchUser(id: number) {
    return fetch(`/api/user/${id}`);
}
''')
        _write_file(tmp_path, "contracts/Token.sol", '''\
contract Token {
    function mint(address to, uint256 amount) public {
        _mint(to, amount);
    }
}
''')

        ingestor = _make_ingestor(tmp_path)
        surfaces = await ingestor.map_attack_surfaces()

        file_paths = {s.file_path for s in surfaces}
        assert any("routes.py" in p for p in file_paths)
        assert any("api.ts" in p for p in file_paths)
        assert any("Token.sol" in p for p in file_paths)


# ── Ingest from GitHub (mocked) ─────────────────────────────────────────────


class TestIngestFromGitHub:
    @pytest.mark.asyncio
    async def test_class_method_returns_ingestor(self, tmp_path: Path):
        """Verify the factory method wires up correctly."""
        from unittest.mock import AsyncMock, patch

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)):
            (tmp_path / "repo").mkdir(exist_ok=True)

            ingestor = await TargetIngestor.ingest_from_github(
                "https://github.com/test/repo",
            )

            assert isinstance(ingestor, TargetIngestor)
            assert ingestor.workspace.workspace_type == "external_repo"
