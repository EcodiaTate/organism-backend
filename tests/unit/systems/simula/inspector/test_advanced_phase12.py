"""
Unit tests for Phase 12: Advanced Inspector Features

Tests cover:
  1. Multi-Language Support (Go, Rust, TypeScript detection)
  2. Exploit Chain Detection (vulnerability sequences)
  3. Zero-Day Marketplace (cryptographic transactions)
  4. Autonomous Patching (GitHub PR generation)
  5. Continuous Hunting (scheduled recurring scans)
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from systems.simula.inspector.advanced import (
    AutonomousPatchingOrchestrator,
    ContinuousHuntingScheduler,
    ExploitChainAnalyzer,
    GitHubPRConfig,
    GoFunctionSignature,
    HuntScheduleRun,
    MultiLanguageSurfaceDetector,
    RustFunctionSignature,
    ScheduledHuntConfig,
    TypeScriptFunctionSignature,
    ZeroDayMarketplace,
)
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from systems.simula.inspector.workspace import TargetWorkspace

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.1: Multi-Language Support Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGoFunctionDetection:
    """Test Go handler detection."""

    def test_extract_simple_handler(self) -> None:
        """Test extracting a basic Go HTTP handler."""
        go_code = """
package main

func HandleUser(w http.ResponseWriter, r *http.Request) {
    id := r.URL.Query().Get("id")
    user := getUser(id)
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}
"""
        surfaces = GoFunctionSignature.extract_handlers(go_code, "handlers/user.go")

        assert len(surfaces) >= 1
        assert any(s.entry_point == "HandleUser" for s in surfaces)

    def test_extract_multiple_handlers(self) -> None:
        """Test extracting multiple handlers from same file."""
        go_code = """
func HandleUser(w http.ResponseWriter, r *http.Request) {}
func HandleAdmin(w http.ResponseWriter, r *http.Request) {}
func HandleAPI(w http.ResponseWriter, r *http.Request) {}
"""
        surfaces = GoFunctionSignature.extract_handlers(go_code, "handlers.go")
        handler_names = [s.entry_point for s in surfaces]

        assert "HandleUser" in handler_names
        assert "HandleAdmin" in handler_names
        assert "HandleAPI" in handler_names


class TestRustFunctionDetection:
    """Test Rust handler detection."""

    def test_extract_actix_handler(self) -> None:
        """Test extracting an actix-web handler."""
        rust_code = '''
#[get("/api/user/{id}")]
async fn get_user(
    id: web::Path<u32>,
    db: web::Data<DbPool>,
) -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(user))
}
'''
        surfaces = RustFunctionSignature.extract_handlers(rust_code, "src/handlers.rs")

        assert len(surfaces) >= 1
        assert any(s.entry_point == "get_user" for s in surfaces)

    def test_extract_axum_route(self) -> None:
        """Test extracting an axum route definition."""
        rust_code = '''
let app = Router::new()
    .route("/api/user", get(get_user))
    .route("/api/admin", post(admin_action))
    .with_state(state);
'''
        surfaces = RustFunctionSignature.extract_handlers(rust_code, "src/routes.rs")

        assert any(s.entry_point == "get_user" for s in surfaces)
        assert any(s.entry_point == "admin_action" for s in surfaces)


class TestTypeScriptFunctionDetection:
    """Test TypeScript handler detection."""

    def test_extract_nestjs_handler(self) -> None:
        """Test extracting NestJS @Controller handlers."""
        ts_code = """
@Controller('users')
export class UserController {
    @Get(':id')
    findOne(@Param('id') id: string) {
        return this.userService.findOne(+id);
    }

    @Post()
    create(@Body() createUserDto: CreateUserDto) {
        return this.userService.create(createUserDto);
    }
}
"""
        surfaces = TypeScriptFunctionSignature.extract_handlers(ts_code, "src/user.controller.ts")

        assert len(surfaces) >= 2
        assert any(s.entry_point == "findOne" for s in surfaces)
        assert any(s.entry_point == "create" for s in surfaces)


@pytest.mark.asyncio
async def test_multi_language_detector() -> None:
    """Test multi-language surface detection integration."""
    detector = MultiLanguageSurfaceDetector()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = TargetWorkspace(Path(tmpdir), "external_repo")

        # Create test files in different languages
        go_file = Path(tmpdir) / "main.go"
        go_file.write_text(
            """
func HandleRequest(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
}
"""
        )

        rust_file = Path(tmpdir) / "main.rs"
        rust_file.write_text(
            """
#[get("/api")]
async fn api_handler() -> Result<String> {
    Ok("ok".to_string())
}
"""
        )

        ts_file = Path(tmpdir) / "handler.ts"
        ts_file.write_text(
            """
@Get('/data')
getData() {
    return { data: [] };
}
"""
        )

        surfaces = await detector.detect_attack_surfaces(workspace)

        # Should find at least one handler from each language
        assert len(surfaces) >= 3


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.2: Exploit Chain Detection Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_exploit_chain_analyzer() -> None:
    """Test exploit chain detection."""
    analyzer = ExploitChainAnalyzer()

    # Create two related vulnerabilities
    vuln1 = VulnerabilityReport(
        target_url="https://github.com/example/repo",
        vulnerability_class=VulnerabilityClass.IDOR,
        severity=VulnerabilitySeverity.HIGH,
        attack_surface=AttackSurface(
            entry_point="get_user_profile",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="handlers/user.py",
        ),
        attack_goal="Access another user's profile",
        z3_counterexample="user_id != authenticated_user_id",
    )

    vuln2 = VulnerabilityReport(
        target_url="https://github.com/example/repo",
        vulnerability_class=VulnerabilityClass.XSS,
        severity=VulnerabilitySeverity.HIGH,
        attack_surface=AttackSurface(
            entry_point="get_user_profile",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="handlers/user.py",
        ),
        attack_goal="Inject malicious script in profile bio",
        z3_counterexample="bio_field not sanitized",
    )

    chains = await analyzer.analyze_chains([vuln1, vuln2])

    # Should discover at least one chain (same entry point + escalating severity)
    assert len(chains) >= 1
    assert len(chains[0].links) == 2

    # Chain severity should be at least HIGH
    assert chains[0].chain_severity in [
        VulnerabilitySeverity.HIGH,
        VulnerabilitySeverity.CRITICAL,
    ]


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.3: Zero-Day Marketplace Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_marketplace_keypair_generation() -> None:
    """Test RSA keypair generation for marketplace."""
    marketplace = ZeroDayMarketplace()

    private_key, public_key = marketplace.generate_keypair()

    assert private_key is not None
    assert public_key is not None

    # Public key should be serializable to PEM
    pem_str = marketplace.export_public_key(public_key)
    assert pem_str.startswith("-----BEGIN PUBLIC KEY-----")


def test_marketplace_key_export_import() -> None:
    """Test exporting and re-importing public keys."""
    marketplace = ZeroDayMarketplace()

    _, public_key = marketplace.generate_keypair()
    pem_str = marketplace.export_public_key(public_key)

    # Re-import should work
    imported_key = marketplace.import_public_key(pem_str)
    assert imported_key is not None

    # Exported form should match
    reimported_pem = marketplace.export_public_key(imported_key)
    assert reimported_pem == pem_str


@pytest.mark.asyncio
async def test_publish_vulnerability_to_marketplace() -> None:
    """Test publishing a vulnerability to the marketplace."""
    private_key, public_key = ZeroDayMarketplace().generate_keypair()
    marketplace = ZeroDayMarketplace(seller_private_key=private_key)

    vuln = VulnerabilityReport(
        target_url="https://github.com/example/repo",
        vulnerability_class=VulnerabilityClass.SQL_INJECTION,
        severity=VulnerabilitySeverity.CRITICAL,
        attack_surface=AttackSurface(
            entry_point="search_user",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="handlers/search.py",
        ),
        attack_goal="SQL injection via search parameter",
        z3_counterexample="query_param not escaped",
        proof_of_concept_code="import requests\nresponse = requests.get(...)",
    )

    listing = await marketplace.publish_vulnerability(vuln, asking_price_usd=5000.0, seller_public_key=public_key)

    assert listing.vulnerability_id == vuln.id
    assert listing.asking_price_usd == 5000.0
    assert listing.sold is False
    assert listing.proof_of_concept_encrypted != ""


@pytest.mark.asyncio
async def test_create_purchase_agreement() -> None:
    """Test creating a purchase agreement on the marketplace."""
    seller_key = ZeroDayMarketplace().generate_keypair()
    buyer_key = ZeroDayMarketplace().generate_keypair()

    marketplace = ZeroDayMarketplace(seller_private_key=seller_key[0])

    vuln = VulnerabilityReport(
        target_url="https://github.com/example/repo",
        vulnerability_class=VulnerabilityClass.PRIVILEGE_ESCALATION,
        severity=VulnerabilitySeverity.CRITICAL,
        attack_surface=AttackSurface(
            entry_point="admin_panel",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="admin/routes.py",
        ),
        attack_goal="Regular user can access admin functions",
        z3_counterexample="admin_check skipped",
        proof_of_concept_code="# PoC code here",
    )

    listing = await marketplace.publish_vulnerability(vuln, asking_price_usd=10000.0, seller_public_key=seller_key[1])

    # Create purchase agreement
    agreement = await marketplace.create_purchase_agreement(
        listing.id,
        buyer_public_key=buyer_key[1],
        purchase_price_usd=10000.0,
    )

    assert agreement.listing_id == listing.id
    assert agreement.purchase_price_usd == 10000.0
    assert agreement.proof_signature != ""


@pytest.mark.asyncio
async def test_finalize_marketplace_sale() -> None:
    """Test finalizing a marketplace sale."""
    seller_key = ZeroDayMarketplace().generate_keypair()
    buyer_key = ZeroDayMarketplace().generate_keypair()

    marketplace = ZeroDayMarketplace(seller_private_key=seller_key[0])

    vuln = VulnerabilityReport(
        target_url="https://github.com/example/repo",
        vulnerability_class=VulnerabilityClass.REENTRANCY,
        severity=VulnerabilitySeverity.CRITICAL,
        attack_surface=AttackSurface(
            entry_point="transfer",
            surface_type=AttackSurfaceType.SMART_CONTRACT_PUBLIC,
            file_path="contracts/token.sol",
        ),
        attack_goal="Reentrancy in token transfer",
        z3_counterexample="recursive call allowed",
        proof_of_concept_code="// PoC exploit code",
    )

    listing = await marketplace.publish_vulnerability(vuln, asking_price_usd=50000.0, seller_public_key=seller_key[1])

    agreement = await marketplace.create_purchase_agreement(
        listing.id,
        buyer_public_key=buyer_key[1],
        purchase_price_usd=50000.0,
    )

    # Finalize the sale
    success = await marketplace.finalize_sale(agreement.id)

    assert success
    assert listing.sold is True
    assert listing.sold_at is not None


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.4: Autonomous Patching Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_autonomous_patching_orchestrator() -> None:
    """Test autonomous PR generation."""
    orchestrator = AutonomousPatchingOrchestrator()

    vuln = VulnerabilityReport(
        target_url="https://github.com/example/repo",
        vulnerability_class=VulnerabilityClass.PATH_TRAVERSAL,
        severity=VulnerabilitySeverity.HIGH,
        attack_surface=AttackSurface(
            entry_point="serve_file",
            surface_type=AttackSurfaceType.API_ENDPOINT,
            file_path="routes/files.py",
            line_number=42,
        ),
        attack_goal="Read arbitrary files via path traversal",
        z3_counterexample="path parameter not validated",
        proof_of_concept_code="import requests; requests.get('/files?path=../../../etc/passwd')",
    )

    config = GitHubPRConfig(
        github_token="test_token",
        target_owner="example",
        target_repo="repo",
    )

    result = await orchestrator.submit_patch_pr(
        target_repo_url="https://github.com/example/repo",
        vulnerability=vuln,
        patched_code="# patched code here",
        patch_diff="--- a/routes/files.py\n+++ b/routes/files.py\n@@ -42,3 +42,4 @@\n",
        config=config,
    )

    assert result.vulnerability_id == vuln.id
    assert result.success is True
    assert result.pr_url is not None


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.5: Continuous Hunting Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_continuous_hunting_scheduler_registration() -> None:
    """Test registering a hunt schedule."""
    scheduler = ContinuousHuntingScheduler()

    config = ScheduledHuntConfig(
        target_repo_url="https://github.com/example/repo",
        cron_expression="0 2 * * *",
        enabled=True,
    )

    registered = scheduler.register_hunt_schedule(config)

    assert registered.id is not None
    assert registered.target_repo_url == "https://github.com/example/repo"
    assert config.id in scheduler.schedules


def test_scheduled_hunt_statistics() -> None:
    """Test aggregating statistics from scheduled hunts."""
    scheduler = ContinuousHuntingScheduler()

    config = ScheduledHuntConfig(
        target_repo_url="https://github.com/example/repo",
        cron_expression="0 * * * *",  # Hourly
    )

    registered = scheduler.register_hunt_schedule(config)

    # Add some fake run history
    run1 = HuntScheduleRun(
        schedule_config_id=registered.id,
        target_repo_url=config.target_repo_url,
        vulns_discovered=5,
        duration_ms=12000,
    )
    run1.run_completed_at = datetime.now()

    run2 = HuntScheduleRun(
        schedule_config_id=registered.id,
        target_repo_url=config.target_repo_url,
        vulns_discovered=3,
        duration_ms=10000,
    )
    run2.run_completed_at = datetime.now()

    scheduler.run_history[registered.id] = [run1, run2]

    stats = scheduler.get_schedule_statistics(registered.id)

    assert stats["total_runs"] == 2
    assert stats["total_vulnerabilities_discovered"] == 8
    assert stats["avg_vulns_per_run"] == 4.0


# ═════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_full_phase12_workflow() -> None:
    """Integration test: Multi-language → Chain detection → Marketplace → Patch → Schedule."""
    # This tests the integration of all Phase 12 features working together

    MultiLanguageSurfaceDetector()
    chain_analyzer = ExploitChainAnalyzer()

    # Create mock vulnerabilities from different languages
    vulns = [
        VulnerabilityReport(
            target_url="https://github.com/example/repo",
            vulnerability_class=VulnerabilityClass.IDOR,
            severity=VulnerabilitySeverity.HIGH,
            attack_surface=AttackSurface(
                entry_point="get_user",
                surface_type=AttackSurfaceType.API_ENDPOINT,
                file_path="handlers.go",
            ),
            attack_goal="Access another user's data",
            z3_counterexample="auth_check bypassed",
        ),
        VulnerabilityReport(
            target_url="https://github.com/example/repo",
            vulnerability_class=VulnerabilityClass.XSS,
            severity=VulnerabilitySeverity.MEDIUM,
            attack_surface=AttackSurface(
                entry_point="get_user",
                surface_type=AttackSurfaceType.API_ENDPOINT,
                file_path="handlers.go",
            ),
            attack_goal="Inject malicious script",
            z3_counterexample="input_not_sanitized",
        ),
    ]

    # Detect chains
    chains = await chain_analyzer.analyze_chains(vulns)
    assert len(chains) >= 1

    # Publish to marketplace
    seller_key = ZeroDayMarketplace().generate_keypair()
    marketplace = ZeroDayMarketplace(seller_private_key=seller_key[0])

    for vuln in vulns:
        listing = await marketplace.publish_vulnerability(
            vuln,
            asking_price_usd=5000.0,
            seller_public_key=seller_key[1],
        )
        assert listing.sold is False

    # Setup autonomous patching
    AutonomousPatchingOrchestrator()
    GitHubPRConfig(
        github_token="token",
        target_owner="example",
        target_repo="repo",
    )

    # Setup continuous hunting
    scheduler = ContinuousHuntingScheduler()
    schedule = ScheduledHuntConfig(
        target_repo_url="https://github.com/example/repo",
        cron_expression="0 2 * * *",
    )
    registered = scheduler.register_hunt_schedule(schedule)
    assert registered.id in scheduler.schedules
