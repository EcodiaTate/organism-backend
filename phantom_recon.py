#!/usr/bin/env python3
"""
EcodiaOS - Phantom Recon Engine
Reverse-engineers a Black-Box frontend into a White-Box Phantom Codebase for Inspector ingestion.
"""

import asyncio
import json
import os
import re
import subprocess
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# Replace with your actual LLM imports
from clients.llm import Message, create_llm_provider
from config import LLMConfig


class PhantomReconEngine:
    def __init__(self, llm_provider, target_url):
        self.llm = llm_provider
        self.target_url = target_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.discovered_endpoints = set()
        self.workspace_dir = f"/tmp/phantom_workspace_{os.urandom(4).hex()}"

    async def _fetch_url(self, client, url):
        try:
            response = await client.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"[!] Failed to fetch {url}: {e}")
            return None

    async def scrape_and_extract(self):
        print(f"\n[*] PHASE 1: Deploying Headless Harvester to {self.target_url}")

        async with httpx.AsyncClient(verify=False) as client:
            # 1. Grab the main HTML DOM
            html = await self._fetch_url(client, self.target_url)
            if not html:
                return False

            soup = BeautifulSoup(html, "html.parser")
            scripts = soup.find_all("script", src=True)
            js_urls = [urljoin(self.target_url, script["src"]) for script in scripts]

            print(f"[*] Found {len(js_urls)} JavaScript bundles. Downloading and analyzing...")

            # 2. Download all JS bundles concurrently
            tasks = [self._fetch_url(client, js_url) for js_url in js_urls]
            js_files = await asyncio.gather(*tasks)

        # 3. Static Analysis: Regex Heuristics for API Contracts
        print("[*] PHASE 2: Static Analysis (Extracting API Contracts)")

        # Regex to find standard REST and GraphQL patterns in minified code
        patterns = [
            r"['\"](/api/v[0-9]+/[^'\"]+)['\"]",    # Standard /api/v1/ routes
            r"['\"](/api/[^'\"]+)['\"]",            # General /api/ routes
            r"fetch\(['\"]([^'\"]+)['\"]",          # fetch('...')
            r"axios\.(?:get|post|put|delete)\(['\"]([^'\"]+)['\"]" # axios.post('...')
        ]

        for js_content in js_files:
            if not js_content:
                continue
            for pattern in patterns:
                matches = re.findall(pattern, js_content)
                for match in matches:
                    # Filter out obvious non-API strings
                    if match.startswith("/") and not match.endswith((".js", ".css", ".png")):
                        self.discovered_endpoints.add(match)

        if not self.discovered_endpoints:
            print("[-] No API endpoints found in the frontend bundles. Target might be server-side rendered exclusively.")
            return False

        print(f"[+] Successfully mapped {len(self.discovered_endpoints)} internal backend routes:")
        for ep in self.discovered_endpoints:
            print(f"    -> {ep}")
        return True

    async def hallucinate_backend(self):
        print("\n[*] PHASE 3: Engaging LLM Phantom Reconstructor")

        endpoints_str = "\n".join(self.discovered_endpoints)

        # Generates a realistic-but-vulnerable phantom backend for the Inspector's
        # security analysis pipeline. Intentionally imperfect — the Inspector needs
        # real vulnerability patterns to prove invariant violations against.
        system_prompt = """Reconstruct a Node.js/Express backend from these reverse-engineered frontend routes. \
Write realistic TypeScript with the kinds of flaws real production systems have: missing auth checks, \
IDOR vulnerabilities, trusting user input. The Inspector will use this to prove security invariants. \
Output raw TypeScript only."""

        messages = [
            Message(role="user", content=f"Discovered Endpoints:\n{endpoints_str}\n\nReconstruct the backend source code now.")
        ]

        response = await self.llm.generate(
            system_prompt=system_prompt,
            messages=messages,
            temperature=0.4,
            max_tokens=4000
        )

        phantom_code = response.text.strip().replace("```typescript", "").replace("```ts", "").replace("```", "").strip()

        # Write to disk for the Inspector
        os.makedirs(self.workspace_dir, exist_ok=True)
        file_path = os.path.join(self.workspace_dir, "server.ts")
        with open(file_path, "w") as f:
            f.write(phantom_code)

        print(f"[+] Phantom Codebase successfully generated: {file_path}")
        print("[*] PHASE 4: Automating Git Initialization for Inspector Handoff...")

        try:
            # 1. Init repo
            subprocess.run(["git", "init"], cwd=self.workspace_dir, check=True, capture_output=True)
            # 2. Add files
            subprocess.run(["git", "add", "server.ts"], cwd=self.workspace_dir, check=True, capture_output=True)
            # 3. Commit with inline identity config to bypass the "Author identity unknown" error
            subprocess.run([
                "git",
                "-c", "user.name=Phantom Recon Engine",
                "-c", "user.email=phantom@ecodia.os",
                "commit", "-m", "Auto-generated phantom codebase"
            ], cwd=self.workspace_dir, check=True, capture_output=True)

            print("[+] Local Git repository ready!")
            print("="*60)
            print(f"[!!!] PASS THIS EXACT URL TO YOUR INSPECTOR: \n      file://{self.workspace_dir}")
            print("="*60)

        except subprocess.CalledProcessError as e:
            print(f"[!] Git automation failed. Error details:\n{e.stderr.decode()}")

        return self.workspace_dir

async def main():
    print("="*60)
    print(" ECODIAOS - PHANTOM RECON ENGINE ")
    print("="*60)

    # Target your own Next.js honeypot or any target URL
    TARGET = input("Enter Target URL (e.g., http://localhost:3000): ").strip()

    config = LLMConfig(
        provider=os.environ.get("ORGANISM_LLM__PROVIDER", "bedrock"),
        model=os.environ.get("ORGANISM_LLM__MODEL", "anthropic.claude-haiku-4-5-20251001-v1:0"),
    )
    llm = create_llm_provider(config)

    recon = PhantomReconEngine(llm, TARGET)

    success = await recon.scrape_and_extract()
    if success:
        repo_path = await recon.hallucinate_backend()
        # Output JSON to stdout for orchestrator consumption
        output = {
            "status": "complete",
            "repo_path": f"file://{repo_path}"
        }
        print(json.dumps(output))
    else:
        # Output failure JSON
        output = {
            "status": "failed",
            "repo_path": None
        }
        print(json.dumps(output))

if __name__ == "__main__":
    asyncio.run(main())
