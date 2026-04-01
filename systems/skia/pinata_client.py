"""
EcodiaOS - Skia Pinata Client

Thin httpx wrapper for the Pinata IPFS REST API.
No SDK dependency - uses httpx (already in deps).

Docs: https://docs.pinata.cloud/api-reference
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger("systems.skia.pinata")


class PinataError(Exception):
    """Raised when a Pinata API call fails."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"Pinata API error {status}: {detail}")


class PinataClient:
    """
    IPFS pinning via Pinata REST API.

    Lifecycle:
        client = PinataClient(api_url, gateway_url, jwt)
        await client.connect()
        cid, pin_id = await client.pin_bytes(data, "snapshot-001")
        blob = await client.get_by_cid(cid)
        await client.unpin(cid)
        await client.close()
    """

    def __init__(self, api_url: str, gateway_url: str, jwt: str) -> None:
        self._api_url = api_url.rstrip("/")
        self._gateway_url = gateway_url.rstrip("/")
        self._jwt = jwt
        self._client: httpx.AsyncClient | None = None
        self._gateway_client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._api_url,
            headers={"Authorization": f"Bearer {self._jwt}"},
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
        )
        self._gateway_client = httpx.AsyncClient(
            base_url=self._gateway_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._gateway_client:
            await self._gateway_client.aclose()
            self._gateway_client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("PinataClient not connected. Call connect() first.")
        return self._client

    def _require_gateway(self) -> httpx.AsyncClient:
        if self._gateway_client is None:
            raise RuntimeError("PinataClient not connected. Call connect() first.")
        return self._gateway_client

    async def pin_bytes(
        self,
        data: bytes,
        name: str,
        group_name: str = "",
    ) -> tuple[str, str]:
        """
        Pin raw bytes to IPFS.

        Returns (ipfs_cid, pin_id).
        """
        client = self._require_client()
        import orjson

        metadata = orjson.dumps({"name": name}).decode()
        files = {"file": (name, data, "application/octet-stream")}
        form_data = {"pinataMetadata": metadata}

        if group_name:
            options = orjson.dumps({"groupId": group_name}).decode()
            form_data["pinataOptions"] = options

        resp = await client.post(
            "/pinning/pinFileToIPFS",
            files=files,
            data=form_data,
        )
        if resp.status_code != 200:
            raise PinataError(resp.status_code, resp.text)

        body = resp.json()
        cid: str = body["IpfsHash"]
        pin_id: str = str(body.get("id", ""))
        logger.info("pinata_pinned", cid=cid, name=name, size=len(data))
        return cid, pin_id

    async def unpin(self, cid: str) -> None:
        """Unpin a CID from Pinata."""
        client = self._require_client()
        resp = await client.delete(f"/pinning/unpin/{cid}")
        if resp.status_code not in (200, 404):
            raise PinataError(resp.status_code, resp.text)
        logger.info("pinata_unpinned", cid=cid)

    async def list_pins(
        self,
        name_contains: str = "",
        limit: int = 10,
        sort_order: str = "DESC",
    ) -> list[dict[str, Any]]:
        """List pinned items. Returns newest first by default."""
        client = self._require_client()
        params: dict[str, str | int] = {
            "pageLimit": limit,
            "sortOrder": sort_order,
            "status": "pinned",
        }
        if name_contains:
            params["metadata[name]"] = name_contains

        resp = await client.get("/data/pinList", params=params)
        if resp.status_code != 200:
            raise PinataError(resp.status_code, resp.text)

        data = resp.json()
        rows = data.get("rows", [])
        return list(rows) if isinstance(rows, list) else []

    async def get_by_cid(self, cid: str) -> bytes:
        """Download pinned content from IPFS gateway."""
        gateway = self._require_gateway()
        resp = await gateway.get(f"/ipfs/{cid}")
        if resp.status_code != 200:
            raise PinataError(resp.status_code, f"Gateway fetch failed for {cid}")
        return resp.content

    async def health_check(self) -> bool:
        """Verify Pinata authentication is valid."""
        client = self._require_client()
        try:
            resp = await client.get("/data/testAuthentication")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
