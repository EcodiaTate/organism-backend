"""
EcodiaOS — Tollbooth Ledger

Redis-backed credit balance store.  Each customer API key maps to a
credit balance in a Redis hash.  All mutations are atomic (HINCRBY)
so concurrent requests against the same key are safe.

Redis layout:
  Hash key:  {prefix}:tollbooth:balances
  Field:     <api_key>
  Value:     integer credit balance

Separate hash for metadata:
  Hash key:  {prefix}:tollbooth:keys
  Field:     <api_key>
  Value:     JSON {"created_at": ..., "label": ...}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = structlog.get_logger("api.monetization.ledger")

_BALANCE_KEY = "tollbooth:balances"
_KEYS_KEY = "tollbooth:keys"

# Lua script for atomic check-and-decrement.
# Returns new balance on success, or -1 if insufficient.
# This uses Redis server-side Lua execution (EVALSHA/EVAL), NOT Python eval.
_DEBIT_LUA = """
local bal = tonumber(redis.call('HGET', KEYS[1], ARGV[1]) or 0)
if bal < tonumber(ARGV[2]) then
    return -1
end
return redis.call('HINCRBY', KEYS[1], ARGV[1], -tonumber(ARGV[2]))
"""


class CreditLedger:
    """Atomic credit balance operations backed by a single Redis hash."""

    def __init__(self, redis: Redis, prefix: str = "eos") -> None:
        self._redis = redis
        self._prefix = prefix

    def _key(self, suffix: str) -> str:
        return f"{self._prefix}:{suffix}"

    # ── Reads ─────────────────────────────────────────────────────

    async def balance(self, api_key: str) -> int:
        """Return the current credit balance for *api_key* (0 if absent)."""
        raw = await self._redis.hget(self._key(_BALANCE_KEY), api_key)  # type: ignore[misc]
        if raw is None:
            return 0
        return int(raw)

    async def key_exists(self, api_key: str) -> bool:
        """Return True if *api_key* is a registered Tollbooth key."""
        return bool(
            await self._redis.hexists(self._key(_KEYS_KEY), api_key)  # type: ignore[misc]
        )

    # ── Mutations ─────────────────────────────────────────────────

    async def credit(self, api_key: str, amount: int) -> int:
        """
        Add *amount* credits to *api_key*.  Returns the new balance.

        Uses HINCRBY for atomic increment — safe under concurrency.
        """
        new_balance: int = await self._redis.hincrby(  # type: ignore[misc]
            self._key(_BALANCE_KEY), api_key, amount
        )
        logger.info(
            "tollbooth_credit",
            api_key=api_key[:8],
            amount=amount,
            new_balance=new_balance,
        )
        return new_balance

    async def debit(self, api_key: str, amount: int) -> int | None:
        """
        Atomically debit *amount* credits from *api_key*.

        Returns the new balance on success, or ``None`` if the current
        balance is insufficient (no mutation occurs).

        Uses a Redis server-side Lua script for atomic check-and-decrement.
        """
        if amount <= 0:
            raise ValueError(f"debit amount must be positive, got {amount}")
        # Redis EVAL executes Lua on the server — this is the standard
        # pattern for atomic multi-step operations in Redis.
        result: int = await self._redis.eval(  # type: ignore[misc]
            _DEBIT_LUA,
            1,
            self._key(_BALANCE_KEY),
            api_key,
            str(amount),
        )
        if result == -1:
            logger.warning(
                "tollbooth_insufficient_credits",
                api_key=api_key[:8],
                required=amount,
            )
            return None

        logger.info(
            "tollbooth_debit",
            api_key=api_key[:8],
            amount=amount,
            new_balance=result,
        )
        return int(result)

    async def refund(self, api_key: str, amount: int, reason: str = "") -> int:
        """
        Refund *amount* credits to *api_key* after a failed backend call.

        Semantically identical to credit() but logged separately so that
        refunds are distinguishable from payment credits in the audit trail.
        Returns the new balance.
        """
        new_balance: int = await self._redis.hincrby(  # type: ignore[misc]
            self._key(_BALANCE_KEY), api_key, amount
        )
        logger.warning(
            "tollbooth_refund",
            api_key=api_key[:8],
            amount=amount,
            reason=reason,
            new_balance=new_balance,
        )
        return new_balance

    # ── Key management ────────────────────────────────────────────

    async def register_key(self, api_key: str, label: str = "") -> None:
        """Register a new API key in the keys hash."""
        import orjson

        from primitives.common import utc_now

        meta = orjson.dumps(
            {"created_at": utc_now().isoformat(), "label": label}
        ).decode()
        await self._redis.hset(self._key(_KEYS_KEY), api_key, meta)  # type: ignore[misc]
        logger.info("tollbooth_key_registered", api_key=api_key[:8], label=label)

    async def rotate_key(self, old_key: str, new_key: str) -> int:
        """
        Atomically rotate *old_key* to *new_key*.

        Transfers the balance and metadata from the old key to the new key,
        then deletes the old key from both hashes.  Returns the balance
        carried over to the new key.

        Uses a pipeline (multi-exec) so all mutations are atomic — no
        concurrent request can see a partially rotated state.
        """
        import orjson

        from primitives.common import utc_now

        balance_hash = self._key(_BALANCE_KEY)
        keys_hash = self._key(_KEYS_KEY)

        # Read current balance and metadata outside the pipeline (reads
        # cannot be inside MULTI/EXEC on standard Redis).
        raw_balance = await self._redis.hget(balance_hash, old_key)  # type: ignore[misc]
        current_balance = int(raw_balance) if raw_balance is not None else 0

        raw_meta = await self._redis.hget(keys_hash, old_key)  # type: ignore[misc]
        old_meta: dict = orjson.loads(raw_meta) if raw_meta else {}

        new_meta = orjson.dumps(
            {
                "created_at": utc_now().isoformat(),
                "label": old_meta.get("label", ""),
                "rotated_from": old_key[:8] + "****",
            }
        ).decode()

        # Atomic pipeline: write new key data, remove old key.
        pipe = self._redis.pipeline(transaction=True)
        pipe.hset(balance_hash, new_key, current_balance)  # type: ignore[misc]
        pipe.hset(keys_hash, new_key, new_meta)  # type: ignore[misc]
        pipe.hdel(balance_hash, old_key)  # type: ignore[misc]
        pipe.hdel(keys_hash, old_key)  # type: ignore[misc]
        await pipe.execute()

        logger.info(
            "tollbooth_key_rotated",
            old_key=old_key[:8],
            new_key=new_key[:8],
            balance_transferred=current_balance,
        )
        return current_balance
