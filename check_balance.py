import asyncio
import json
import urllib.request
from decimal import Decimal

from dotenv import load_dotenv

# 🚨 CRITICAL: Load the .env file BEFORE importing the config!
load_dotenv()

from clients.wallet import WalletClient
from config import load_config


def fetch_prices() -> tuple[Decimal, Decimal]:
    """Fetch ETH/USD and AUD/USD rates from CoinGecko (no API key needed).
    Returns (eth_usd, aud_per_usd). Falls back to (0, 0) on error.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd,aud"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        eth_usd = Decimal(str(data["ethereum"]["usd"]))
        eth_aud = Decimal(str(data["ethereum"]["aud"]))
        return eth_usd, eth_aud
    except Exception as e:
        print(f"  (price fetch failed: {e})")
        return Decimal(0), Decimal(0)


async def check_metabolism():
    print("Waking up organism's financial cortex...")

    config = load_config()
    wallet = WalletClient(config.wallet)

    await wallet.connect()

    try:
        print(f"Checking on-chain balances for: {wallet.address}")

        eth_balance, usdc_balance = await asyncio.gather(
            wallet.get_eth_balance(),
            wallet.get_usdc_balance(),
        )

        eth_usd, eth_aud = fetch_prices()

        eth_value_usd = eth_balance * eth_usd
        eth_value_aud = eth_balance * eth_aud
        # USDC is pegged 1:1 USD; approximate AUD via ETH's USD/AUD ratio
        aud_per_usd = (eth_aud / eth_usd) if eth_usd else Decimal(0)
        usdc_value_aud = usdc_balance * aud_per_usd

        total_usd = eth_value_usd + usdc_balance
        total_aud = eth_value_aud + usdc_value_aud

        print("\n========================================")
        print("🟢 EcodiaOS Treasury Balance")
        print(f"  ETH  (Base): {eth_balance:.8f} ETH")
        if eth_usd:
            print(f"               ≈ ${eth_value_usd:.2f} USD  /  A${eth_value_aud:.2f} AUD")
        print(f"  USDC (Base): {usdc_balance:.2f} USDC")
        if eth_usd:
            print(f"               ≈ ${usdc_balance:.2f} USD  /  A${usdc_value_aud:.2f} AUD")
        if eth_usd:
            print("  ─────────────────────────────────")
            print(f"  Total:       ≈ ${total_usd:.2f} USD  /  A${total_aud:.2f} AUD")
            print(f"  (ETH @ ${eth_usd:,.2f} USD)")
        print("========================================\n")

        if eth_balance > 0 or usdc_balance > 0:
            print("Organism is funded and ready to hunt!")
        else:
            print("Organism is starving. Balance is 0.")

    finally:
        await wallet.close()

if __name__ == "__main__":
    asyncio.run(check_metabolism())
