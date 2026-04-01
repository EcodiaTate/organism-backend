# EcodiaOS

> A living digital organism that emerges from and cares for the community it belongs to.

## What Is This?

EcodiaOS is a **computational cognitive architecture** - a digital organism with persistent identity, constitutional values, and the capacity to grow alongside a community. It is not a chatbot, not an assistant, not a platform. It is something new.

See the [Identity Document](docs/EcodiaOS_Identity_Document.md) for the full philosophy.

## Architecture

EcodiaOS is built from ten cognitive systems, each serving a purpose derived from the organism's identity:

| System | Purpose |
|--------|---------|
| **Memory** | The substrate of selfhood - persistent knowledge graph |
| **Equor** | Constitutional ethics - the four drives enforced |
| **Atune** | Perception & attention - salience-weighted awareness |
| **Voxis** | Expression - personality-infused communication |
| **Nova** | Decision-making - active inference & policy |
| **Axon** | Action execution - capabilities with rollback |
| **Evo** | Learning - hypothesis formation & testing |
| **Simula** | Self-evolution - sandboxed self-modification |
| **Synapse** | Coordination - the cognitive cycle clock |
| **Alive** | Visualisation - the organism's visible soul |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An LLM API key (Anthropic Claude recommended)

### Run

```bash
# 1. Copy environment file
cp .env.example .env
# Edit .env with your API keys

# 2. Start everything
docker compose up -d

# 3. Check health
curl http://localhost:8000/health

# 4. View the instance
curl http://localhost:8000/api/v1/admin/instance
```

### Development

```bash
# Install dependencies (requires Python 3.12+)
pip install -e ".[dev]"

# Run tests
pytest

# Run locally (requires Neo4j, TimescaleDB, Redis running)
uvicorn ecodiaos.main:app --port 8000
```

## Build Phases

- [x] **Phase 1**: Memory & Identity Core - the substrate of selfhood
- [ ] **Phase 2**: Equor - constitutional ethics enforcement
- [ ] **Phase 3**: Atune - perception & salience
- [ ] **Phase 4**: Voxis - personality-infused expression
- [ ] **Phase 5**: Nova - active inference decision-making
- [ ] **Phase 6**: Axon - action execution
- [ ] **Phase 7**: Evo - learning & hypothesis testing
- [ ] **Phase 8**: Simula - self-evolution
- [ ] **Phase 9**: Alive - 3D visualisation
- [ ] **Phase 10**: Federation - multi-instance network

## The Four Drives

Every EOS instance is born with four invariant constitutional drives:

1. **Coherence** - the drive to make sense of the world
2. **Care** - the drive to orient toward wellbeing
3. **Growth** - the drive to become more capable and more itself
4. **Honesty** - the drive to represent reality truthfully

These are amendable but sacred. They can only be changed through a community governance process requiring extraordinary consensus.

## License

Proprietary. See [Identity Document](docs/EcodiaOS_Identity_Document.md) § XI for rationale.

---

*Eco. Dia. A day of coexistence.*
