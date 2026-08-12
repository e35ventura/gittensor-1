# Gittensor Compute Sub-Subnet

## Purpose

One shared pool of verified RTX 5090s serves every approved Gittensor model and runtime. The pool is not split into model lanes. A global Gepetto decides which release each GPU runs, and a regional router sends each request to the compatible GPU expected to finish first.

## System contract

| Component | Required behavior |
|---|---|
| Fleet target | Keep a baseline supply online and scale the desired GPU count from real utilization. There is no product-level maximum. |
| Funding | Pay a target price per GPU-hour from a fixed hourly pool. More supply dilutes payment; less supply concentrates it. |
| Global Gepetto | Assign approved model-runtime releases across the entire verified fleet and execute changes through miner agents. |
| Verification | Prove hardware, uptime, exact release, model weights, runtime integrity, and inference-stream origin. |
| Router | Reserve the compatible READY GPU with the lowest expected completion time, or immediately return 429. |
| Validator | Convert finalized verified GPU-seconds into the compute share of Bittensor weights. |

## 1. Dynamic GPU target and price

Configuration defines:

- `F`: minimum desired GPU floor.
- `T_req`: utilization-driven desired GPU target.
- `P`: target reward per verified GPU-hour.
- `B_max`: compute emission budget per hour.
- `K`: certified concurrency per GPU.

The funded target is:

```text
T = min(T_req, floor(B_max / P))
```

All verified GPUs may join. `T` controls the funded supply, not admission. The window budget is divided by verified READY-seconds:

| Example: `T=4`, pool=`2.60/hour` | Payment per GPU-hour |
|---:|---:|
| 2 READY GPUs | 1.30 |
| 4 READY GPUs | 0.65 |
| 8 READY GPUs | 0.325 |

Demand and utilization are:

```text
D = active reservations + EWMA(429 rate × expected service time)
U = D / (T × K)
```

- Sustained `U >= utilization_up`: raise `T_req` enough to bring utilization back to the threshold.
- Sustained `U <= utilization_down`: reduce `T_req` by one after the scale-down cooldown, never below `F`.
- No free slot: return 429 immediately. Rejections feed `D`; requests are never hidden in an internal queue.
- If `B_max` cannot fund `T_req`, expose the exact `funding_shortfall`.

## 2. One global Gepetto

Gepetto reads the approved release catalog, live demand, verified GPUs, minimum replicas, and placement weights. It produces and executes one global map:

```text
gpu_id -> approved release_digest
```

Miners do not choose their UID, release, concurrency, canary binding, or placement. They register hardware with a signed SN74 hotkey request. The control plane resolves the UID from the live metagraph, checks that the SparkCompute node was enrolled to the same hotkey, and assigns the release.

Every change is bound to a monotonic assignment epoch:

```text
REGISTERED -> DRAINING -> LOADING -> RUNTIME_VERIFY -> READY
```

The miner agent receives the full immutable release manifest and acknowledges each epoch. It cannot skip states or acknowledge an old assignment. Routing and earnings begin only after the final verification pass.

## 3. Layered verification

The design follows the overlapping-check pattern used by [Chutes](https://chutes.ai/docs/core-concepts/security-architecture), with SparkCompute as the hardware layer.

| Layer | Check |
|---|---|
| Hardware | SparkCompute verifies RTX 5090 identity, PCI data, driver status, timed GPU work, liveness, uptime, and unique GPU UUID. |
| Verifier provenance | Strict mode requires the configured SparkCompute protocol, source commit, and measured verifier build in every trusted snapshot. |
| Immutable release | `release_digest` is the SHA256 of the model, exact 40-character model/tokenizer revisions, runtime commit, container image and digest, filesystem digest, and weight-file manifest. A digest cannot be redefined. |
| Signed runtime | Release admission runs Cosign against the exact `image@sha256:digest` using the configured Gittensor public key. The SparkCompute snapshot must then attest that exact release, runtime, container, filesystem, model repository, and revision. |
| Model weights | The validator selects an unpredictable weight file and byte range, fetches reference bytes from the pinned Hugging Face revision, and compares `SHA256(nonce || bytes)` with the miner response. The stalest GPUs are selected first with randomized ties. Each batch starts at three concurrent GPUs and grows with fleet size so every GPU is refreshed twice inside the lease window. |
| Inference stream | The gateway verifies an HMAC commitment on streamed chunks, bound to request, release, model, revision, index, and content hash. The session key must be bound to the attested runtime. |
| Optional TEE | TDX and NVIDIA attestation evidence can bind runtime measurements and the stream-proof key to protected hardware. |

The controls are complementary. SparkCompute proves the GPU is genuine and alive. Random weight checks prove the pinned artifacts remain loaded. Runtime measurements prove the approved software is executing. Stream commitments prove returned chunks passed through that verified runtime session.

The current pinned SparkCompute commit is `aa61fbc20d24c217ea9f33e20be67d75a388e45d`. Its present `/api/status` schema supplies hardware, liveness, uptime, and model-canary evidence. Strict runtime mode also requires the verifier to add `verifier`, `report.runtime`, and `attestation` fields defined in this adapter. Until that deployment reports the configured measurement, the GPU fails closed and cannot become READY.

```json
{
  "verifier": {
    "protocol": "sparkcompute-v1",
    "source_commit": "aa61fbc20d24c217ea9f33e20be67d75a388e45d",
    "measurement": "sha256:<deployed-verifier-measurement>"
  },
  "report": {
    "runtime": {
      "release_digest": "sha256:<canonical-release>",
      "model_repository": "owner/model",
      "model_revision": "<40-character-commit>",
      "runtime_digest": "sha256:<runtime>",
      "runtime_commit": "<40-character-commit>",
      "container_image": "registry/image",
      "container_digest": "sha256:<container>",
      "filesystem_digest": "sha256:<filesystem>"
    }
  },
  "attestation": {
    "verified": true,
    "evidence_digest": "sha256:<signed-evidence>"
  }
}
```

## 4. Fastest-finish router

The router considers only GPUs with:

- a live verification lease;
- `READY` state;
- the exact release requested;
- a free certified slot;
- fresh queue telemetry.

It minimizes:

```text
expected completion = gateway-measured RTT + remaining work + expected service time
```

Expected completion is always the primary rule. Within a small equivalent-time band, fewer active requests wins. This spreads requests across equivalent nearby GPUs before stacking them, while still choosing a busy local GPU when it will finish sooner than an idle remote GPU.

The router atomically creates a durable reservation before returning the miner endpoint. Authenticated gateway observations update RTT, active work, remaining work, and service-time EWMAs. Miner telemetry is retained for health reconciliation but cannot override gateway-owned routing measurements.

## Persistence and automation

SQLite WAL storage persists releases, registrations, ownership, leases, lifecycle state, assignment epochs, placement, reservations, autoscaler state, READY-second accounting, demand counters, replay nonces, and finalized settlements.

Starting `gitt-compute` also starts the control loop:

1. Refresh SparkCompute leases.
2. Run random weight challenges.
3. Recalculate the fleet target and global placement.
4. Dispatch Gepetto transitions.
5. Finalize settlements on schedule.

Restarting the process restores unfinished assignments, live reservations, and reward accounting. Settlement windows are stored under a unique ID so the validator reads finalized results instead of an HTTP response.

## Validator emissions

Set `GITTENSOR_COMPUTE_DB` on the validator to the same settlement database. Each scoring round maps settled hotkey rewards onto current metagraph UIDs and normalizes them into `COMPUTE_EMISSION_SHARE`.

- With no configured database, existing Gittensor economics remain unchanged.
- With a configured database, 10% is carved from the existing OSS pool for compute.
- If no fresh valid compute settlement exists, that 10% recycles to UID 0.
- Deregistered or replaced hotkeys receive nothing because mapping happens against the current metagraph.

## Security boundaries

- Operator endpoints require `GITTENSOR_COMPUTE_TOKEN`.
- Miner registration, telemetry, and assignment acknowledgements require a fresh SN74 hotkey signature and one-time nonce.
- The operator-managed SparkCompute enrollment map binds each verifier node ID to its owner hotkey.
- Miner registration accepts no UID, release, concurrency, canary, latency, or performance fields.
- Assignment and weight challenges use `GITTENSOR_ASSIGNMENT_TOKEN` on the private miner-agent channel.
- The control plane stores routing metadata only. It never stores prompts, messages, model output, or proof text.
- The inference gateway verifies stream commitments in memory and discards user content.

## Run

```bash
export GITTENSOR_COMPUTE_TOKEN='<operator-token>'
export GITTENSOR_ASSIGNMENT_TOKEN='<miner-agent-token>'
export GITTENSOR_COMPUTE_DB='var/gittensor-compute.sqlite3'
uv run gitt-compute --config config/compute.example.json --host 127.0.0.1 --port 8780
```

Before production, install Cosign, copy the runtime signing public key to the configured path, and replace the example verifier measurement with the deployed attested measurement. The service intentionally rejects strict-mode GPUs while that value or the required SparkCompute evidence is missing.
