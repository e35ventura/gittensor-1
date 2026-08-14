# Gittensor Compute Sub-Subnet

## Build contract

One global pool of verified RTX 5090s serves every approved Gittensor model and runtime. There are no model lanes and miners do not choose placements.

| Pillar | Required behavior |
|---|---|
| GPU market | Maintain a baseline GPU floor. Scale one desired target from time-weighted GPU-equivalent demand. Use a sublinear scarcity premium below target and dilution above target. |
| Global Gepetto | Produce and execute one `gpu_id -> release_digest` map across the entire verified fleet. Distinguish total supply shortage from wrong-release placement shortage. |
| Verification | Verify unique RTX 5090 hardware, driver and uptime, pinned model/tokenizer revisions, signed runtime identity, local weights and response continuity. No-retention and host-resistant proof are a separate hardware tier. |
| Router | Reserve the compatible READY GPU with the earliest predicted completion. Return 429 immediately when concurrency or KV capacity is full. |

## Request path

```text
OpenAI request
  -> regional gateway
  -> global fastest-finish reservation
  -> one-use reservation capability
  -> isolated approved runtime
  -> per-chunk runtime signature
  -> gateway verification
  -> user
```

The gateway never queues requests internally. It either creates an atomic, expiring reservation or returns 429. Successful and rejected requests feed the same global demand measurement.

## Dynamic target and emissions

Configuration defines:

- `F`: minimum GPU floor.
- `C_r`: certified maximum concurrency for release `r`.
- `M_r`: certified KV-cache byte budget for release `r`.
- `T_req`: utilization-driven desired target.
- `P`: target accounting price per verified GPU-hour.
- `V`: current realized value of total subnet miner emissions per hour, in the same unit as `P`.
- `B_cap`: optional external budget cap. `null` means target funding grows automatically.
- `S_max`: maximum compute share available from the subnet.

```text
request GPU fraction = max(1 / C_r, request KV bytes / M_r)
accepted demand = average time-weighted GPU fractions during the control window
rejected demand = EWMA(sum(429 request GPU fractions × expected service time) / window time)
demand = max(live reservations, accepted demand) + rejected demand
utilization = demand / funded target
required target = max(F, floor(demand / utilization_up) + 1)
target budget = T_req × P
emission-value ceiling = V × S_max
funded target = min(T_req, floor(min(B_cap, emission-value ceiling) / P))
funded pool = min(T_req × P, B_cap, emission-value ceiling)
paid compute share = distributed pool / V
reserved compute share = funded pool / V
```

- Each approved release declares `C_r`, maximum context, KV bytes per token, `M_r`, request overhead and load cost. These values are part of its canonical release digest and signed Gepetto assignment. The fleet-level `certified_slots_per_gpu` is only an operator safety ceiling.
- Sustained utilization at or above `utilization_up` raises `T_req` to `required target`, which leaves utilization strictly below the threshold after scaling.
- Sustained utilization at or below `utilization_down` lowers `T_req` by one after cooldown, never below `F`.
- Product scaling has no configured GPU maximum. The subnet emission cap is the economic ceiling and is surfaced as a funding shortfall.
- A funding shortfall never makes `T_req` grow by itself. It rises only when measured demand requires more target GPUs.
- The production oracle derives `V` from the latest finalized SN74 `IncentiveAlphaEmittedToMiners` event, exact event-block timestamps, the subnet alpha-to-TAO moving price at the finalized head and, for USD targets, the median of independent Coinbase and CoinGecko TAO/USD observations. It rejects missing, stale or divergent evidence and does not require an archive RPC.
- Oracle freshness is part of funding. If refreshes fail for five minutes, the funding request falls back to `F × P`; scaled capacity is not funded from stale conversion data. If the subnet cap cannot fully fund `F`, the status exposes that funding shortfall instead of claiming the baseline is economically guaranteed. Manual emission-value changes are disabled while the automatic oracle is enabled.
- `funded target` is a whole-GPU routing count. Settlement retains the fractional `funded pool`, so an emission ceiling worth 3.8 GPU-hours reports three funded target GPUs and 3.8 effective funded GPUs for reward calculation.
- The current implementation intentionally refuses multiple on-chain incentive mechanisms because the miner-emission event does not identify a mechanism. Add a mechanism-specific event or oracle before enabling that Bittensor feature.
- `P` is an accounting target, not a guaranteed fiat payout. Realized miner income also depends on the accuracy of `V`, validator participation and Bittensor's final on-chain emission allocation.

Settlement applies a sublinear scarcity curve below the funded target:

```text
R = effective READY GPUs during the window
E = effective funded GPUs represented by the funded pool
scarcity multiplier = min(cap, (E / R) ^ beta) when 0 < R < E, otherwise 1
distributed pool = min(funded pool, P × R × window hours × scarcity multiplier)
```

`beta` must be between zero and one. This pays each scarce GPU more than `P` while preserving a positive reward for every additional GPU. Any reserved compute share not distributed by the curve goes to recycle, not to open-source rewards.

Example: `F=4`, `P=0.65`, `V=26.00/hour`, `beta=0.5`, scarcity cap `2.0`.

| READY supply | Effective funded GPUs | Distributed/hour | Paid compute share | Payment/GPU |
|---:|---:|---:|---:|---:|
| 1 | 4 | 1.30 | 5.00% | 1.300 |
| 2 | 4 | 1.84 | 7.07% | 0.919 |
| 4 | 4 | 2.60 | 10.00% | 0.650 |
| 8 | 4 | 2.60 | 10.00% | 0.325 |
| 6 after scale-up | 6 | 3.90 | 15.00% | 0.650 |

Settlement uses verified READY-seconds, not registration, self-reported uptime or request count. Settlement and its next accounting checkpoint commit in one SQLite transaction.

## Global Gepetto

Gepetto reads approved releases, verified hardware, per-release GPU-equivalent demand, minimum replicas and placement weights. It emits one global assignment map.

The control plane reports two independent shortages:

- Supply shortage: total demand exceeds the funded global GPU target.
- Placement shortage: enough GPUs may exist globally, but too few currently run a requested release.

Minimum replicas repair immediately. Other model switches require the placement shortage to remain above a configured gain band for `switch_sustain_seconds`. A switch is also deferred when drain time plus certified load time exceeds the capacity-seconds it can recover over `planning_horizon_seconds`. Shortage timers and deferred reasons persist across restart.

Every transition is monotonic and durable:

```text
REGISTERED -> DRAINING -> LOADING -> RUNTIME_VERIFY -> READY
```

The control plane persists an epoch before delivery. Commands are signed by the validator hotkey and bind the method, path, exact body, timestamp and one-time nonce. Failed deliveries retry the same epoch. Miner acknowledgements are signed by the miner hotkey and are idempotent across network retries.

Each assignment rotates a separate GPU-scoped inference token. Only route-authorized gateways receive it. Gateways never hold the validator hotkey and cannot issue placements, challenges or settlements.

The miner agent:

1. Stops accepting new inference and drains active requests.
2. Downloads the exact model and tokenizer commits into an atomic local snapshot.
3. Checks every declared weight-file size.
4. Pulls the exact signed `image@sha256:digest`.
5. Runs it read-only, capability-free, PID-limited, without container logs and on an internal network with no external egress.
6. Starts its model backend on loopback and `gitt-compute-runtime-proxy` on port 8000. The proxy creates its signing key inside the approved runtime process. The miner agent receives only the public key.
7. Accepts runtime health only when the proxy can reach the loopback model backend, validates the runtime identity and proof key, then waits for independent verification before routing can begin.

## Layered verification

| Layer | Admission rule |
|---|---|
| Ownership | A live SN74 hotkey signature resolves the UID from the metagraph. The SparkCompute enrollment map must bind the node to that same hotkey. |
| Hardware | SparkCompute must verify RTX 5090 identity, GPU UUID, driver, timed GPU work, heartbeat and uptime. The trusted verifier also challenges unpredictable overlapping batches (three by default), so one physical GPU cannot satisfy several identities sequentially. Duplicate UUID claims are quarantined. |
| Verifier | Strict mode requires the configured protocol, source commit and measured verifier build. Each verifier shard signs its exact complete status response with an offline-pinned public key. |
| Release | The canonical digest includes exact model and tokenizer repositories and commits, runtime commit, container digest, filesystem digest, weight manifest, stream-proof scheme, concurrency, context, KV budget, request overhead and load cost. |
| Runtime | Cosign verifies the image digest before release admission. SparkCompute must report the exact running release and runtime measurements. This is host-resistant only when backed by a real TEE. |
| Weights | Repeated unpredictable byte-range challenges are answered from the active local release and checked against the pinned Hugging Face commit. |
| Response | Each OpenAI chunk is signed by a runtime key whose public key is bound into the verifier evidence. The gateway rejects missing, invalid or reordered proofs. This binding is host-resistant only in the confidential tier. |
| Serving | Real gateway failures are bound to exact reservations. Repeated failures quarantine the GPU until newer verification evidence is observed. |
| User data | The 5090 tier is read-only, logless and egress-isolated, but it cannot make a cryptographic no-retention guarantee against the machine owner. `ephemeral-no-retention-v1` requires a separately configured confidential-compute GPU tier. |

The checks are complementary. Weight challenges alone do not prove inference used those weights. Stream signatures alone do not prove the signing process ran the correct model. READY requires the complete configured chain. On RTX 5090 that chain is layered software verification, not a cryptographic guarantee against the host owner.

### SparkCompute integration contract

The pinned upstream SparkCompute commit does not currently emit every strict field below. The verifier integration must add them. Until it does, production configuration intentionally fails closed and no GPU becomes READY.

`config/compute.example.json` contains an all-zero verifier measurement and an empty signer list on purpose. The control plane refuses to start until the verification engineer replaces them with the measured digest and trusted public key of the deployed verifier build.

SparkCompute currently polls nodes sequentially inside each verifier. The verification engineer must add a coordinator that selects unpredictable groups, challenges every identity in a group concurrently with overlapping GPU-saturating deadlines and records one batch result per identity. Production fleets should shard ordinary polling across multiple verifier processes and list their `/api/status` endpoints in `verification.status_urls`; the control plane fetches those shards concurrently and merges them by node ID.

Every shard must sign the exact raw `/api/status` body as `sr25519("gittensor-spark-verifier-status-v1\\n" + sha256(body))`, returning its public key in `X-Gittensor-Verifier-Public-Key` and signature in `X-Gittensor-Verifier-Signature`. Pin allowed 32-byte public keys in `verification.trusted_verifier_public_keys`. TLS protects transport; this signature establishes evidence provenance and supports explicit key rotation. Duplicate JSON keys, duplicate security headers, wrong content types, malformed signatures and untrusted keys are rejected.

```json
{
  "id": "enrolled-spark-node-id",
  "verdict": "VERIFIED",
  "last_checked": 1770000000,
  "verifier": {
    "protocol": "sparkcompute-v1",
    "source_commit": "aa61fbc20d24c217ea9f33e20be67d75a388e45d",
    "measurement": "sha256:<deployed-verifier-measurement>"
  },
  "report": {
    "gpu": {
      "name": "NVIDIA GeForce RTX 5090",
      "uuid": "GPU-...",
      "driver_version": "..."
    },
    "runtime": {
      "release_digest": "sha256:<canonical-release>",
      "model_repository": "owner/model",
      "model_revision": "<40-character-commit>",
      "tokenizer_repository": "owner/tokenizer",
      "tokenizer_revision": "<40-character-commit>",
      "runtime_digest": "sha256:<runtime>",
      "runtime_commit": "<40-character-commit>",
      "container_image": "registry/runtime",
      "container_digest": "sha256:<image>",
      "filesystem_digest": "sha256:<filesystem>"
    }
  },
  "liveness": {
    "online": true,
    "online_age_sec": 1,
    "gpu_live": true,
    "model_canary_enabled": true,
    "model_verified": true,
    "model_age_sec": 1
  },
  "uniqueness": {
    "verified": true,
    "batch_id": "random-overlapping-batch-id",
    "batch_size": 3,
    "verified_at": 1770000000,
    "challenge_digest": "sha256:<verifier-checked-batch-evidence>"
  },
  "attestation": {
    "verified": true,
    "evidence_digest": "sha256:<signed-evidence>",
    "stream_public_key": "<32-byte-sr25519-public-key-hex>",
    "confidential_compute": false,
    "data_policy": "isolated-no-logging-v1"
  }
}
```

The approved image must run `gitt-compute-runtime-proxy` in front of a loopback-only OpenAI backend. The proxy exposes `/health` and `/v1/gittensor/runtime`, enforces exact assignment/request headers, rejects backend redirects and unexpected response types, removes backend-supplied proof fields and signs every response with `sr25519-response-v1`. It signs the complete OpenAI payload, canonical request-body digest, request ID, creation time, release, model, revision and chunk index. A signed terminal event must precede `[DONE]`, so the gateway rejects truncated streams as failures.

The response-signing private key exists only in the approved runtime process. A changed key immediately revokes the old assignment, rotates its capability secret, removes its reservations and requires a new Gepetto assignment plus fresh SparkCompute evidence before the GPU can return to READY. The miner agent never receives or stores the private key.

NVIDIA's current [Trusted Computing supported-SKU list](https://docs.nvidia.com/590trd1-trusted-computing-solutions-release-notes.pdf) includes data-center products such as RTX PRO 6000 Blackwell Server Edition, but not GeForce RTX 5090. Therefore the 5090 profile sets `require_confidential_compute: false`. Set it to true only for a hardware and CVM deployment that can produce fresh CPU, GPU, guest-image and policy attestation. Marketing or software-only claims must never populate that field.

## Fastest-finish routing

Only READY GPUs with a live verification lease, the exact release, free certified concurrency and enough unreserved KV bytes are candidates. Before routing, the gateway converts `max_completion_tokens` to one canonical `max_tokens` cap and inserts a 256-token cap when neither field is supplied. Requests that set both fields or request `n != 1` are rejected with HTTP 400 because their exact KV use cannot be reserved. The gateway and miner use the same deterministic byte-level upper bound for request context, and the miner independently repeats the normalization. Each reservation records its KV bytes and GPU fraction, and the one-use HMAC capability binds the KV amount. The miner rejects any request larger than its signed reservation.

```text
predicted finish = measured gateway RTT
                 + measured runtime backlog, if any
                 + request-specific predicted service time × GPU/release calibration
```

Completion time is primary. Inside a small equivalent-finish band, lower active concurrency wins. Equivalent GPUs in one location therefore receive one request each before requests stack on a single GPU.

Reservations are small transactional rows, not full control-plane snapshots. Creation is atomic and expiry extends past the predicted request duration. Old reservations without capacity fields migrate fail-closed until their short TTL expires. Gateways cache independent HTTPS health-probe RTT by endpoint and separate it from full inference time; observations update per-region RTT and per-release performance calibration.
The miner independently enforces the assignment's signed concurrency limit. Every route receives a unique, expiring, one-use inference capability rather than the reusable assignment secret. Capabilities and reservations are capped by the verification lease. Any verification, weight or serving failure that quarantines a GPU rotates its assignment secret, deletes its reservations, removes its Gepetto placement and sends the miner a signed epoch revocation. Recovery requires a new Gepetto assignment and the full verification chain. The gateway renews reservations during long responses, and renewal fails if the GPU loses READY status or changes release.

## Processes and credentials

| Process | Command | Credential scope |
|---|---|---|
| Control plane and global Gepetto | `gitt-compute` | Operator token plus validator hotkey |
| Regional inference gateway | `gitt-compute-gateway` | Gateway API token and route-only control token. It holds no validator private key. |
| One miner GPU agent | `gitt-compute-miner` | Miner hotkey and expected validator hotkey |
| Approved runtime sidecar | `gitt-compute-runtime-proxy` | In-memory response-signing key by default |

```bash
export GITTENSOR_COMPUTE_TOKEN='<operator-only-token>'
export GITTENSOR_GATEWAY_CONTROL_TOKEN='<route-only-token>'
export GITTENSOR_GATEWAY_TOKEN='<user-facing-api-token>'
export GITTENSOR_COMPUTE_DB='var/gittensor-compute.sqlite3'
export GITTENSOR_COMPUTE_SETTLEMENT_URL='https://compute-control.gittensor.ai/v1/settlements/latest'
export GITTENSOR_COMPUTE_SETTLEMENT_HOTKEY='<validator-control-hotkey>'

uv run gitt-compute --config config/compute.example.json --host 127.0.0.1 --port 8780
uv run gitt-compute-gateway --config config/compute-gateway.example.json --port 8782
uv run gitt-compute-miner --config config/compute-miner.example.json --port 8781
```

Run one active control-plane process per SQLite database. Multiple regional gateways share that control plane, so reservations and demand remain global. An HA deployment must use a single elected writer or replace SQLite with an equivalent transactional shared store.

The included gateway token is an internal deployment credential, not a public multi-tenant billing system. Production traffic must enter through authenticated quotas or billing and rate limits so only legitimate inference demand can move the GPU target.

### Emergency controls

The operator credential, never the gateway credential, controls three fail-closed actions:

| Endpoint | Effect |
| --- | --- |
| `POST /v1/gpus/disable` | Immediately removes one GPU from placement and routing, closes its reservations and persists the quarantine. |
| `POST /v1/gpus/enable` | Clears the operator quarantine but returns the GPU unassigned. Gepetto assignment and the full verification chain must run again. |
| `POST /v1/releases/revoke` | Removes an approved release and invalidates every GPU assignment, capability and reservation bound to it. |

Each disable or revocation requires a non-empty audit reason. The control plane rotates the capability and sends a signed, epoch-bound tombstone to the miner agent. That tombstone is durable, idempotent and retried after restart, so previously issued capabilities stop working even before runtime cleanup finishes.

## Validator integration

Co-located validators may read `GITTENSOR_COMPUTE_DB`. Remote validators use `GITTENSOR_COMPUTE_SETTLEMENT_URL` and pin `GITTENSOR_COMPUTE_SETTLEMENT_HOTKEY`. The public feed is signed by the global control hotkey, binds the exact window, rewards and emission share, and is rejected if it is stale or tampered with. Each round:

1. Map settlement hotkeys onto the current metagraph.
2. Read the target-scaled compute emission share from the atomic settlement.
3. Normalize verified READY-time rewards inside that share.
4. Recycle stale, malformed or empty compute allocation instead of paying unverifiable work or releasing the reserved compute slice.

Deregistered or replaced hotkeys receive nothing because UID mapping happens at weight time.
The validator installs each finalized allocation without an additional score EMA. This preserves the target-price compute percentage and prevents departed or quarantined GPUs from retaining historical payout weight.
