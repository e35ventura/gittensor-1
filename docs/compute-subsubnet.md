# Compute Sub-Subnet

This fork adds a shared RTX 5090 execution layer for approved Gittensor model and runtime releases. It is a control-plane reference implementation with four explicit owners.

| Owner | Responsibility |
|---|---|
| SparkCompute | Prove RTX 5090 hardware, driver health, liveness, uptime, GPU work, and the model canary. |
| Fleet control | Maintain a demand-driven desired target and apply the hourly funding guard. |
| Global Gepetto | Produce one `gpu_id -> release_digest` placement map across the verified fleet. |
| Router | Atomically reserve the compatible READY GPU expected to finish first. |

## Economic contract

Configuration defines:

- `F`: minimum desired GPU floor.
- `T_req`: demand-driven desired GPU target.
- `P`: target reward per GPU-hour.
- `B_max`: maximum availability budget per hour.
- `K`: operator-certified request slots per GPU.

The funded target is:

```text
T = min(T_req, floor(B_max / P))
```

Every verified READY GPU may join. `T` is a price target, not an admission cap. The window budget is divided by verified READY seconds, so supply below `T` earns more per GPU and supply above `T` dilutes the reward.

With `T_req=4`, `P=$0.65`, and `B_max=$2.60` for a one-hour window:

| READY GPUs | Reward per GPU-hour |
|---:|---:|
| 2 | $1.30 |
| 4 | $0.65 |
| 8 | $0.325 |

## Autoscaling contract

Demand is measured as:

```text
D = active reservations + EWMA(429 requests/second * expected service seconds)
U = D / (T * K)
```

- If `U >= utilization_up` continuously for `sustain_up_seconds`, increase `T_req` to at least `ceil(D / (K * utilization_up))`.
- If `U <= utilization_down` continuously for `sustain_down_seconds`, decrease `T_req` by one, never below `F`, and at most once per cooldown.
- `T_req` has no product-level maximum.
- If funding cannot cover `T_req`, the system exposes `funding_shortfall`.
- The validator or treasury updates `B_max` through `POST /v1/funding` as the compute share of subnet emissions changes. Raising `B_max` funds more of the unbounded desired target without changing the per-GPU target price. An update below `F * P` is rejected so the baseline fleet remains funded.
- If no compatible READY slot exists, the router returns 429 immediately. It does not create an internal request queue. The rejection feeds back into demand.

## Verification contract

The implementation consumes SparkCompute's `GET /api/status` response. A GPU receives a short-lived Gittensor lease only when all conditions are true:

1. SparkCompute verdict is `VERIFIED`.
2. Reported hardware is an RTX 5090 and the driver version is present.
3. Heartbeat and GPU micro-challenge are live.
4. Model canaries are enabled and currently verified.
5. The SparkCompute snapshot is fresh.
6. The trusted canary binding matches an approved Gittensor `release_digest`.

The control plane also rejects duplicate GPU UUIDs, so one SparkCompute-reported device cannot occupy two registrations accidentally. The UUID is a useful uniqueness guard, not cryptographic proof. Production verification still needs simultaneous randomized challenges across sampled node identities to prevent a sophisticated operator from proxying one physical GPU behind spoofed UUIDs.

Lease expiry immediately removes the GPU from routing and READY-second earnings.

SparkCompute currently proves canary continuity but does not publish a cryptographic release digest in `/api/status`. This fork therefore stores a trusted `canary_release_digest` binding. Operators must bootstrap a new canary set after every model, quantization, runtime, build, or behavior-affecting configuration change, then bind that canary set to the approved release digest. This is the MVP trust boundary to remove in a later SparkCompute protocol revision.

SparkCompute's verifier API is unauthenticated by default. Keep it on a private network or place it behind an authenticated reverse proxy. The adapter supports a bearer token through `verification.bearer_token_env`.

## Global Gepetto contract

Gepetto is subnet-level, not miner-level. It reads approved releases, per-release demand, current assignments, verified GPUs, and minimum residency. It emits one global assignment map.

Changing a GPU assignment follows:

```text
DRAINING > LOADING > RUNTIME_VERIFY > READY
```

The GPU does not route or earn during the transition. It becomes READY again only after SparkCompute verifies the newly bound runtime canary.

There are no fixed hardware lanes. Releases receive replicas from the same global GPU supply according to demand, declared minimum replicas, and placement weight.

## Routing contract

The router considers only GPUs with:

- a live verification lease;
- READY state;
- the exact requested release;
- at least one free certified slot.

For equivalent RTX 5090s it spreads one request to every idle GPU before stacking requests. It then minimizes:

```text
expected completion = measured RTT + remaining work + expected service time
```

Capacity is reserved atomically before the endpoint is returned. Reservation expiry prevents abandoned requests from holding capacity forever.

## Privacy contract

The control plane accepts only routing metadata: release digest, requester region, and expected service time. The API rejects unexpected request fields, including inference payloads. It must not store prompts, messages, user content, or model outputs.

## Run locally

1. Run the pinned SparkCompute node agent and verifier from `gittensor-ai-lab/sparkcompute`.
2. Copy `config/compute.example.json` and set the verifier URL, target, budget, concurrency, and thresholds.
3. Start the control plane:

```bash
export GITTENSOR_COMPUTE_TOKEN='<strong-random-token>'
uv run gitt-compute --config config/compute.example.json --host 127.0.0.1 --port 8780
```

The mutable endpoints require `Authorization: Bearer <token>`. `GET /health` is public.

## API sequence

1. `POST /v1/releases`: approve immutable model/runtime releases.
2. `POST /v1/gpus`: register the GPU, SparkCompute node ID, miner UID, endpoint, current release, and trusted canary binding.
3. `POST /v1/verification/refresh`: consume SparkCompute state and issue or revoke leases.
4. `POST /v1/control/tick`: update the dynamic target and global placement map.
5. `POST /v1/funding`: update the hourly compute budget from current subnet emissions.
6. `POST /v1/route`: reserve capacity or receive 429.
7. `POST /v1/reservations/complete`: release the reservation.
8. `POST /v1/settlement`: divide the funded window budget by verified READY seconds.

## Production boundary

This fork implements and tests the core rules, state machine, SparkCompute adapter, HTTP contract, and reference single-process service. Before public traffic, replace in-memory registrations and assignments with durable storage, use a regional atomic reservation store, add signed assignment epochs, connect normalized miner rewards to the validator's chosen compute emission pool, and run the router behind the provider gateway. No mainnet emission split is changed by this fork.
