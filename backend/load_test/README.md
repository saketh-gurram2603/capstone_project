# Load Testing — Incident KB Assistant

Tool: **Locust** (Python-native, browser UI + HTML reports)

## Install
```bash
pip install locust
```

## Run — Baseline (do this first)
Steady 10 concurrent users for 2 minutes. Confirms all endpoints are healthy
before stress testing.

```bash
cd backend
locust -f load_test/locustfile.py \
       --host http://localhost:8000 \
       --users 10 --spawn-rate 2 --run-time 2m \
       --headless \
       --html load_test/reports/baseline.html
```

## Run — Stress (find the ceiling)
Ramps 0 → 50 users over 5 minutes, holds, then ramps down.

```bash
cd backend
locust -f load_test/locustfile.py \
       --host http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 8m \
       --headless \
       --html load_test/reports/stress.html
```

## Run — Interactive UI (best for demos)
Opens a live dashboard at http://localhost:8089.
Set users and spawn-rate in the browser, watch RPS and latency charts live.

```bash
cd backend
locust -f load_test/locustfile.py --host http://localhost:8000
```

## User mix
| User type   | Weight | What it tests |
|-------------|--------|---------------|
| SearchUser  | 50%    | Hybrid retrieval — BM25 + vector + reranker |
| ChatUser    | 30%    | Multi-turn sessions — session manager + LLM intent classifier |
| TriageUser  | 20%    | L1→L2 agent pipeline — most expensive (LLM + web search) |

## Target SLOs
| Endpoint  | p50   | p95    | Error rate |
|-----------|-------|--------|------------|
| /search   | <1s   | <3s    | <1%        |
| /chat     | <3s   | <8s    | <2%        |
| /triage   | <10s  | <20s   | <5%        |

## Reports
HTML reports are written to `load_test/reports/`.
Open in any browser — include screenshots in SME documentation.
