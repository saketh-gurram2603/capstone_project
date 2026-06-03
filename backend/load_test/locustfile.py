"""
Locust load test — Incident KB Assistant
=========================================

Tests three user types:
  SearchUser  — POST /search (pure retrieval, no LLM)
  ChatUser    — POST /chat   (multi-turn session with LLM intent classifier)
  TriageUser  — POST /triage (L1→L2→L3 agent pipeline, most expensive)

Usage
-----
  # Install (once):
  pip install locust

  # Baseline test (10 users, 2 min):
  locust -f load_test/locustfile.py --host http://localhost:8000 \
         --users 10 --spawn-rate 2 --run-time 2m --headless \
         --html load_test/reports/baseline.html

  # Stress test (ramp to 50 users):
  locust -f load_test/locustfile.py --host http://localhost:8000 \
         --users 50 --spawn-rate 5 --run-time 5m --headless \
         --html load_test/reports/stress.html

  # Interactive UI (open http://localhost:8089 in browser):
  locust -f load_test/locustfile.py --host http://localhost:8000

Shape classes
-------------
  BaselineShape  — steady 10 users for 2 min  (realistic concurrent ops)
  StressShape    — ramp 0→50 over 5 min, hold, then ramp down (find ceiling)
"""

from __future__ import annotations

import random
import time

from locust import HttpUser, LoadTestShape, between, events, task

# ── API prefix (matches app_config.json API_PREFIX) ───────────────────────────
PREFIX = "/it-kb"

# ── Realistic query pools ─────────────────────────────────────────────────────

SEARCH_QUERIES = [
    "Storage volume exceeded threshold causing upload failures",
    "Live video stream dropping out repeatedly for remote viewers",
    "Database queries timing out under peak load",
    "Multiple failed login attempts detected from external IPs",
    "Application response very slow under normal load",
    "CDN cache stale causing media content delays",
    "Encoder service crashed during video conversion job",
    "Processing service consuming excessive CPU",
    "Disk space running out on media server",
    "VPN users cannot authenticate remotely",
    "Memory usage spiking causing application restarts",
    "SSL certificate expired on load balancer",
    "Network broadcast stream failing intermittently",
    "Reports timing out with database query errors",
    "Monitoring agent not reporting to Grafana dashboard",
]

CHAT_OPENERS = [
    "Storage volume at 95% and new uploads are failing",
    "Live stream keeps dropping every few minutes",
    "Application is very slow and timing out for all users",
    "Getting failed login alerts from multiple IP addresses",
    "Encoder service keeps crashing on video jobs",
]

CHAT_FOLLOWUPS = [
    "ok this is not working",
    "I tried that but it didn't help, still seeing the issue",
    "what does step 2 mean exactly?",
    "where do I find that config file?",
    "can you explain the restart command?",
]

TRIAGE_QUERIES = [
    {
        "description": "Database connection pool exhausted during peak hours — all requests timing out with 'too many connections' error",
        "impact": "High",
        "urgency": "High",
    },
    {
        "description": "Storage volume at 98% capacity causing transcoding jobs to fail with no space left on device",
        "impact": "High",
        "urgency": "Medium",
    },
    {
        "description": "Live streaming service unresponsive — remote viewers getting connection refused errors",
        "impact": "High",
        "urgency": "High",
    },
    {
        "description": "Application intermittently slow under normal load — no obvious pattern in logs",
        "impact": "Medium",
        "urgency": "Medium",
    },
    {
        "description": "Multiple brute-force login attempts detected from external IP range over the last 30 minutes",
        "impact": "High",
        "urgency": "High",
    },
]


# ── User classes ──────────────────────────────────────────────────────────────

class SearchUser(HttpUser):
    """
    Simulates an ops engineer using the search page.
    Sends one search query per task — tests pure retrieval throughput.
    Expected: < 3s p95, no errors.
    """
    wait_time = between(1, 3)
    weight = 5   # 50% of virtual users

    @task
    def search(self):
        query = random.choice(SEARCH_QUERIES)
        with self.client.post(
            f"{PREFIX}/search",
            json={"query": query, "top_k": 10},
            catch_response=True,
            name="/search",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("results"):
                    resp.failure("Empty results returned")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(2)
    def search_with_filter(self):
        """Search with category + priority filter — exercises metadata filtering."""
        query = random.choice(SEARCH_QUERIES)
        category = random.choice(["Database", "Network", "Application", "Storage", "Security"])
        with self.client.post(
            f"{PREFIX}/search",
            json={"query": query, "top_k": 10, "category": category},
            catch_response=True,
            name="/search [filtered]",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


class ChatUser(HttpUser):
    """
    Simulates a user working through a multi-turn chat session.
    Opens a session, sends 2-4 follow-up turns, then closes.
    Tests: session manager, intent classifier, LLM calls.
    Expected: < 8s p95 per turn (LLM latency), no session corruption.
    """
    wait_time = between(2, 5)
    weight = 3   # 30% of virtual users

    def on_start(self):
        self.session_id: str | None = None

    @task
    def full_chat_session(self):
        """Open a session, simulate a real user working through a fix."""
        # Turn 1 — open new session
        opener = random.choice(CHAT_OPENERS)
        with self.client.post(
            f"{PREFIX}/chat",
            json={"message": opener},
            catch_response=True,
            name="/chat [new session]",
            timeout=15,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Session open failed: HTTP {resp.status_code}")
                return
            data = resp.json()
            session_id = data.get("session_id")
            if not session_id:
                resp.failure("No session_id in response")
                return
            resp.success()

        # Wait briefly — simulate user reading the response
        time.sleep(random.uniform(1, 3))

        # Turns 2–4 — follow-ups within the same session
        num_turns = random.randint(1, 3)
        for i in range(num_turns):
            followup = random.choice(CHAT_FOLLOWUPS)
            with self.client.post(
                f"{PREFIX}/chat",
                json={"session_id": session_id, "message": followup},
                catch_response=True,
                name="/chat [follow-up]",
                timeout=15,
            ) as resp:
                if resp.status_code == 200:
                    resp.success()
                elif resp.status_code == 404:
                    # Session expired — treat as expected under high load
                    resp.success()
                else:
                    resp.failure(f"Follow-up failed: HTTP {resp.status_code}")
                    break
            time.sleep(random.uniform(1, 2))

        # Final turn — resolve
        with self.client.post(
            f"{PREFIX}/chat",
            json={"session_id": session_id, "message": "that worked, thanks"},
            catch_response=True,
            name="/chat [resolve]",
            timeout=15,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Resolve failed: HTTP {resp.status_code}")


class TriageUser(HttpUser):
    """
    Simulates automated triage on a new incident.
    Most expensive user type — L1→L2 involves two LLM calls + web search.
    Expected: < 20s p95 (L2 path), escalation to L3 for novel queries.
    """
    wait_time = between(5, 15)
    weight = 2   # 20% of virtual users

    @task
    def triage_incident(self):
        payload = random.choice(TRIAGE_QUERIES)
        with self.client.post(
            f"{PREFIX}/triage",
            json=payload,
            catch_response=True,
            name="/triage",
            timeout=60,   # L2 web search + LLM can take up to 30s
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                level = data.get("escalation_level", "?")
                resp.success()
                # Tag the response label by escalation level for metrics breakdown
                self.environment.events.request.fire(
                    request_type="TRIAGE_LEVEL",
                    name=f"escalated_to_{level}",
                    response_time=0,
                    response_length=0,
                    exception=None,
                    context={},
                )
            else:
                resp.failure(f"HTTP {resp.status_code}")


# ── Load shapes ───────────────────────────────────────────────────────────────

class BaselineShape(LoadTestShape):
    """
    Steady 10 users for 2 minutes.
    Use this first to establish a healthy baseline and confirm all endpoints
    respond correctly before ramping up.

    Activate with:  --shape-class BaselineShape
    """
    stages = [
        {"duration": 30,  "users": 5,  "spawn_rate": 2},   # warm up
        {"duration": 120, "users": 10, "spawn_rate": 2},   # steady baseline
        {"duration": 150, "users": 0,  "spawn_rate": 5},   # cool down
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None


class StressShape(LoadTestShape):
    """
    Ramp 0 → 50 users over 5 minutes, hold for 3 minutes, then ramp down.
    Use this to find the saturation point and identify which component
    (LLM, Qdrant, BM25, SQLite) breaks first.

    Activate with:  --shape-class StressShape
    """
    stages = [
        {"duration": 60,  "users": 10, "spawn_rate": 2},   # gentle start
        {"duration": 120, "users": 25, "spawn_rate": 5},   # ramp
        {"duration": 200, "users": 50, "spawn_rate": 5},   # peak stress
        {"duration": 300, "users": 50, "spawn_rate": 1},   # hold at peak
        {"duration": 360, "users": 10, "spawn_rate": 10},  # ramp down
        {"duration": 390, "users": 0,  "spawn_rate": 10},  # done
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None


# ── Event hooks (printed to console during test) ──────────────────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "="*55)
    print("  Incident KB Load Test Starting")
    print(f"  Target: {environment.host}{PREFIX}")
    print(f"  Users: Search(50%) Chat(30%) Triage(20%)")
    print("="*55 + "\n")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total
    print("\n" + "="*55)
    print("  Load Test Complete")
    print(f"  Total requests : {stats.num_requests}")
    print(f"  Failures       : {stats.num_failures}")
    print(f"  Failure rate   : {stats.fail_ratio:.1%}")
    print(f"  Median latency : {stats.median_response_time:.0f} ms")
    print(f"  p95 latency    : {stats.get_response_time_percentile(0.95):.0f} ms")
    print(f"  p99 latency    : {stats.get_response_time_percentile(0.99):.0f} ms")
    print(f"  RPS            : {stats.current_rps:.1f}")
    print("="*55 + "\n")
