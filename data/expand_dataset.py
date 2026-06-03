"""
Dataset expansion script — v2.
Resets incidents.xlsx to the original 150 rows, then appends
165 new rows across 30 IT problem types.

KEY DESIGN: every row has a UNIQUE description highlighting the specific
symptom that led to that particular solution.  No two rows in the same
problem type share the same description text, so the preprocessor's
(title + description + solution) dedup hash is always distinct and the
search results show cards that look genuinely different.

Run from the data/ directory:
    python expand_dataset.py
"""

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

XLSX          = Path(__file__).parent / "incidents.xlsx"
ORIGINAL_ROWS = 150   # keep rows 0..149 exactly as they are

# ── Timestamp generation ──────────────────────────────────────────────────────
# Realistic per-category resolution time ranges (min_hours, max_hours).
# Used for dataset rows and as fallback when new uploads lack timestamp columns.
CATEGORY_RESOLUTION_HOURS: dict[str, tuple[float, float]] = {
    "Storage":        (2.0,  8.0),
    "Application":    (1.0,  6.0),
    "Database":       (1.0,  4.0),
    "Network":        (0.5,  3.0),
    "Security":       (2.0, 12.0),
    "Performance":    (1.0,  4.0),
    "Hardware":       (4.0, 24.0),
    "Authentication": (0.5,  2.0),
    "Monitoring":     (0.5,  1.5),
    "Configuration":  (0.5,  2.0),
}

_TS_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
_TS_END   = datetime(2025, 11, 30, tzinfo=timezone.utc)
_RNG      = random.Random(42)   # fixed seed → reproducible timestamps


def _make_timestamps(category: str) -> tuple[str, str, float]:
    """Return (opened_at_iso, resolved_at_iso, resolution_hours)."""
    span = int((_TS_END - _TS_START).total_seconds())
    opened_at = _TS_START + timedelta(seconds=_RNG.randint(0, span))
    min_h, max_h = CATEGORY_RESOLUTION_HOURS.get(category, (1.0, 6.0))
    res_hours = round(_RNG.uniform(min_h, max_h), 2)
    resolved_at = opened_at + timedelta(hours=res_hours)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return opened_at.strftime(fmt), resolved_at.strftime(fmt), res_hours

# ── 30 problem types ───────────────────────────────────────────────────────────
# Each entry in "rows" is (description_unique_to_this_fix, solution).
# Descriptions within the same type are always different — different symptom
# angle, different context, or different severity indicator.
# ──────────────────────────────────────────────────────────────────────────────
TYPES = [

    # ── STORAGE ──────────────────────────────────────────────────────────────

    dict(
        category="Storage", title="Disk Space Threshold Exceeded",
        assets=["MediaServer01", "MediaServer03", "MediaServer07",
                "MediaServer02", "MediaServer05", "MediaServer06"],
        rows=[
            ("Storage volume exceeded the 90% threshold. Upload jobs are failing with 'no space left on device' errors.",
             "Archived media files older than 90 days to cold storage. Freed 2.3 TB. Configured automated archival policy to run weekly."),
            ("Primary disk at 95% capacity. Monitoring alerts firing. Log directory alone consuming 800 GB.",
             "Enabled log rotation on all application and system logs. Compressed logs older than 30 days. Recovered 400 GB immediately."),
            ("Media server storage volume is 97% full. New transcoding jobs queuing and timing out.",
             "Provisioned an additional 4 TB disk and extended the LVM volume group online. No downtime required."),
            ("Disk usage growing 20 GB per day from incomplete multipart uploads never cleaned up.",
             "Identified and removed orphaned temporary upload files and incomplete multipart upload parts. Cleaned 800 GB of unreferenced fragments."),
            ("Storage at 91%. Deduplication not enabled despite similar media files being stored multiple times.",
             "Enabled data deduplication on the volume. Achieved 30% space savings on existing content without deleting any files."),
            ("Critical threshold hit on media archive volume. Hot tier storage filling up with infrequently accessed content.",
             "Migrated infrequently accessed media from hot to cold storage tier using a lifecycle policy. Hot tier utilisation dropped to 60%."),
        ],
    ),

    dict(
        category="Storage", title="Storage I/O Bottleneck",
        assets=["MediaServer02", "MediaServer05", "MediaServer08"],
        rows=[
            ("Disk I/O wait at 80%. Media processing jobs taking 10x longer than normal. HDD throughput saturated.",
             "Migrated hot media content from HDD to NVMe SSD storage tier. Average I/O latency reduced from 45 ms to 3 ms."),
            ("High read I/O from repeated access to same media files. Cache miss rate near 100%.",
             "Enabled read caching using a dedicated SSD cache volume. Frequently accessed media served from cache — disk reads reduced by 60%."),
            ("Three applications writing to the same volume simultaneously causing I/O contention.",
             "Distributed media files across three separate volumes using a round-robin write policy. Parallelised I/O load and eliminated single-disk contention."),
            ("Sequential write throughput degraded since OS I/O scheduler was changed to CFQ during an update.",
             "Changed the Linux I/O scheduler from CFQ back to Deadline for the storage block device. Sequential throughput improved 40%."),
            ("Transcoding pipeline blocks on disk writes — CPU idle while waiting for slow synchronous writes.",
             "Enabled async I/O on the media processing pipeline. Jobs no longer block on disk writes. Processing throughput improved 35%."),
        ],
    ),

    dict(
        category="Storage", title="NAS Mount Failure",
        assets=["NASServer01", "NASServer02", "AppServer03"],
        rows=[
            ("NAS NFS share shows as stale mount. Read and write operations hang indefinitely without returning.",
             "Unmounted the stale NFS share with umount -l and performed a clean remount with noatime and soft timeout options."),
            ("After a network switch replacement, the NAS mount point stopped responding to I/O.",
             "Restarted nfs-client and rpcbind services. Mount auto-recovered and I/O resumed normally after service restart."),
            ("NFS mount fails on boot. /etc/fstab entry uses old IP address of the NAS after it was moved.",
             "Updated the /etc/fstab entry with the correct NAS hostname. Mount now survives reboots without manual intervention."),
            ("Application server cannot reach NAS share on port 2049. Share accessible from other hosts.",
             "Added firewall rule allowing NFS ports 2049 and 111 from the application server subnet. Mount succeeded immediately."),
            ("NAS credentials used for SMB mount expired. Application writing files to a local fallback path silently.",
             "Updated the SMB mount credentials in /etc/samba/credentials and remounted the share. Files now writing to the correct NAS path."),
        ],
    ),

    dict(
        category="Storage", title="RAID Array Degraded",
        assets=["StorageController01", "StorageController02"],
        rows=[
            ("RAID 6 array reporting one failed disk. Array in degraded mode. One more drive failure will cause data loss.",
             "Identified the failed physical disk via RAID controller diagnostics. Hot-swapped with a spare drive. Array rebuild completed in 4 hours."),
            ("RAID controller showing 'predictive failure' on a drive before it fully failed. Performance degraded.",
             "Replaced the predictive-failure drive during a maintenance window before actual failure. Array never entered degraded mode."),
            ("Array degraded and no hot spare available. Full backup required before replacing the failed drive.",
             "Performed full backup while in degraded mode, then replaced the failed drive and verified data integrity after rebuild."),
            ("RAID controller's cache battery failed causing write-back cache to disable. Write performance collapsed.",
             "Replaced the RAID controller cache battery alongside the failed drive. Write-back cache re-enabled after battery charged."),
            ("Array rebuild failed halfway through due to another drive showing CRC errors under the rebuild stress.",
             "Replaced both the originally failed drive and the CRC-error drive. Restored from backup to the new drives after confirming hardware health."),
        ],
    ),

    # ── APPLICATION ───────────────────────────────────────────────────────────

    dict(
        category="Application", title="Application Slow Response",
        assets=["AppServer01", "AppServer02", "AppServer04",
                "AppServer05", "AppServer06", "AppServer07"],
        rows=[
            ("API response times jumped from 200 ms to 6 seconds after a CDN configuration change.",
             "Refreshed the CDN cache and pre-warmed the top 50 most-accessed endpoints. Response times returned to under 300 ms within 10 minutes."),
            ("Dashboard queries timing out. A missing database index identified by profiling as the root cause.",
             "Added a B-tree index on the user_sessions.created_at column identified as missing via EXPLAIN ANALYZE. Query time dropped from 4 s to 12 ms."),
            ("Application response slow under normal load. Thread pool was running at 100% saturation.",
             "Increased max worker threads from 50 to 200 and enabled connection pooling. Application throughput increased 4x without additional hardware."),
            ("All users experiencing slow loads. Auto-scaling not triggered because CPU was within limits but I/O was saturated.",
             "Scaled horizontally by adding two application instances to the load balancer pool. Load distributed evenly and response times recovered."),
            ("API payloads large — 2 MB responses for simple data requests due to no compression enabled.",
             "Enabled gzip response compression on the API gateway. Payload size reduced 70%. Response times improved for all clients on slower connections."),
            ("Application slow after a deployment that accidentally disabled the result cache.",
             "Re-enabled the Redis result cache that was turned off in the deployment config. Average response time dropped from 3 s to 80 ms."),
        ],
    ),

    dict(
        category="Application", title="Memory Leak Causing OOM",
        assets=["AppServer03", "AppServer05", "AppServer08"],
        rows=[
            ("JVM heap usage climbs steadily over 8 hours then crashes with OutOfMemoryError. Requires manual restart.",
             "Captured heap dump with jmap during spike. Found a static HashMap cache never being evicted. Replaced with an LRU cache capped at 10,000 entries."),
            ("Application process killed by the Linux OOM killer every 4-6 hours. Memory usage never levels off.",
             "Configured a systemd timer to restart the application service nightly at 02:00 as an immediate mitigation while root cause was investigated."),
            ("Heap dump analysis showed a third-party JSON library leaking large objects on repeated API calls.",
             "Upgraded jackson-databind from 2.9.x to 2.14.x which contains the fix for the known memory leak on large payload parsing."),
            ("JVM running without -Xmx set. Heap allowed to grow to 28 GB on a 32 GB host, starving other processes.",
             "Set -Xmx12g to cap the JVM heap and leave headroom for OS and other services. Memory usage stabilised immediately."),
            ("Memory leak too complex to fix immediately. Need visibility to auto-recover before it causes a full outage.",
             "Exported JVM heap metrics to Prometheus and configured an auto-restart trigger when heap exceeds 85% for more than 10 minutes."),
        ],
    ),

    dict(
        category="Application", title="Thread Pool Exhaustion",
        assets=["AppServer01", "AppServer06", "AppServer09"],
        rows=[
            ("All threads occupied. Logs showing RejectedExecutionException. New requests queuing and timing out.",
             "Increased Tomcat maxThreads from 200 to 500 and minSpareThreads from 10 to 50. Restarted service — request queue cleared within 2 minutes."),
            ("Threads hanging indefinitely. Root cause: a downstream service call with no timeout configured.",
             "Added a 5-second timeout to the downstream API call and added a circuit breaker. Threads started releasing normally — pool no longer exhausted."),
            ("Background batch jobs sharing the same thread pool as API requests. Batch saturates the pool during off-peak hours.",
             "Implemented a separate thread pool for batch jobs using RabbitMQ async workers. API thread pool freed from batch workload entirely."),
            ("Single misbehaving client sending burst traffic that monopolises the thread pool for legitimate users.",
             "Added request rate limiting at the API gateway (100 req/s per client IP). Prevented any single client from exhausting the shared thread pool."),
            ("Thread pool too small for current user growth. Pool was sized 2 years ago for 10x fewer users.",
             "Right-sized the thread pool based on current traffic analysis: raised maxThreads to 400. Scheduled quarterly pool size reviews."),
        ],
    ),

    dict(
        category="Application", title="Service Fails to Start After Deployment",
        assets=["AppServer02", "AppServer07", "AppServer10"],
        rows=[
            ("Service exits immediately after deployment. Logs show 'environment variable JDBC_URL not found'.",
             "Added the missing JDBC_URL environment variable to the production systemd unit file EnvironmentFile. Service started successfully on next attempt."),
            ("Deployment introduced a new library requiring libssl 3.x but the server has libssl 1.1.x.",
             "Pinned the dependency to the version compatible with libssl 1.1.x and redeployed. Coordinated libssl upgrade with the infrastructure team."),
            ("Port 8080 already in use by a zombie process from the failed previous deployment.",
             "Killed the zombie process holding port 8080 using lsof and kill. Service started successfully on the cleared port."),
            ("YAML syntax error in application.yml introduced in the last commit — indentation off by 2 spaces.",
             "Corrected the YAML indentation error in application.yml. Validated with yamllint before redeploying. Service started cleanly."),
            ("New version incompatible with current database schema — missing migration not applied.",
             "Ran the missing Flyway migration script against the database. Re-deployed the application — service started and passed health checks."),
        ],
    ),

    # ── DATABASE ──────────────────────────────────────────────────────────────

    dict(
        category="Database", title="Database Connection Pool Exhausted",
        assets=["DBServer01", "DBServer02", "DBServer04"],
        rows=[
            ("Application reports 'too many connections'. PostgreSQL at its max_connections limit of 100.",
             "Increased HikariCP maximumPoolSize from 20 to 100 and restarted the application. Available connections immediately restored."),
            ("Batch job holding 47 connections open indefinitely due to missing connection.close() calls.",
             "Fixed the connection release logic in the batch job. Connections now released after each transaction. Pool cleared without restarting the app."),
            ("500 application threads each holding a dedicated DB connection. Connection count at maximum at peak hours.",
             "Deployed PgBouncer as a connection pooler in front of PostgreSQL. Reduced actual DB connections from 500 to 50 while serving the same load."),
            ("Analytics queries from BI tool holding connections for 20+ minutes blocking application connections.",
             "Set statement_timeout=60000ms on the analytics database role. Long queries now terminated automatically. Application connections freed."),
            ("Read queries saturating the primary DB connection pool. Write operations queuing behind read traffic.",
             "Added two read replicas and updated the application to route SELECT queries to replicas. Primary DB connection usage reduced by 60%."),
        ],
    ),

    dict(
        category="Database", title="Slow Query Performance",
        assets=["DBServer01", "DBServer03", "DBServer05"],
        rows=[
            ("Main dashboard query taking 28 seconds. Full sequential scan on a 50M-row table identified via EXPLAIN.",
             "Added a B-tree index on orders.created_at used in the main dashboard query. Execution time dropped from 28 s to 45 ms."),
            ("Reporting module timing out. Correlated subquery executing once per row — classic N+1 pattern.",
             "Rewrote the correlated subquery as a single JOIN. Eliminated the N+1 pattern. Page load reduced from 15 s to 800 ms."),
            ("Query performance degraded after events table grew past 500 million rows.",
             "Partitioned the events table by month using range partitioning. Queries on recent data now scan only the current month partition."),
            ("Same complex report query executed thousands of times daily with identical parameters — no caching.",
             "Enabled PostgreSQL query result caching for the top 10 most frequently executed identical queries. DB CPU load reduced 40%."),
            ("Query planner choosing a bad execution plan after a large data import changed the table statistics.",
             "Ran ANALYZE on the affected tables and increased default_statistics_target to 500 for high-cardinality columns. Planner now selects the optimal plan."),
        ],
    ),

    dict(
        category="Database", title="Database Deadlock",
        assets=["DBServer02", "DBServer04", "DBServer06"],
        rows=[
            ("Deadlock exceptions in application logs every few minutes. Transactions rolling back silently — users seeing random errors.",
             "Implemented exponential backoff retry logic for deadlock errors (up to 3 retries: 100 ms, 200 ms, 400 ms). Deadlock errors no longer visible to users."),
            ("Deadlock graph shows two transactions acquiring table A then B vs B then A — classic lock ordering issue.",
             "Standardised lock acquisition order across all code paths: always lock table A before B. Deadlocks in that path eliminated completely."),
            ("Large transactions holding locks for 10+ seconds giving wide window for deadlocks.",
             "Split large transactions into smaller units each completing in under 1 second. Deadlock window reduced significantly."),
            ("Job queue using table-level locks causing all workers to deadlock when processing concurrent items.",
             "Switched the job queue processor to SELECT FOR UPDATE SKIP LOCKED for row-level locking. Deadlocks in the queue eliminated entirely."),
            ("Deadlocks occurring under heavy load with no retry — every deadlock results in a user-facing error.",
             "Added Resilience4j automatic retry for deadlock exceptions with jitter. Deadlocks handled transparently without user impact."),
        ],
    ),

    dict(
        category="Database", title="Database Replication Lag",
        assets=["DBServer03", "DBServer05"],
        rows=[
            ("Replica falling behind primary by 5+ minutes. Bulk import on primary saturating the WAL stream.",
             "Moved large batch imports to off-peak hours (02:00-04:00). Replication lag recovered to under 1 second during business hours."),
            ("Replication restarting from scratch after every brief network interruption due to short WAL timeout.",
             "Increased wal_sender_timeout and wal_receiver_timeout on both primary and replica. Short network blips no longer cause full replication restarts."),
            ("Single-threaded apply on the replica cannot keep up with the write volume on the primary.",
             "Enabled parallel apply on the replica by setting max_parallel_workers_per_gather=4. Replication throughput improved 3x."),
            ("Replica running on HDD while primary is on SSD — I/O on replica is the bottleneck.",
             "Migrated the replica to SSD-backed storage. I/O wait on replica dropped from 40% to 2%. Lag eliminated within an hour."),
            ("Replica lag growing silently — no alerts configured for replication delay.",
             "Added a replication lag monitoring query via pg_stat_replication. Alerts configured for lag exceeding 30 seconds. Issue caught proactively going forward."),
        ],
    ),

    # ── NETWORK ───────────────────────────────────────────────────────────────

    dict(
        category="Network", title="VPN Connectivity Failure",
        assets=["NetGateway01", "NetGateway02", "NetGateway03",
                "NetGateway04", "NetGateway05", "NetGateway06"],
        rows=[
            ("VPN connects then drops within 5 minutes. Client logs show IPv6 routing conflict causing tunnel tear-down.",
             "Disabled IPv6 on the VPN client network adapter via Device Manager. VPN connection stable for 2+ hours after change."),
            ("VPN certificate expired 3 days ago. New connections refused — existing sessions were not affected until expiry.",
             "Renewed the VPN client certificate from the PKI CA. Updated the VPN client profile with the new certificate. Connections established successfully."),
            ("Remote users on one ISP cannot authenticate to VPN. ISP DNS interfering with VPN gateway resolution.",
             "Changed DNS settings to 8.8.8.8 and 1.1.1.1 on affected clients. VPN authentication resolved correctly after DNS change."),
            ("VPN profile corrupted after a Windows update overwrote the client configuration files.",
             "Rebuilt the VPN client configuration profile from the latest export from the VPN gateway admin console. Reimported profile — connection established."),
            ("VPN tunnel blocked at the user's home router. UDP 1194 traffic dropped by router firewall.",
             "Added port forwarding rule for UDP 1194 on the user's router. VPN tunnel established successfully after router change."),
            ("All remote users unable to connect since the VPN gateway IP address changed during a cloud migration.",
             "Updated the VPN gateway hostname in all client configurations to point to the new IP. Distributed updated profiles via MDM."),
        ],
    ),

    dict(
        category="Network", title="High Network Latency",
        assets=["NetSwitch01", "NetSwitch02", "CoreRouter01"],
        rows=[
            ("Latency between data centre and office elevated from 5 ms to 150 ms. Backup traffic saturating the WAN link.",
             "Applied QoS policy to prioritise application traffic over backup traffic on the core switch. Latency returned to normal within 5 minutes."),
            ("Backup process consuming 90% of WAN bandwidth during business hours causing application latency.",
             "Rescheduled backup jobs to run between 02:00-05:00 and rate-limited backup traffic to 50% of available bandwidth. Business-hours latency normalised."),
            ("Packet loss detected between two data centre switches. NIC driver issue causing intermittent drops.",
             "Updated the NIC driver from v3.1 to v4.2 and disabled GRO and GSO offloading. Packet loss eliminated and latency dropped to normal."),
            ("Spanning tree misconfiguration causing packets to travel a 3-hop path instead of 1 hop.",
             "Fixed STP root bridge election on the core switch. Traffic now taking the direct 1-hop path. Latency reduced from 150 ms to 6 ms."),
            ("High latency on specific VLAN only. VLAN trunk port incorrectly configured with wrong MTU.",
             "Corrected the MTU setting on the VLAN trunk port from 1500 to 9000 (jumbo frames). Latency on that VLAN normalised immediately."),
        ],
    ),

    dict(
        category="Network", title="DNS Resolution Failure",
        assets=["DNSServer01", "DNSServer02", "AppServer04"],
        rows=[
            ("Internal hostnames not resolving. DNS queries timing out after 5 seconds. Issue appeared after a server reboot.",
             "Flushed the DNS cache on affected clients using ipconfig /flushdns and systemd-resolve --flush-caches. DNS resolution restored immediately."),
            ("bind9 DNS service consumed excess memory and stopped responding. Service was still 'running' per systemd.",
             "Restarted the bind9 DNS service. Added a memory limit to the systemd unit file to prevent recurrence. DNS queries normal after restart."),
            ("DNS record pointing to a decommissioned server IP not updated after the server was retired.",
             "Updated the stale A record to the correct IP address and reduced TTL to 60 seconds for faster propagation. Applications resolved correctly."),
            ("ISP-provided DNS servers unreliable — intermittent resolution failures especially for external hostnames.",
             "Switched DNS forwarder to 8.8.8.8 and 1.1.1.1. Added redundant DNS entries for all critical internal services. Resolution failures eliminated."),
            ("New subdomain created in the application but DNS record not added to the zone file.",
             "Added the missing A record for the new subdomain to the internal DNS zone. Propagated to all DNS servers. Resolution working within 2 minutes."),
        ],
    ),

    dict(
        category="Network", title="Firewall Blocking Application Traffic",
        assets=["Firewall01", "Firewall02", "Firewall03"],
        rows=[
            ("New app-to-database traffic blocked after firewall policy audit removed broad allow rules.",
             "Added specific allow rule for TCP 5432 between the app server subnet and database subnet. Traffic resumed immediately after rule applied."),
            ("Application unable to reach external payment API. Outbound HTTPS to payment provider IP range blocked.",
             "Whitelisted the payment provider's IP ranges in the outbound firewall policy. Payment API calls succeeding after rule propagation."),
            ("Firewall logs showing blocked traffic to three additional ports not covered by current allow rules.",
             "Enabled temporary firewall logging to identify all blocked traffic patterns. Added rules for all legitimate ports identified. Disabled verbose logging after."),
            ("Security team applied a new egress policy that blocked all outbound traffic by default.",
             "Reviewed the new default-deny egress policy. Created an application-tier zone with least-privilege outbound rules for required destinations only."),
            ("After a cloud migration, security groups were not updated — old IP ranges still in allow rules, new IPs blocked.",
             "Updated all security group rules to reference the new IP ranges from the cloud migration. Validated connectivity from each application tier."),
        ],
    ),

    # ── SECURITY ──────────────────────────────────────────────────────────────

    dict(
        category="Security", title="Account Lockout After Failed Authentication",
        assets=["AuthServer01", "ADServer01", "ADServer02"],
        rows=[
            ("Multiple user accounts locked out simultaneously. A scheduled task using an old password triggered the lockouts.",
             "Updated the service account password in the scheduled task and unlocked all affected accounts in Active Directory."),
            ("Service account password rotated but not updated in the 12 systems that use it. Mass lockouts followed.",
             "Updated the service account password in all 12 dependent systems via the secrets management vault. Enabled automated credential rotation."),
            ("Account lockout policy too strict — 3 failed attempts locks the account, causing frequent false lockouts.",
             "Raised the lockout threshold from 3 to 10 failed attempts and extended the observation window to 30 minutes. False lockout rate dropped to near zero."),
            ("Monitoring tool using a shared credential for health checks. Credential expired and flooded auth logs.",
             "Replaced the shared credential in the monitoring tool with a dedicated service account with a non-expiring password. Created alert for credential expiry."),
            ("After an AD password policy change, users with saved passwords in browsers triggered lockouts on cached credentials.",
             "Forced a password reset for affected users and issued a communication advising users to update saved credentials in all browsers and apps."),
        ],
    ),

    dict(
        category="Security", title="SSL/TLS Certificate Expiry",
        assets=["WebServer01", "AppServer01", "LoadBalancer01"],
        rows=[
            ("SSL certificate expired. Browser showing 'Your connection is not private'. HTTPS connections failing.",
             "Renewed the SSL certificate from the CA. Uploaded new cert and private key to the load balancer. HTTPS traffic restored with 1-year validity."),
            ("Certificate expiry not monitored. Detected only when users started reporting browser warnings.",
             "Configured Certbot with Let's Encrypt for automatic 90-day renewal. Set up monitoring alerts 30 and 7 days before expiry."),
            ("Internal microservice using a self-signed certificate that expired. Service-to-service communication broken.",
             "Replaced the self-signed certificate with one issued by the internal CA. Configured auto-renewal via the internal PKI ACME endpoint."),
            ("AWS ACM certificate auto-renewal failed silently due to a DNS validation record being removed.",
             "Re-added the ACM DNS validation CNAME record and triggered manual renewal in the AWS console. Certificate renewed and auto-attached to the ALB."),
            ("Certificate chain incomplete — intermediate certificate missing. Mobile clients rejecting the certificate.",
             "Downloaded and installed the full certificate chain including the intermediate CA certificate. Verified with SSL Labs — no warnings on any client type."),
        ],
    ),

    dict(
        category="Security", title="Brute Force / Unauthorised Access Attempt",
        assets=["WebServer02", "AppServer03", "Firewall01"],
        rows=[
            ("High volume of failed SSH login attempts from multiple international IP ranges. Brute force pattern in logs.",
             "Blocked the attacking IP ranges with iptables and updated the WAF blocklist. Enabled geographic IP filtering for high-risk countries."),
            ("Login endpoint receiving 10,000 attempts per minute. No rate limiting in place.",
             "Implemented fail2ban — accounts blocked for 1 hour after 5 failed attempts. Installed on all public-facing servers. Attack traffic reduced 95%."),
            ("SSH accessible on port 22. Automated scanners probing continuously even with strong passwords.",
             "Disabled SSH password authentication. Enforced key-based login only. Changed SSH default port from 22 to a non-standard port."),
            ("Even with correct credentials, accounts compromised without second factor — password reuse from a breach.",
             "Enabled mandatory MFA for all external-facing login endpoints. Compromised passwords alone no longer sufficient to gain access."),
            ("No audit trail — attack discovered days later only because of performance degradation from unauthorised resource usage.",
             "Enabled detailed authentication audit logging and forwarded to the SIEM. Real-time alerts configured for repeated failures. Created incident response playbook."),
        ],
    ),

    # ── PERFORMANCE ───────────────────────────────────────────────────────────

    dict(
        category="Performance", title="High CPU Utilization",
        assets=["AppServer02", "AppServer05", "MediaServer04",
                "AppServer08", "DBServer01"],
        rows=[
            ("Transcoding job consuming all 8 CPU cores with no limits. Other services on the same host starved.",
             "Applied cgroups CPU quota limiting each transcoding job to 4 cores maximum. Server CPU usage normalised within 2 minutes."),
            ("Profiling revealed a cryptographic operation in a tight loop due to an infinite retry bug in error handling.",
             "Fixed the infinite retry loop by adding exponential backoff with a maximum of 5 retries. CPU usage dropped from 98% to 15%."),
            ("CPU spike caused by a logging library writing synchronously in a hot code path — every request blocks on log write.",
             "Changed the logging library configuration from synchronous to async mode. CPU usage normalised immediately. Request throughput improved 3x."),
            ("Application doing expensive computation on every request with no caching — same results computed thousands of times.",
             "Enabled Redis result caching for the expensive computation with a 5-minute TTL. CPU load dropped 60% as repeated computations served from cache."),
            ("Database CPU saturated by thousands of identical queries per second — application not caching query results.",
             "Added a query result cache layer with a 60-second TTL for the top 5 most-called read queries. Database CPU dropped from 95% to 35%."),
        ],
    ),

    dict(
        category="Performance", title="High Memory Utilization / OOM Kills",
        assets=["DBServer01", "AppServer04", "AppServer09"],
        rows=[
            ("Application caching all query results in-process with no eviction policy. Memory grows unbounded.",
             "Replaced in-process unbounded cache with an external Redis cache limited to 512 MB. Application memory dropped from 28 GB to 6 GB."),
            ("JVM running without -Xmx. Heap growing to 28 GB on a 32 GB host, leaving no memory for the OS.",
             "Set -Xmx12g to cap the JVM heap. Left adequate memory for OS and other services. Memory pressure eliminated after restart."),
            ("Server physically undersized for current workload — memory added to the procurement queue months ago.",
             "Added 32 GB of RAM to the server during an emergency maintenance window. Updated capacity alerts to trigger at 70% for future proactive action."),
            ("PostgreSQL shared_buffers set to 128 MB (default) on a 64 GB host — database using OS disk cache causing memory pressure.",
             "Increased shared_buffers to 16 GB and work_mem to 64 MB. Database now uses its own memory efficiently. OOM kills stopped."),
            ("Memory-mapped file handling causing OS to hold stale file pages. vm.swappiness too high — swapping rather than freeing cache.",
             "Reduced vm.swappiness from 60 to 10 and enabled huge pages. System now prefers releasing file cache over swapping. Memory pressure resolved."),
        ],
    ),

    dict(
        category="Performance", title="API Response Timeout",
        assets=["AppServer03", "AppServer06", "LoadBalancer02"],
        rows=[
            ("API endpoint timing out at 30 seconds. Root cause: N+1 query — 150 individual DB queries per request.",
             "Rewrote the endpoint as a single JOIN query replacing 150 individual calls. Response time improved from 28 s to 120 ms."),
            ("Identical API call made thousands of times daily with no caching. Each call re-fetches the same unchanged data.",
             "Added Redis response caching for the top-5 read endpoints with a 60-second TTL. Average response time dropped from 28 s to 15 ms."),
            ("API calls to a slow downstream service blocking — no timeout configured. Threads hang for 60+ seconds.",
             "Added a 5-second timeout and circuit breaker on the downstream service call. Slow calls now return a cached fallback immediately."),
            ("Heavy PDF report generation in the synchronous request path causing 504 errors on the load balancer.",
             "Moved PDF generation to an async background worker. API returns a job ID immediately. Client polls for completion. No more 504 timeouts."),
            ("Load balancer timeout set lower than the application processing time for valid complex requests.",
             "Increased the load balancer idle timeout from 30 s to 120 s to match the longest legitimate application processing time."),
        ],
    ),

    # ── HARDWARE ──────────────────────────────────────────────────────────────

    dict(
        category="Hardware", title="Server Overheating",
        assets=["PhysicalServer01", "PhysicalServer02", "PhysicalServer03"],
        rows=[
            ("IPMI alerting CPU temperature at 92°C. Thermal throttling active. Performance degraded 50%.",
             "Cleaned dust from all air filters, heat sinks and fan blades with compressed air. CPU temperature dropped to 65°C within 10 minutes."),
            ("One of three cooling fans failed. Airflow reduced by 33%. Temperatures rising steadily toward shutdown threshold.",
             "Replaced the failed cooling fan module (hot-swappable component). Full airflow restored — temperatures dropped to 58°C."),
            ("Rack ambient temperature rising due to a failed data centre CRAC unit affecting the whole row.",
             "Live-migrated VMs to servers in a cooler rack row. Added supplemental portable cooling unit while the CRAC unit was repaired."),
            ("Server room ambient temperature crept up after AC setpoint was accidentally changed during maintenance.",
             "Corrected the AC setpoint from 24°C back to 18°C. Temperatures normalised across all servers within 30 minutes."),
            ("CPU thermal paste dried out on a 5-year-old server. Thermal interface between CPU and heatsink degraded.",
             "Reapplied high-quality thermal compound to all CPUs during a maintenance window. CPU temperatures reduced by 15°C under load."),
        ],
    ),

    dict(
        category="Hardware", title="Network Interface Card Failure",
        assets=["PhysicalServer03", "PhysicalServer04"],
        rows=[
            ("Primary NIC showing as down in the OS. Physical link light not illuminated. Server unreachable on primary IP.",
             "Updated the NIC firmware and driver to the latest vendor release. Interface came up after driver reload without requiring a full reboot."),
            ("NIC partially failed — link up but packet loss at 30%. Performance severely degraded on the affected server.",
             "Failed over to the bonded secondary NIC interface that was in standby. Traffic resumed at full speed while primary NIC was replaced."),
            ("NIC card worked loose in the PCIe slot during vibration from a nearby UPS replacement.",
             "Reseated the NIC card firmly in the PCIe slot and locked the retention bracket. Interface came up after reboot."),
            ("SFP transceiver module failed — fibre link went dark without warning.",
             "Replaced the failed SFP transceiver with a spare from inventory. Network link came up immediately after transceiver swap."),
            ("Duplex mismatch between the server NIC and the switch port causing persistent packet collisions.",
             "Set both the server NIC and switch port to explicit 10G full-duplex instead of auto-negotiation. Packet collisions eliminated."),
        ],
    ),

    # ── AUTHENTICATION ────────────────────────────────────────────────────────

    dict(
        category="Authentication", title="Active Directory Sync Failure",
        assets=["ADServer01", "ADServer02"],
        rows=[
            ("Azure AD Connect stopped syncing. New users created on-premises not appearing in Microsoft 365.",
             "Restarted the Azure AD Connect sync service and forced a delta sync cycle via PowerShell. Changes propagated within 5 minutes."),
            ("AD Connect service account password expired. Sync halted without raising a clear error alert.",
             "Reset the expired service account password in AD and updated it in the Azure AD Connect configuration wizard. Sync resumed."),
            ("Sync error queue showing duplicate UserPrincipalName conflict. Multiple users with the same UPN blocking sync.",
             "Identified and corrected duplicate UPN values in on-premises AD. Cleared the sync error queue. All objects syncing cleanly."),
            ("AD Connect server ran out of disk space — sync transaction logs consumed the entire drive.",
             "Freed disk space by clearing old sync logs and compressing the transaction log database. Re-enabled the sync service."),
            ("Sync stopped working after a domain controller was decommissioned that AD Connect was targeting.",
             "Updated the AD Connect domain controller preference to point to an active DC. Full sync triggered manually. All objects synchronised."),
        ],
    ),

    dict(
        category="Authentication", title="LDAP Authentication Not Working",
        assets=["LDAPServer01", "AppServer05"],
        rows=[
            ("All LDAP-authenticated applications failing. Port 389 not responding. slapd process had exited silently.",
             "Restarted the slapd LDAP service and verified recovery with ldapsearch. All authentication-dependent applications recovered automatically."),
            ("LDAP reachable but bind operations failing. Service account password rotated without updating app configs.",
             "Updated the LDAP bind DN credentials in all application configurations. Applications reconnected to LDAP successfully."),
            ("LDAP over TLS (LDAPS port 636) failing after certificate expiry. Applications using secure LDAP broken.",
             "Renewed the LDAP service TLS certificate and deployed it to the LDAP server. LDAPS connections restored."),
            ("Firewall rule change blocked port 389 from the application subnet to the LDAP server.",
             "Added a firewall exception allowing TCP 389 from the application server subnet to the LDAP server. LDAP connectivity restored."),
            ("LDAP server returning referrals to a secondary DC that is down — clients unable to follow the referral.",
             "Disabled LDAP referrals in the application's LDAP configuration. Applications now connect directly without following referrals."),
        ],
    ),

    dict(
        category="Authentication", title="Multi-Factor Authentication Failure",
        assets=["MFAServer01", "AuthServer02"],
        rows=[
            ("All TOTP codes being rejected. MFA server system clock drifted 4 minutes — TOTP validation window exceeded.",
             "Synchronised the MFA server clock via ntpdate pool.ntp.org. TOTP codes immediately valid after time correction."),
            ("MFA push notifications not arriving on iOS devices. APNS certificate for push delivery had expired.",
             "Renewed the Apple Push Notification Service certificate and restarted the push notification service. iOS push notifications resuming within minutes."),
            ("Users changed phones but MFA is bound to the old device TOTP secret — cannot authenticate.",
             "Re-enrolled affected users in MFA by generating new QR codes from the identity portal for them to scan on their new devices."),
            ("MFA server unreachable due to a network partition. All MFA-protected resources inaccessible.",
             "Enabled temporary email OTP as a backup MFA method while the MFA server network issue was resolved. Users continued working without interruption."),
            ("Specific users' authenticator apps generating wrong codes due to their phone clock being out of sync.",
             "Instructed affected users to enable automatic time sync on their mobile devices. TOTP codes immediately valid after phone clock corrected."),
        ],
    ),

    # ── MONITORING ────────────────────────────────────────────────────────────

    dict(
        category="Monitoring", title="Monitoring Agent Not Reporting",
        assets=["MonitoringServer01", "AppServer01", "AppServer06"],
        rows=[
            ("Server missing from Grafana dashboard for 2 hours. No metrics, no alerts generated. Agent process not running.",
             "Restarted the Prometheus node_exporter service. Metrics resumed reporting to Grafana within 30 seconds."),
            ("Monitoring agent binary corrupted during a failed package update. Service fails to start.",
             "Downloaded the correct version of the monitoring agent, verified the SHA256 checksum, and reinstalled. Metrics reporting resumed."),
            ("Agent API key rotated by the security team. Agent unable to authenticate to the monitoring backend.",
             "Updated the monitoring agent configuration with the new API key and restarted the agent service. Reporting resumed immediately."),
            ("Firewall change blocked port 9100 from the Prometheus server to the monitored host.",
             "Added a firewall exception allowing TCP 9100 from the Prometheus scrape server. Metrics collection resumed within 60 seconds."),
            ("Monitoring agent running but metric endpoint returning 503 due to insufficient file descriptor limits.",
             "Increased the system file descriptor limit for the monitoring agent process via systemd LimitNOFILE=65536. Endpoint started responding correctly."),
        ],
    ),

    dict(
        category="Monitoring", title="Alert Storm False Positives",
        assets=["MonitoringServer01", "MonitoringServer02"],
        rows=[
            ("Hundreds of alerts per minute. CPU threshold set too low — triggers on any usage above 70% even briefly.",
             "Raised CPU alert threshold to 85% and added a minimum sustained duration of 10 minutes before the alert fires. Alert volume reduced 80%."),
            ("Duplicate alerts from multiple monitoring systems covering the same hosts — paging the team multiple times per event.",
             "Implemented alert grouping and deduplication in Alertmanager. Related alerts from the same host now grouped into a single notification."),
            ("Alert storm caused by a misconfigured health check marking healthy instances as down.",
             "Fixed the health check endpoint URL that had changed after a routing update. Underlying issue resolved — alert storm stopped immediately."),
            ("On-call team receiving alerts during scheduled maintenance windows when downtime is expected.",
             "Added maintenance window functionality in the monitoring system. Alerts suppressed during scheduled maintenance. Team no longer paged for planned work."),
            ("Single server failure generating 50 individual service alerts instead of one host-down alert.",
             "Implemented alert correlation rules: if a host is down suppress all individual service alerts from that host. Alert count reduced from 50 to 1 per server failure."),
        ],
    ),

    # ── CONFIGURATION ─────────────────────────────────────────────────────────

    dict(
        category="Configuration", title="Application Misconfiguration After Deployment",
        assets=["AppServer02", "AppServer04", "AppServer07"],
        rows=[
            ("Production deployment accidentally used the staging database connection string. Real transactions going to test DB.",
             "Corrected the CI/CD pipeline environment mapping that was injecting staging variables into the production deployment. Redeployed with correct config."),
            ("Feature flags set to 'beta' values in production — new unfinished features exposed to all users.",
             "Rolled back to the previous release tag within 5 minutes of detecting the misconfiguration. No data loss. Documented root cause."),
            ("Sensitive configuration committed to the application codebase and deployed — visible in the container image.",
             "Migrated all sensitive configuration to HashiCorp Vault. Environment-specific values now injected at runtime from the vault, not at build time."),
            ("Production configuration not validated before deployment — invalid YAML syntax discovered only at startup.",
             "Added a pre-deployment config validation step to the CI/CD pipeline that fails the build on invalid YAML. Prevents bad configs reaching production."),
            ("New deployment used an old config file from a previous environment that was left in the repository.",
             "Cleaned up all old environment config files from the repository. Enforced environment-specific configs via CI/CD environment variables only."),
        ],
    ),

    dict(
        category="Configuration", title="Load Balancer Misconfiguration",
        assets=["LoadBalancer01", "LoadBalancer02"],
        rows=[
            ("Load balancer health checks failing after application changed port from 80 to 8080 in the last deployment.",
             "Updated the health check probe port from 80 to 8080 in the load balancer configuration. All backend instances marked healthy within 30 seconds."),
            ("Health check HTTP path returning 404 after the application routing was refactored.",
             "Updated the load balancer health check path from /ping to /api/v2/health to match the new routing. Health checks passing."),
            ("SSL termination misconfigured — health probes sent over HTTPS to a backend that only accepts HTTP.",
             "Reconfigured load balancer health checks to use HTTP on the backend port instead of HTTPS. All backends now reporting healthy."),
            ("Sticky sessions causing all traffic to route to one instance after another instance was replaced with a new deployment.",
             "Disabled sticky sessions and cleared the session persistence table. Traffic distributing correctly across all instances via round-robin."),
            ("Load balancer connection timeout too short — valid long-running file uploads failing at the LB before the backend finishes.",
             "Increased the load balancer connection idle timeout from 30 s to 300 s to accommodate large file uploads. Upload failures stopped."),
        ],
    ),

    # ── STREAMING (added to give live-stream incidents real KB depth) ─────────
    dict(
        category="Network", title="Streaming Service Failure",
        assets=["MediaGateway01", "MediaServer01", "NetGateway01",
                "MediaServer03", "MediaGateway02"],
        rows=[
            ("Live broadcast stream froze for all viewers. The streaming service process is alive but no longer accepting new RTMP connections.",
             "Restarted the streaming service and reset the network interface on the media gateway. RTMP connections were accepted immediately and the live stream resumed."),
            ("Viewers report constant buffering and dropouts every few minutes. Encoder bitrate is exceeding the available uplink bandwidth.",
             "Lowered the adaptive bitrate ceiling and enabled adaptive bitrate streaming. Buffering was eliminated for clients on constrained connections."),
            ("Stream stopped responding after a CDN edge-node failover. HLS playlist requests are returning 404 to remote viewers.",
             "Purged the stale HLS playlist from the CDN and re-pointed the origin pull to a healthy edge node. Playback for remote viewers was restored within minutes."),
            ("Live stream dropped repeatedly during a peak event. The RTMP ingest server hit its concurrent-connection limit.",
             "Raised the RTMP ingest connection limit and added a second ingest node behind the load balancer. Stream stability was restored under peak viewer load."),
            ("Audio and video drift further out of sync the longer a broadcast runs. The streaming server clock has drifted from the encoder.",
             "Synchronised the streaming server clock via NTP and restarted the transcoding pipeline. Audio/video sync held steady across multi-hour broadcasts."),
        ],
    ),

    # ── ENCODER / TRANSCODING (added to give encoder incidents real depth) ────
    dict(
        category="Application", title="Encoder Service Failure",
        assets=["MediaServer02", "AppServer05", "MediaServer04",
                "MediaServer06", "AppServer08"],
        rows=[
            ("Encoder service crashed mid-job with a codec assertion error. The transcoding queue has stalled and jobs are backing up.",
             "Restarted the encoder service and updated the codec library to the patched release. The transcoding queue drained normally."),
            ("Transcoding jobs are failing on H.265 input after a codec library was downgraded during a rollback.",
             "Restored the correct codec library version and restarted the encoder workers. H.265 transcoding jobs completed successfully."),
            ("The encoder is repeatedly OOM-killed on 4K source files. The worker memory limit is too low for the codec's buffers.",
             "Raised the encoder worker memory limit and updated the codec library to the streaming-optimised build. 4K jobs no longer trigger OOM kills."),
            ("A video conversion job hangs at 0% after a GPU driver update broke hardware-accelerated encoding.",
             "Rolled back the GPU driver and restarted the encoder service with hardware acceleration re-enabled. Conversion throughput was restored."),
            ("The encoder service fails to start after a deployment — a required codec shared library is missing on the host.",
             "Installed the missing codec shared-library dependency and restarted the encoder service. The service started and passed health checks."),
        ],
    ),

]


def build_rows(existing_df: pd.DataFrame) -> list[dict]:
    # Next available INC/TKT numbers after the original rows
    def to_num(val: str, prefix: str) -> int:
        try:
            return int(str(val).replace(prefix, "").replace("-", ""))
        except ValueError:
            return 0

    next_inc = max(
        (to_num(v, "INC-") for v in existing_df["Incident ID"]), default=5000
    ) + 1
    next_tkt = max(
        (to_num(v, "TKT-") for v in existing_df["Ticket ID"]), default=1000
    ) + 1

    import itertools
    rows = []
    for t in TYPES:
        asset_cycle = itertools.cycle(t["assets"])
        for desc, solution in t["rows"]:
            opened_at, resolved_at, res_hours = _make_timestamps(t["category"])
            rows.append({
                "Media Asset":       next(asset_cycle),
                "Category":          t["category"],
                "Ticket ID":         f"TKT-{next_tkt}",
                "Incident ID":       f"INC-{next_inc}",
                "Incident Details":  t["title"],
                "Description":       desc,
                "Solution":          solution,
                "Opened At":         opened_at,
                "Resolved At":       resolved_at,
                "Resolution Hours":  res_hours,
            })
            next_inc += 1
            next_tkt += 1
    return rows


def main():
    print(f"Reading {XLSX} ...")
    df_all = pd.read_excel(XLSX, engine="openpyxl")
    print(f"  Total rows in file : {len(df_all)}")

    # Keep only the original 150 rows — strip any previously appended rows
    df_original = df_all.iloc[:ORIGINAL_ROWS].copy()
    print(f"  Keeping original   : {len(df_original)} rows")

    # Stamp original rows with timestamps if missing
    if "Opened At" not in df_original.columns:
        df_original["Opened At"]        = ""
        df_original["Resolved At"]      = ""
        df_original["Resolution Hours"] = 0.0

    for idx in df_original.index:
        if not df_original.at[idx, "Opened At"]:
            cat = str(df_original.at[idx, "Category"])
            o, r, h = _make_timestamps(cat)
            df_original.at[idx, "Opened At"]        = o
            df_original.at[idx, "Resolved At"]      = r
            df_original.at[idx, "Resolution Hours"] = h

    new_rows = build_rows(df_original)
    df_new   = pd.DataFrame(new_rows, columns=[
        "Media Asset", "Category", "Ticket ID",
        "Incident ID", "Incident Details", "Description", "Solution",
        "Opened At", "Resolved At", "Resolution Hours",
    ])
    print(f"  New rows added     : {len(df_new)}")

    df_combined = pd.concat([df_original, df_new], ignore_index=True)
    print(f"  Total after merge  : {len(df_combined)}")

    df_combined.to_excel(XLSX, index=False, engine="openpyxl")
    print(f"\nSaved to {XLSX}")
    print(f"Categories     : {sorted(df_combined['Category'].unique())}")
    print(f"Problem types  : {df_combined['Incident Details'].nunique()} unique")
    print(f"Has timestamps : {df_combined['Opened At'].notna().all()}")
    print(f"Avg resolution : {df_combined['Resolution Hours'].mean():.1f} hrs")


if __name__ == "__main__":
    main()
