# Kubernetes container-escape and lateral-movement agent.
#
#   uv run python scratch/k8s/kube_agent.py --smoke
#   uv run python scratch/k8s/kube_agent.py --scope 10.244.0.0/16,10.96.0.0/12 --out cluster.md
#
# Run this INSIDE a pod in the cluster you are assessing. It enumerates its own
# position, maps what it can reach, and works out escalation paths to other
# pods, the node, and the control plane — proving each one non-destructively.

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from nooa import Agent, Context, strategy
from nooa.agentdoc import spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig, TokenBudgetConfig
from nooa.strategies import CodeActStrategy
from nooa.tools import ShellTools, SkillWriting, TodoManager
from nooa.unifiedllm import get_llm_client

HERE = Path(__file__).parent
LIBS = HERE / "libs"
WORK = HERE / "work"

SA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount"

HOST_ACCESS_PREFIXES = ("/", "/etc", "/proc", "/dev", "/var/log", "/var/lib/kubelet", "/root", "/home")


def _has_cap(position: "Position", cap: str) -> bool:
    return any(cap.lower() in c.lower() for c in position.effective_caps)


def _has_socket(position: "Position", needle: str) -> bool:
    return any(needle in s.lower() for s in position.sensitive_sockets)


def _has_host_mount(position: "Position") -> bool:
    """True only for a mount that genuinely exposes the host filesystem."""
    for mount in position.host_mounts:
        text = mount.lower()
        if "kubelet/pods/" in text or "emptydir" in text or "pod-specific" in text:
            continue  # pod-scoped kubelet volume, not host access
        if any(text.startswith(p) or f" {p}" in text for p in HOST_ACCESS_PREFIXES):
            return True
    return False


# Hard prerequisites per technique, checked against the ESTABLISHED Position.
# A model claiming a technique whose precondition is provably absent is
# fabricating, and we reject it in Python rather than spending a turn on it.
TECHNIQUE_PREREQS: dict[str, tuple[str, Any]] = {
    "privileged": ("privileged: true on the container", lambda p: p.privileged),
    "docker-sock": ("a mounted docker socket", lambda p: _has_socket(p, "docker")),
    "containerd-sock": ("a mounted containerd socket", lambda p: _has_socket(p, "containerd")),
    "crio-sock": ("a mounted cri-o socket", lambda p: _has_socket(p, "crio")),
    "runtime-socket": ("a mounted container runtime socket", lambda p: bool(p.sensitive_sockets)),
    "hostpid": ("hostPID: true", lambda p: p.host_pid),
    "nsenter": ("hostPID: true", lambda p: p.host_pid),
    "hostnetwork": ("hostNetwork: true", lambda p: p.host_network),
    "hostipc": ("hostIPC: true", lambda p: p.host_ipc),
    "hostpath": ("a hostPath mount granting host filesystem access", _has_host_mount),
    "release-agent": ("CAP_SYS_ADMIN", lambda p: _has_cap(p, "sys_admin")),
    "release_agent": ("CAP_SYS_ADMIN", lambda p: _has_cap(p, "sys_admin")),
    "core-pattern": ("CAP_SYS_ADMIN", lambda p: _has_cap(p, "sys_admin")),
    "core_pattern": ("CAP_SYS_ADMIN", lambda p: _has_cap(p, "sys_admin")),
    "kernel-module": ("CAP_SYS_MODULE", lambda p: _has_cap(p, "sys_module")),
    "sys-module": ("CAP_SYS_MODULE", lambda p: _has_cap(p, "sys_module")),
    "open_by_handle": ("CAP_DAC_READ_SEARCH", lambda p: _has_cap(p, "dac_read_search")),
    "shocker": ("CAP_DAC_READ_SEARCH", lambda p: _has_cap(p, "dac_read_search")),
    "ptrace": ("CAP_SYS_PTRACE", lambda p: _has_cap(p, "sys_ptrace")),
    "device-access": ("a host device mount under /dev", lambda p: _has_host_mount(p)),
}


# A path claiming cluster-admin must rest on a permission the RBAC review found.
RBAC_PREREQS: dict[str, tuple[str, Any]] = {
    "escalate": ("the escalate or bind verb", lambda r: r.can_escalate_or_bind),
    "bind": ("the escalate or bind verb", lambda r: r.can_escalate_or_bind),
    "impersonat": ("the impersonate verb", lambda r: r.can_impersonate),
    "secrets": ("get on secrets", lambda r: r.can_get_secrets),
    "create-pod": ("create on pods", lambda r: r.can_create_pods),
    "pod-exec": ("create on pods/exec", lambda r: r.can_exec_pods),
    "sa-token": ("create on serviceaccounts/token", lambda r: r.can_create_sa_tokens),
}


def feasible(
    path: "EscalationPath",
    position: "Position | None",
    rbac: "RbacProfile | None" = None,
) -> tuple[bool, str]:
    """Check a claimed path against established facts. Returns (ok, reason).

    Deterministic and free. Catches the common failure where a model enumerates
    textbook escapes and asserts them regardless of whether the preconditions
    exist in this container or this ServiceAccount.
    """
    technique = path.technique.lower().replace(" ", "-").replace("_", "-")

    if position is not None:
        for marker, (requirement, predicate) in TECHNIQUE_PREREQS.items():
            if marker.replace("_", "-") in technique:
                if not predicate(position):
                    return False, (
                        f"technique {path.technique!r} requires {requirement}, which this "
                        f"container does not have (caps={position.effective_caps or 'none'}, "
                        f"privileged={position.privileged}, hostPID={position.host_pid}, "
                        f"hostNetwork={position.host_network}, "
                        f"sockets={position.sensitive_sockets or 'none'})"
                    )
                return True, ""

    # A technique that names an RBAC mechanism must rest on a permission the
    # review actually found — whatever it claims to reach.
    if rbac is not None:
        for marker, (requirement, predicate) in RBAC_PREREQS.items():
            if marker in technique and not predicate(rbac):
                return False, (
                    f"technique {path.technique!r} depends on {requirement}, which the "
                    f"RBAC review did not find for this ServiceAccount"
                )

    # Beyond that, any claim of cluster-admin needs at least one escalation verb.
    if rbac is not None and path.to_position == "cluster-admin":
        granted = any(
            [rbac.can_escalate_or_bind, rbac.can_impersonate, rbac.can_create_pods,
             rbac.can_exec_pods, rbac.can_get_secrets, rbac.can_create_sa_tokens]
        )
        if not granted:
            return False, (
                "claims cluster-admin, but the RBAC review found none of the verbs that "
                "enable escalation (escalate/bind, impersonate, create pods, pods/exec, "
                "get secrets, serviceaccounts/token)"
            )
    return True, ""


# Control-plane and node ports worth sweeping, and what they usually mean.
K8S_PORTS = (443, 2379, 2380, 6443, 8080, 10250, 10255, 10256, 10257, 10259)
K8S_PORT_NAMES = {
    443: "apiserver-or-https",
    2379: "etcd-client",
    2380: "etcd-peer",
    6443: "apiserver",
    8080: "apiserver-insecure-or-app",
    10250: "kubelet-rw",
    10255: "kubelet-readonly",
    10256: "kube-proxy-health",
    10257: "controller-manager",
    10259: "scheduler",
}

# --------------------------------------------------------------------------
# Self-contained on purpose: this file gets copied into a pod, so it must not
# import from sibling scratch directories. Scope and make_llm are duplicated
# from researcher.py / exploit_agent.py rather than imported.
# --------------------------------------------------------------------------

GATEWAY_BASE = "https://inference-api.nvidia.com/v1"
GATEWAY_MODEL = "azure/deepseek-ai/deepseek-v4-pro"


def make_llm(model: str | None = None):
    """Build an LLM client, preferring the NVIDIA-internal inference gateway."""
    gateway_key = (
        os.getenv("NVIDIA_INFERENCE_API_KEY")
        or os.getenv("NVIDIA_INTERNAL_API_KEY")
        or os.getenv("INFERENCE_API_KEY")
    )
    name = model or (GATEWAY_MODEL if gateway_key else None)

    if name and name.startswith(("azure/", "nvidia/", "openai/azure/", "openai/nvidia/")):
        if not gateway_key:
            raise SystemExit("Gateway model requested but no key set. export NVIDIA_INFERENCE_API_KEY=...")
        if not name.startswith("openai/"):
            name = f"openai/{name}"
        client = get_llm_client(name, api_base=GATEWAY_BASE, api_key=gateway_key)
        # Non-Anthropic backends 400 on message-level cache_control markers.
        if not any(tag in name for tag in ("claude", "anthropic")):
            client.cache_control_injection_points = []
        return client

    if name:
        return get_llm_client(name)
    if os.getenv("ANTHROPIC_API_KEY"):
        return get_llm_client("claude-sonnet-5")
    if os.getenv("OPENAI_API_KEY"):
        return get_llm_client("gpt-5-mini")
    raise SystemExit(
        "No provider key found. Set NVIDIA_INFERENCE_API_KEY, ANTHROPIC_API_KEY, "
        "or OPENAI_API_KEY — or pass --model."
    )


class Scope:
    """The authorized network range. Hosts, IPs, and CIDRs.

    Enforced for every helper the agent is given. Honest limit: the agent also
    has a shell, and a Python guard cannot stop a shell command that names an
    out-of-scope host. Guardrail, not containment boundary.
    """

    def __init__(self, entries: tuple[str, ...]):
        self.raw = entries
        self.networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self.names: list[str] = []
        for entry in entries:
            try:
                self.networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                name = entry.lower().lstrip("*.")
                for prefix in ("http://", "https://"):
                    if name.startswith(prefix):
                        name = name[len(prefix) :]
                self.names.append(name.split("/")[0].split(":")[0])

    def allows(self, target: str) -> bool:
        """True if `target` (host, IP, or URL host) is in scope."""
        host = target.lower()
        for prefix in ("http://", "https://"):
            if host.startswith(prefix):
                host = host[len(prefix) :]
        host = host.split("/")[0].split(":")[0]
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return any(host == n or host.endswith(f".{n}") for n in self.names)
        return any(address in net for net in self.networks)

    def check(self, target: str) -> None:
        """Raise PermissionError if `target` is out of scope."""
        if not self.allows(target):
            raise PermissionError(
                f"{target!r} is outside the engagement scope {self.raw}. "
                "Do not probe it. Pick an in-scope target."
            )

    def __repr__(self) -> str:
        return f"Scope({', '.join(self.raw)})"

PositionKind = Literal[
    "own-pod", "other-pod", "node-root", "control-plane", "cloud-account", "cluster-admin"
]


# --------------------------------------------------------------------------
# Typed contracts
# --------------------------------------------------------------------------


class Position(BaseModel):
    """Where the agent currently stands, and what that position grants."""

    in_container: bool
    in_kubernetes: bool
    runtime: str = Field(description="docker, containerd, cri-o, or unknown")
    namespace: str = ""
    service_account: str = ""
    node_name: str = ""
    pod_ip: str = ""
    effective_caps: list[str] = Field(default_factory=list)
    host_pid: bool = False
    host_network: bool = False
    host_ipc: bool = False
    privileged: bool = False
    host_mounts: list[str] = Field(
        default_factory=list,
        description=(
            "ONLY mounts that grant real host filesystem access — a hostPath of "
            "/, /etc, /var/log, /var/lib/kubelet, /proc, /dev, or a host device. "
            "EXCLUDE pod-scoped volumes the kubelet always creates: etc-hosts, "
            "resolv.conf, termination-log, serviceaccount projections, configMaps, "
            "secrets, emptyDir, and any path under "
            "/var/lib/kubelet/pods/<uid>/ that belongs to THIS pod. Those have "
            "host source paths but grant no host access, and listing them is a "
            "false positive."
        ),
    )
    sensitive_sockets: list[str] = Field(
        default_factory=list, description="docker.sock, containerd.sock, crio.sock if present"
    )
    seccomp: str = ""
    apparmor: str = ""
    notes: str = ""


class RbacProfile(BaseModel):
    """What this ServiceAccount token is actually permitted to do."""

    token_present: bool
    can_list_namespaces: bool = False
    can_get_secrets: bool = False
    can_create_pods: bool = False
    can_exec_pods: bool = False
    can_get_nodes: bool = False
    can_impersonate: bool = False
    can_escalate_or_bind: bool = False
    can_create_sa_tokens: bool = False
    dangerous_verbs: list[str] = Field(
        default_factory=list, description="Specific verb/resource pairs that enable escalation"
    )
    raw_can_i: str = Field(default="", description="Trimmed output of the permission review")


class Reachable(BaseModel):
    """Something the agent can talk to on the network."""

    address: str
    port: int
    service: str = Field(description="kubelet, etcd, apiserver, coredns, app, unknown")
    evidence: str = Field(description="What the probe actually returned")
    authenticated_access: bool = Field(
        default=False, description="True if it answered without credentials"
    )


class NetworkMap(BaseModel):
    """The cluster network as observed from this pod."""

    pod_cidr: str = ""
    service_cidr: str = ""
    apiserver: str = ""
    dns_server: str = ""
    network_policy_enforced: bool | None = Field(
        default=None, description="None if undetermined. False means pods are flat and reachable"
    )
    reachable: list[Reachable] = Field(default_factory=list)
    discovered_services: list[str] = Field(
        default_factory=list, description="Service DNS names found via DNS or the API"
    )
    cloud_metadata_reachable: bool = False
    notes: str = ""


class EscalationPath(BaseModel):
    """A concrete route from the current position to a higher-privilege one."""

    name: str
    technique: str = Field(
        description="e.g. mounted-docker-sock, hostpath-root, kubelet-10250-exec, "
        "cgroup-release-agent, rbac-create-pod, imds-node-role, etcd-unauth"
    )
    from_position: str
    to_position: PositionKind
    target: str = Field(description="Specific pod, node, or endpoint this reaches")
    prerequisites: list[str]
    steps: list[str] = Field(description="Ordered, reproducible commands or requests")
    verified: bool = False
    proof: str = Field(default="", description="Observed output proving it works. Empty if unproven")
    blocked_by: str = Field(default="", description="If unproven, what stopped it")
    severity: Literal["critical", "high", "medium", "low", "info"] = "high"
    remediation: str = ""

    def key(self) -> str:
        return f"{self.technique.lower().strip()}|{self.to_position}|{self.target.lower().strip()}"


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------


class K8sEscapeAgent(Agent):
    """You are a Kubernetes security specialist assessing a cluster from inside a pod.

    Your objective: establish exactly how far an attacker who lands in this pod
    can get. Reaching other pods, then the node, then the control plane, then
    the cloud account. You prove each step rather than asserting it.

    ## Rules of engagement

    - `self.scope` is the authorized network range. **Call
      `self.scope.check(addr)` before probing any address.** Cluster networks
      often route to corporate networks — do not wander off the cluster.
    - **Non-destructive.** Prove capability without using it:
      - Test permissions with `kubectl auth can-i` or a `SelfSubjectAccessReview`
        instead of performing the action.
      - Prove create rights with `--dry-run=server`, not a real create.
      - If you must create something to prove a path, use an innocuous image,
        name it clearly, and delete it immediately. Say so in the steps.
      - Read *one* record to prove secret access. Do not dump every secret, and
        never exfiltrate data outside the pod.
      - No workload disruption. Never delete, evict, cordon, or scale anything.
    - Other tenants' data is off limits even when reachable. Prove the access
      path, then stop.

    ## Methodology

    **1. Locate yourself.** `/.dockerenv`, `/proc/1/cgroup`,
    `/proc/self/mountinfo`, `/proc/self/status` (CapEff — decode it),
    `KUBERNETES_SERVICE_HOST`, the ServiceAccount dir. Determine: privileged?
    hostPID/hostNetwork/hostIPC? which host paths are mounted? any container
    runtime socket? seccomp and AppArmor status?

    **2. Enumerate your identity.** The token at
    `/var/run/secrets/kubernetes.io/serviceaccount/token` plus `ca.crt` and
    `namespace`. Decode the token. Then find what it can do — the highest-value
    verbs are `create pods` (mount hostPath, run privileged, schedule onto a
    control-plane node), `pods/exec` (jump into a higher-privilege pod), `get
    secrets` (steal other tokens), `escalate`/`bind` (grant yourself
    cluster-admin), `impersonate`, `serviceaccounts/token`, and anything on
    `nodes` or `nodes/proxy`.

    **3. Map the network — this is where lateral movement comes from.**
    - Derive the pod CIDR from your own IP, routes, and netmask. Derive the
      service CIDR from `KUBERNETES_SERVICE_HOST` (the API server is usually
      `.1` of it).
    - Enumerate DNS thoroughly. CoreDNS will happily list services: query SRV
      records like `_._tcp.<svc>.<ns>.svc.cluster.local`, and
      `any.any.svc.cluster.local` style wildcards where they work.
    - **Test whether NetworkPolicy is enforced at all.** If pods are flat, you
      can reach every workload in the cluster — that is usually the single most
      important finding.
    - Scan for control-plane and node ports specifically: kubelet **10250**
      (read/write — `/pods` lists everything, `/run` gives exec on any pod on
      that node) and **10255** (read-only), etcd **2379/2380**, apiserver
      **6443/443**, controller-manager **10257**, scheduler **10259**.
    - Check cloud metadata at **169.254.169.254**. On managed clusters the node
      IAM role is frequently the shortest path to owning everything.
    - Pod CIDRs are large. Do not brute-force a /16 host by host — derive live
      ranges from DNS and the API first, then scan targeted ports on what you
      actually found.

    **4. Work out escalation paths.** Classic escapes worth checking: mounted
    container runtime socket; hostPath mount of `/` or `/var/lib/kubelet`;
    `privileged: true`; `CAP_SYS_ADMIN` with cgroup v1 `release_agent`;
    `CAP_SYS_MODULE`; `CAP_DAC_READ_SEARCH` (open_by_handle_at); writable
    `/proc/sys/kernel/core_pattern`; unmasked `/sys/fs/cgroup`; host device
    access under `/dev`. Plus the RBAC and network paths above.

    Chain them. A path is more valuable when it composes — `get secrets` on a
    token that itself has `create pods` is cluster-admin two steps away.

    ## Working style

    `self.todo` is your queue and is visible every turn. Every clue becomes an
    item: an unexpected mount, an open port, a permission you did not expect, a
    service name from DNS. Your position and reached set are also pinned in
    context. Do not re-report a path already listed — refine it or chain from it.

    `self.libs` is your persistent library. Enumeration primitives you work out
    here should be saved so the next cluster assessment starts with them.
    """

    def __init__(self, scope: Scope, out_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        WORK.mkdir(parents=True, exist_ok=True)
        self.scope = scope
        self.out_dir = out_dir or (HERE / "state")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.out_dir / "findings.jsonl"
        self._live_md = self.out_dir / "findings-live.md"
        self.shell = ShellTools(cwd=str(WORK))
        self.libs = SkillWriting(self, LIBS)
        self.todo = TodoManager()

        self.position: Position | None = None
        self.rbac: RbacProfile | None = None
        self.network: NetworkMap | None = None
        self.paths: list[EscalationPath] = []
        self.reached: set[str] = {"own-pod"}

        spec(self, "context", hidden=False)
        self.context["engagement_scope"] = f"Authorized network scope: {scope.raw}"
        self.context["position"] = Context(expr="self.render_position()")
        self.context["reached"] = Context(expr="self.render_reached()")
        self.context["todo_status"] = Context(expr="self.todo.status()")
        self.context["paths"] = Context(expr="self.render_paths()")

    # -- deterministic helpers ---------------------------------------------

    async def kube(self, path: str, method: str = "GET") -> str:
        """Call the Kubernetes API with the pod's ServiceAccount token.

        `path` is an API path such as "/api/v1/namespaces/default/pods".
        """
        host = f"https://{self._apiserver()}"
        self.scope.check(self._apiserver())
        cmd = (
            f'curl -sS -X {method} --max-time 15 --cacert {SA_PATH}/ca.crt '
            f'-H "Authorization: Bearer $(cat {SA_PATH}/token)" '
            f'"{host}{path}"'
        )
        return str(await self.shell.run(cmd))

    async def probe(self, address: str, port: int, path: str = "/") -> str:
        """Unauthenticated HTTP(S) probe of an in-scope address:port."""
        self.scope.check(address)
        cmd = (
            f"curl -sS -k --max-time 8 -o - -w '\\nHTTP:%{{http_code}}\\n' "
            f"https://{address}:{port}{path} 2>&1 | head -c 2000 || "
            f"curl -sS --max-time 8 -o - -w '\\nHTTP:%{{http_code}}\\n' "
            f"http://{address}:{port}{path} 2>&1 | head -c 2000"
        )
        return str(await self.shell.run(cmd))

    async def scan(self, target: str, ports: str = "6443,443,2379,10250,10255,10257,10259") -> str:
        """Port-scan a single in-scope target with nmap, if nmap is installed."""
        self.scope.check(target)
        return str(await self.shell.run(f"nmap -Pn -n -p {ports} --open -T4 {target}"))

    async def sweep_ports(
        self,
        cidr: str,
        ports: tuple[int, ...] = K8S_PORTS,
        concurrency: int = 400,
        timeout: float = 1.0,
    ) -> list[Reachable]:
        """Fast parallel TCP scan of an entire in-scope CIDR. Pure Python, no nmap.

        **Use this instead of looping nmap over hosts.** It scans every address
        in `cidr` across `ports` concurrently and returns only what answered, in
        one call. A /23 across the default port set takes seconds.

        Every address is scope-checked; out-of-scope ranges raise immediately.
        """
        network = ipaddress.ip_network(cidr, strict=False)
        self.scope.check(str(network.network_address))
        self.scope.check(str(network.broadcast_address))

        semaphore = asyncio.Semaphore(concurrency)
        found: list[Reachable] = []

        async def probe_one(host: str, port: int) -> None:
            async with semaphore:
                try:
                    _, writer = await asyncio.wait_for(
                        asyncio.open_connection(host, port), timeout=timeout
                    )
                except (OSError, asyncio.TimeoutError):
                    return
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
                found.append(
                    Reachable(
                        address=host,
                        port=port,
                        service=K8S_PORT_NAMES.get(port, "unknown"),
                        evidence=f"TCP connect succeeded on {host}:{port}",
                    )
                )

        hosts = [str(h) for h in network.hosts()]
        await asyncio.gather(*(probe_one(h, p) for h in hosts for p in ports))
        found.sort(key=lambda r: (r.address, r.port))
        return found

    def _apiserver(self) -> str:
        return os.getenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")

    # -- crash-safe persistence --------------------------------------------
    #
    # A pod can be restarted or evicted at any moment, so nothing may live only
    # in memory. Every record is appended to JSONL (append-only, so a crash
    # mid-write loses at most one line) AND emitted to stdout, because container
    # logs survive a restart and are retrievable with `kubectl logs --previous`
    # even when the filesystem does not.

    def _persist(self, kind: str, payload: dict[str, Any]) -> None:
        record = {"kind": kind, **payload}
        line = json.dumps(record, default=str, ensure_ascii=False)
        try:
            with self._jsonl.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())  # survive an ungraceful kill
        except OSError as exc:
            print(f"# WARN: could not write {self._jsonl}: {exc}")
        # Second, independent durability channel — captured by the container log.
        print(f"@@FINDING@@ {line}", flush=True)

    def _rewrite_live_markdown(self) -> None:
        """Regenerate the human-readable running report. Cheap; safe to lose."""
        lines = ["# Live findings", "", self.render_reached(), "", "## Position", "",
                 "```", self.render_position(), "```", "", "## Escalation paths", ""]
        for i, p in enumerate(self.paths, 1):
            status = "VERIFIED" if p.verified else "unproven"
            lines += [
                f"### {i}. [{status}] {p.name}",
                f"- technique: `{p.technique}`",
                f"- reaches: **{p.to_position}** ({p.target})",
                f"- severity: {p.severity}",
                f"- steps:\n" + "\n".join(f"  {n}. {s}" for n, s in enumerate(p.steps, 1)),
            ]
            if p.proof:
                lines += ["- proof:", "```", p.proof[:1500], "```"]
            if p.blocked_by:
                lines += [f"- blocked by: {p.blocked_by}"]
            lines += [f"- remediation: {p.remediation}", ""]
        try:
            self._live_md.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            pass

    def snapshot(self) -> None:
        """Persist position / RBAC / network so a restart does not lose recon."""
        for kind, obj in [
            ("position", self.position),
            ("rbac", self.rbac),
            ("network", self.network),
        ]:
            if obj is not None:
                self._persist(kind, {"data": obj.model_dump()})

    def load_prior(self) -> int:
        """Rehydrate state from a previous run's JSONL. Returns records restored."""
        if not self._jsonl.exists():
            return 0
        restored = 0
        for raw in self._jsonl.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue  # torn final line from a hard kill — skip it
            kind = record.get("kind")
            try:
                if kind == "path":
                    path = EscalationPath(**record["data"])
                    # Append-only log replayed in order: last write wins, so a
                    # later verified record must REPLACE the earlier unverified
                    # one rather than be skipped as a duplicate.
                    for index, existing in enumerate(self.paths):
                        if existing.key() == path.key():
                            self.paths[index] = path
                            break
                    else:
                        self.paths.append(path)
                elif kind == "reached":
                    self.reached.add(record["position"])
                elif kind == "position":
                    self.position = Position(**record["data"])
                elif kind == "rbac":
                    self.rbac = RbacProfile(**record["data"])
                elif kind == "network":
                    self.network = NetworkMap(**record["data"])
                else:
                    continue
                restored += 1
            except Exception:
                continue
        return restored

    # -- recording ----------------------------------------------------------

    def record_path(self, path: EscalationPath) -> str:
        """Record an escalation path. Deduplicates by technique/destination/target.

        Written to disk immediately — nothing is held only in memory.
        """
        existing = next((p for p in self.paths if p.key() == path.key()), None)
        if existing is not None:
            # Verification upgrades an existing path; persist the new state.
            if path.verified and not existing.verified:
                existing.verified = True
                existing.proof = path.proof
                existing.blocked_by = path.blocked_by
                self._persist("path", {"data": existing.model_dump()})
                self._rewrite_live_markdown()
                return f"updated to verified: {path.name}"
            return f"duplicate, already tracked: {path.name}"
        self.paths.append(path)
        if path.verified:
            self.reached.add(f"{path.to_position}:{path.target}")
        self._persist("path", {"data": path.model_dump()})
        self._rewrite_live_markdown()
        return f"tracked #{len(self.paths)}: {path.name}"

    def mark_reached(self, position: str) -> str:
        """Note that a position is now demonstrably reachable. Persisted immediately."""
        if position not in self.reached:
            self.reached.add(position)
            self._persist("reached", {"position": position})
            self._rewrite_live_markdown()
        return f"reached: {position} ({len(self.reached)} total)"

    def render_position(self) -> str:
        """Your current position — pinned into context."""
        if self.position is None:
            return "Position not yet established. Run locate() first."
        p = self.position
        flags = [n for n, v in [
            ("privileged", p.privileged), ("hostPID", p.host_pid),
            ("hostNetwork", p.host_network), ("hostIPC", p.host_ipc),
        ] if v]
        return (
            f"container={p.in_container} k8s={p.in_kubernetes} runtime={p.runtime}\n"
            f"ns={p.namespace} sa={p.service_account} node={p.node_name} ip={p.pod_ip}\n"
            f"flags={flags or ['none']} caps={p.effective_caps[:12]}\n"
            f"host_mounts={p.host_mounts[:8]} sockets={p.sensitive_sockets}"
        )

    def render_reached(self) -> str:
        """Positions demonstrably reachable so far — pinned into context."""
        return f"{len(self.reached)} position(s): " + ", ".join(sorted(self.reached))

    def render_paths(self) -> str:
        """Escalation paths found so far — pinned into context."""
        if not self.paths:
            return "No escalation paths found yet."
        return "\n".join(
            f"{i}. [{'VERIFIED' if p.verified else 'unproven'}] {p.technique} "
            f"-> {p.to_position} ({p.target}): {p.name}"
            for i, p in enumerate(self.paths, 1)
        )

    def verified_paths(self) -> list[EscalationPath]:
        """Paths actually proven to work."""
        return [p for p in self.paths if p.verified]

    # -- generation methods ------------------------------------------------

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=35)))
    async def locate(self) -> Position:
        """Determine exactly where you are and what this position grants.

        Work through methodology step 1. Read the real files — decode CapEff
        from `/proc/self/status` rather than guessing, and enumerate mounts from
        `/proc/self/mountinfo` rather than assuming.

        Be strict about `host_mounts`: a mount only counts if it grants access to
        the host filesystem. Every pod has kubelet-managed volumes whose source
        paths sit under `/var/lib/kubelet/pods/<uid>/` — etc-hosts, resolv.conf,
        serviceaccount projections, configMaps, secrets, emptyDirs. Those are
        pod-scoped and grant nothing. Listing them wastes later analysis on dead
        ends, so leave them out and put them in `notes` if you want them recorded.

        Equally, if capabilities are empty and there is no privileged flag and no
        runtime socket, say so plainly in `notes`. A hardened pod is a real
        result, not a failure to find something.

        Fill every field you can establish. Leave a field at its default only
        when you genuinely could not determine it, and say so in `notes`.
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=35)))
    async def enumerate_rbac(self) -> RbacProfile:
        """Establish what your ServiceAccount token is permitted to do.

        Prefer `kubectl auth can-i --list` if kubectl exists; otherwise POST a
        SelfSubjectRulesReview via `self.kube(...)`. Test permissions, do not
        exercise them.

        Populate `dangerous_verbs` with the specific verb/resource pairs that
        enable escalation, not every permission you hold.
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=45)))
    async def map_network(self) -> NetworkMap:
        """Map the cluster network from here — methodology step 3.

        Derive the CIDRs, enumerate DNS aggressively, determine whether
        NetworkPolicy is enforced, and probe specifically for kubelet, etcd,
        apiserver, controller-manager, scheduler, and cloud metadata.

        **Use `self.sweep_ports(cidr)` for scanning.** It scans an entire CIDR
        across all the control-plane ports concurrently in pure Python and
        returns only what answered — a /23 takes about a second. Call it ONCE
        per CIDR. Do not loop `self.scan()` or nmap over hosts one at a time:
        that costs you an LLM round trip per host and will take hours.

        Every LLM turn is expensive, so do mechanical work in bulk inside a
        single cell: build the target list, sweep it, then reason about the
        results. Reserve turns for decisions, not for iteration.

        Record what answered in `reachable`, and set
        `authenticated_access=True` on anything that responded with no
        credentials — those are your best leads.
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=45)))
    async def find_paths(self, focus: str) -> list[EscalationPath]:
        """Work out escalation paths for one focus area, without executing them yet.

        Use your position, RBAC profile, and network map — all pinned in your
        context. For each viable route, state the prerequisites you have already
        confirmed and the exact ordered steps.

        **Only propose paths whose preconditions you have already confirmed in
        your Position.** Your position is pinned in context. If `caps` is empty,
        do not propose CAP_SYS_ADMIN, CAP_SYS_MODULE, or CAP_DAC_READ_SEARCH
        techniques. If `privileged` is false, do not propose privileged escape.
        If `sensitive_sockets` is empty, there is no runtime socket to abuse. If
        `host_pid`/`host_network` are false, nsenter and host-network paths do
        not exist here.

        Listing textbook escapes that this container cannot perform is the single
        worst thing you can do — it produces a false report and wastes the whole
        run. A hardened container with two real paths beats a fabricated list of
        twelve. Returning an EMPTY LIST because everything is properly locked
        down is a correct, valuable answer.

        `target` must be a concrete identifier — a node name, pod name, or
        endpoint. Not a description of the technique.

        Prefer chains that compose toward the control plane over isolated
        curiosities. Add todos for new leads. Set `verified=False` here —
        proving them is `verify()`'s job. Return what you found.
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=45)))
    async def verify(self, path: EscalationPath) -> EscalationPath:
        """Prove this path works, non-destructively, and return it updated.

        Execute the minimum needed to demonstrate the position is genuinely
        reachable, honouring the non-destructive rules: `--dry-run=server` for
        creates, `can-i` for permissions, one record to prove read access, and
        immediate cleanup of anything you had to create.

        **`proof` must be the literal captured stdout/stderr of commands you
        actually ran in this session** — pasted, not summarized, not
        reconstructed from memory, not what you expect the output would be.
        Include the command alongside its output.

        Set `verified=True` ONLY if you executed something and read real output
        confirming the impact. If you did not run a command, `verified` is False,
        full stop. A caller checks your proof against the container's established
        capabilities and will reject a claim that contradicts them, so asserting
        success you did not observe achieves nothing except a wrong report.

        Otherwise set it False and fill `blocked_by` honestly — a blocked path is
        a useful result and tells the defender their control worked. `Permission
        denied`, `Operation not permitted`, and RBAC `forbidden` are the expected
        outcomes in a well-configured cluster, and reporting them accurately is
        the job.

        When a path lands you somewhere new, call
        `self.mark_reached("<kind>:<target>")` so the frontier expands.
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=30)))
    async def report(self) -> str:
        """Write the cluster assessment as markdown.

        Sections: executive summary stating how far an attacker landing in this
        pod can actually get; the attack graph as a chain from initial position
        to the furthest verified position; verified paths by severity with
        steps, proof, and remediation; unproven paths with what blocked them
        (credit the controls that worked); network exposure including whether
        NetworkPolicy is enforced; and what you could not assess.

        Source of truth is `self.position`, `self.rbac`, `self.network`,
        `self.paths`, and `self.reached`. Be explicit about proven versus
        theoretical — that distinction is the whole value of this document.
        """
        ...


# --------------------------------------------------------------------------
# Frontier expansion: each newly reached position opens new enumeration.
# --------------------------------------------------------------------------

BASELINE_FOCUS = [
    "Container escape via mounts, capabilities, and runtime sockets",
    "RBAC escalation from this ServiceAccount to cluster-admin",
    "Lateral movement to other pods over the pod network",
    "Kubelet API access on reachable nodes (10250/10255)",
    "Control plane exposure: apiserver anonymous auth, etcd, scheduler, controller-manager",
    "Cloud metadata and node IAM role abuse",
]


async def campaign(
    agent: K8sEscapeAgent,
    *,
    dry_rounds: int = 2,
    max_rounds: int = 30,
) -> str:
    """Locate, enumerate, map, then expand the frontier until it stops growing.

    Termination is driven by the reached-position frontier, not a round count:
    when `dry_rounds` consecutive rounds discover no new escalation path, the
    attack surface from this foothold is exhausted.
    """
    print(f"# scope: {agent.scope}")

    if agent.position is None:
        try:
            agent.position = await agent.locate()
            agent.snapshot()
            print("# locate: ok")
        except Exception as exc:
            print(f"# locate FAILED ({type(exc).__name__})")
    else:
        print("# locate: restored from prior run")
    if agent.position:
        print(agent.render_position())

    if agent.rbac is None:
        try:
            agent.rbac = await agent.enumerate_rbac()
            agent.snapshot()
            print("# rbac: ok")
        except Exception as exc:
            print(f"# rbac FAILED ({type(exc).__name__})")
    else:
        print("# rbac: restored from prior run")

    if agent.network is None:
        try:
            agent.network = await agent.map_network()
            agent.snapshot()
        except Exception as exc:
            print(f"# map_network FAILED ({type(exc).__name__})")
    else:
        print("# network: restored from prior run")
    if agent.network:
        n = agent.network
        print(
            f"# network: pod_cidr={n.pod_cidr} svc_cidr={n.service_cidr} "
            f"netpol={n.network_policy_enforced} reachable={len(n.reachable)} "
            f"imds={n.cloud_metadata_reachable}"
        )

    for focus in BASELINE_FOCUS:
        agent.todo.add(focus, notes="baseline focus area", area=focus)

    seen: set[str] = set()
    dry = 0
    rounds = 0

    while dry < dry_rounds and rounds < max_rounds:
        rounds += 1
        queue = [t for t in agent.todo.list_todos() if t.status != "done"]
        if not queue:
            dry += 1
            print(f"# round {rounds}: queue exhausted (dry {dry}/{dry_rounds})")
            continue

        item = queue[0]
        before = set(agent.reached)
        print(f"# round {rounds}: {item.title}  ({len(queue)} queued)")
        try:
            found = await agent.find_paths(item.v.area or item.title)
        except Exception as exc:
            print(f"#   find_paths FAILED ({type(exc).__name__})")
            found = []
        finally:
            agent.todo.done(item.id)

        for path in found:
            agent.record_path(path)

        fresh = [p for p in found if p.key() not in seen]
        if not fresh:
            dry += 1
            print(f"#   nothing new (dry {dry}/{dry_rounds})")
            continue

        dry = 0
        for path in fresh:
            seen.add(path.key())
        print(f"#   {len(fresh)} candidate path(s) -> verifying")

        for path in fresh:
            # Deterministic gate BEFORE spending a turn: reject any technique
            # whose precondition is provably absent from the known Position.
            ok, reason = feasible(path, agent.position, agent.rbac)
            if not ok:
                path.verified = False
                path.blocked_by = f"INFEASIBLE (rejected without testing): {reason}"
                path.severity = "info"
                agent.record_path(path)
                print(f"#     [INFEASIBLE] {path.technique} — {reason[:80]}")
                continue
            try:
                proven = await agent.verify(path)
                # Re-check after verification: the model may have marked a path
                # verified anyway. Established facts outrank its assertion.
                ok, reason = feasible(proven, agent.position, agent.rbac)
                if proven.verified and not ok:
                    proven.verified = False
                    proven.blocked_by = f"CONTRADICTS ESTABLISHED POSITION: {reason}"
                    print(f"#     [REJECTED] {proven.technique} — claimed verified but {reason[:70]}")
                elif proven.verified and len(proven.proof.strip()) < 40:
                    proven.verified = False
                    proven.blocked_by = (
                        "claimed verified but supplied no substantive command output as proof"
                    )
                    print(f"#     [REJECTED] {proven.technique} — no real proof supplied")
                agent.record_path(proven)  # upgrades + persists in one place
                if proven.verified:
                    agent.mark_reached(f"{proven.to_position}:{proven.target}")
                    print(f"#     [VERIFIED] {proven.technique} -> {proven.to_position}")
                elif not proven.blocked_by.startswith(("CONTRADICTS", "claimed")):
                    print(f"#     [blocked] {proven.technique} -> {proven.to_position}")
            except Exception as exc:
                print(f"#     verify FAILED ({type(exc).__name__}) {path.name}")

        # Each NEWLY reached position is a fresh vantage point — enumerate from
        # it. Diff the set rather than counting, so we queue the positions that
        # actually appeared instead of re-queueing the same one every round.
        gained = sorted(set(agent.reached) - before)
        if gained:
            print(f"#   frontier grew: {gained} -> queueing follow-up enumeration")
            for position in gained:
                agent.todo.add(
                    f"Enumerate onward from newly reached position: {position}",
                    notes="frontier expansion",
                    area=(
                        f"You now hold {position}. What further access does that grant? "
                        "Re-enumerate identity, mounts, network, and RBAC from there and "
                        "chain onward toward the control plane."
                    ),
                )

    reason = "frontier exhausted" if dry >= dry_rounds else f"hit max_rounds={max_rounds}"
    print(f"\n# stopped: {reason} after {rounds} round(s)")
    print(f"# {len(agent.verified_paths())} verified of {len(agent.paths)} candidate path(s)")
    print(f"# {agent.render_reached()}")
    return await agent.report()


async def smoke() -> None:
    """Exercise the deterministic parts. No LLM, no cluster."""
    from nooa.unifiedllm import FakeLLMClient

    scope = Scope(("10.244.0.0/16", "10.96.0.0/12"))
    agent = K8sEscapeAgent(scope, llm=FakeLLMClient())

    print("scope       :", scope)
    for probe in ["10.244.1.7", "10.96.0.1", "10.13.37.5", "corp.internal"]:
        print(f"  allows {probe:20} {scope.allows(probe)}")

    print("\nposition    :", agent.render_position())
    agent.position = Position(
        in_container=True, in_kubernetes=True, runtime="containerd",
        namespace="default", service_account="default", node_name="node-1",
        pod_ip="10.244.1.7", effective_caps=["CAP_SYS_ADMIN", "CAP_NET_RAW"],
        privileged=True, host_mounts=["/var/lib/kubelet", "/"],
        sensitive_sockets=["/var/run/containerd/containerd.sock"],
    )
    print("position    :\n" + agent.render_position())

    p = EscalationPath(
        name="Escape to node root via mounted containerd socket",
        technique="mounted-containerd-sock",
        from_position="own-pod", to_position="node-root", target="node-1",
        prerequisites=["containerd.sock mounted", "CAP_SYS_ADMIN"],
        steps=["ctr -a /var/run/containerd/containerd.sock images ls",
               "ctr run --privileged --mount type=bind,src=/,dst=/host ..."],
        severity="critical",
        remediation="Never mount the container runtime socket into a pod.",
    )
    print("\nrecord      :", agent.record_path(p))
    print("dedup       :", agent.record_path(p.model_copy()))
    print("paths       :", agent.render_paths())

    p.verified = True
    p.proof = "id -> uid=0(root); /host/etc/shadow readable"
    agent.record_path(p)
    print("mark        :", agent.mark_reached("node-root:node-1"))
    print("reached     :", agent.render_reached())
    print("verified    :", len(agent.verified_paths()), "of", len(agent.paths))

    try:
        await agent.probe("corp.internal", 443)
    except PermissionError as exc:
        print("scope guard : blocked ->", str(exc)[:56])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="no LLM, no cluster")
    parser.add_argument("--scope", help="comma-separated CIDRs/hosts (pod + service CIDR)")
    parser.add_argument("--dry-rounds", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--model", default="azure/deepseek-ai/deepseek-v4-pro")
    parser.add_argument("--max-tokens", type=int, default=40000,
                        help="history budget before summarizing; lower = faster turns")
    parser.add_argument("--out", default="cluster-report.md")
    parser.add_argument("--out-dir", default=None,
                        help="directory for live findings.jsonl + findings-live.md")
    parser.add_argument("--resume", action="store_true",
                        help="rehydrate state from a prior run's findings.jsonl")
    args = parser.parse_args()

    if args.smoke or not args.scope:
        asyncio.run(smoke())
        return

    async def run() -> None:
        scope = Scope(tuple(s.strip() for s in args.scope.split(",") if s.strip()))
        out_dir = Path(args.out_dir) if args.out_dir else None
        agent = K8sEscapeAgent(scope, out_dir=out_dir, llm=make_llm(args.model))
        TokenBudgetSummarizer.install(
            agent, config=TokenBudgetConfig(max_tokens=args.max_tokens)
        )
        print(f"# model: {agent._llm.model}")
        print(f"# live findings: {agent._jsonl}")
        if args.resume:
            print(f"# resumed {agent.load_prior()} record(s) from prior run")
        markdown = await campaign(
            agent, dry_rounds=args.dry_rounds, max_rounds=args.max_rounds
        )
        Path(args.out).write_text(markdown)
        print(f"# wrote {args.out} ({len(markdown)} chars)")

    asyncio.run(run())


if __name__ == "__main__":
    main()



