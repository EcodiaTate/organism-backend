"""
Simula Observer — Standalone eBPF Network Event Observer

Deploys a privileged sidecar container that attaches BPF kprobes to
tcp_sendmsg, tcp_recvmsg, and security_socket_connect, then prints
PID/comm/flow events continuously to stdout.

Two modes:
  bpftrace  — quickest validation, zero Python, single bpftrace script
  bcc       — full Python observer with ring buffer polling + health endpoint

Container requirements:
  privileged: true, pid: host, network_mode: host
  cap_add: [SYS_ADMIN, BPF, PERFMON, NET_ADMIN]
  volumes: /sys/kernel/debug:ro, /sys/fs/bpf:ro

Usage:
  cd observer/
  docker compose -f docker-compose.observer.yml up --build
"""

__all__: list[str] = []
