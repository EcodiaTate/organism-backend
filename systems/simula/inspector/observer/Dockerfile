# Simula Observer — eBPF Network Event Observer
#
# Provides both bpftrace (quick validation) and BCC Python (full observer)
# in a single image. No clang/llvm needed — BCC does runtime compilation
# and bpftrace has its own compiler.
#
# Matches the ubuntu:24.04 base used by the existing topology.py sidecar.
#
# Build:
#   docker build -t simula-observer .
#
# Run (requires privileged + host PID/network):
#   docker run --rm --privileged --pid=host --network=host \
#     -v /sys/kernel/debug:/sys/kernel/debug:ro \
#     -v /sys/fs/bpf:/sys/fs/bpf:ro \
#     simula-observer

FROM ubuntu:24.04

# Avoid interactive prompts during apt-get install.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        bpftrace \
        python3 \
        python3-bpfcc \
        iproute2 \
        procps \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/simula/observer

COPY observer.py bpftrace_probes.bt run_observer.sh ./
RUN chmod +x run_observer.sh

# Default: BCC mode on port 9472.
ENV OBSERVER_MODE=bcc
ENV OBSERVER_PORT=9472
ENV OBSERVER_PROBES=tcp_sendmsg,tcp_recvmsg,security_socket_connect

EXPOSE 9472

ENTRYPOINT ["/opt/simula/observer/run_observer.sh"]
