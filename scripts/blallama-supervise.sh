#!/usr/bin/env bash
#
# Keep blallama running: restart it whenever it exits on a fault.
#
#   scripts/blallama-supervise.sh [blallama args...]
#   BLALLAMA=target/release/blallama scripts/blallama-supervise.sh models/
#
# blallama exits on purpose when it can no longer trust its process —
# 70 after a panic, 75 after a backend failure llama.cpp does not
# recover from in-process (a Metal OOM, a failed llama_decode) — and
# expects a supervisor to start a fresh one (see bin/blallama, "Run it
# under a supervisor"). launchd and systemd (`Restart=on-failure`) do
# that; this is the minimal version for a terminal or tmux.
#
# - The arguments are passed through unchanged on every start.
# - A run that dies within HEALTHY_SECS of starting doubles the delay
#   before the next start (1 s up to MAX_DELAY_SECS), so a crash loop —
#   a model that cannot load, a port in use — backs off instead of
#   spinning; a run that lasted resets it.
# - Each exit and restart is logged to stderr with its exit code.
# - A clean exit (0: SIGTERM or Ctrl-C drained it) ends the loop, and
#   SIGTERM / SIGINT sent here are forwarded to blallama, which drains,
#   then the loop ends with its exit code.

set -u

bin="${BLALLAMA:-blallama}"
max_delay="${MAX_DELAY_SECS:-60}"
healthy="${HEALTHY_SECS:-60}"

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) blallama-supervise: $*" >&2
}

describe() {
    case "$1" in
        0) echo "clean exit" ;;
        70) echo "panic" ;;
        75) echo "backend failure" ;;
        *) echo "exit code $1" ;;
    esac
}

child=""
stopping=0
forward() {
    stopping=1
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null
    fi
}
trap forward TERM INT

delay=1
while :; do
    started=$(date +%s)
    log "starting: $bin $*"
    "$bin" "$@" &
    child=$!
    # `wait` returns early when a trapped signal arrives; keep waiting
    # until blallama itself has exited (it drains on SIGTERM).
    code=0
    while kill -0 "$child" 2>/dev/null; do
        wait "$child"
        code=$?
    done
    child=""
    ran=$(($(date +%s) - started))

    if [[ "$code" -eq 0 || "$stopping" -eq 1 ]]; then
        log "blallama stopped ($(describe "$code")) after ${ran}s"
        exit "$code"
    fi

    if [[ "$ran" -ge "$healthy" ]]; then
        delay=1
    fi
    log "blallama died ($(describe "$code")) after ${ran}s; restarting in ${delay}s"
    sleep "$delay" &
    wait $! || true
    if [[ "$stopping" -eq 1 ]]; then
        log "stopped while waiting to restart"
        exit "$code"
    fi
    delay=$((delay * 2 > max_delay ? max_delay : delay * 2))
done
