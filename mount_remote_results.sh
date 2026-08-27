#!/bin/sh

set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
projects_dir=$(dirname -- "$project_dir")
mount_dir=${WACKYWACKY_REMOTE_RESULTS_MOUNT:-"$projects_dir/.remote-wackywacky-results"}
remote_results=${WACKYWACKY_REMOTE_RESULTS_SOURCE:-"recod-headnode:/home/matheus.sanches/projects/wackywacky-analysis/results"}
snapshot_id=${WACKYWACKY_SNAPSHOT_ID:-"20260821-cd2cf3a57099"}
snapshot_dir="$mount_dir/$snapshot_id"
command=${1:-mount}

is_mounted() {
  mount | grep -F " on $mount_dir " >/dev/null 2>&1
}

validate_layout() {
  if [ ! -f "$snapshot_dir/manifest.json" ] || \
     [ ! -f "$snapshot_dir/methods.json" ] || \
     [ ! -f "$snapshot_dir/summary.json" ] || \
     [ ! -f "$snapshot_dir/checksums.sha256" ]; then
    printf 'Unexpected remote results layout at %s\n' "$snapshot_dir" >&2
    printf 'Expected manifest.json, methods.json, summary.json and checksums.sha256.\n' >&2
    return 1
  fi
}

show_status() {
  if ! is_mounted; then
    printf 'Remote results are not mounted at %s\n' "$mount_dir"
    return 1
  fi
  validate_layout
  printf 'Remote results mounted read-only at %s\n' "$mount_dir"
  printf 'Snapshot: %s\n' "$snapshot_dir"
  printf 'Remote source: %s\n' "$remote_results"
}

verify_checksums() {
  show_status
  if command -v sha256sum >/dev/null 2>&1; then
    (CDPATH= cd -- "$snapshot_dir" && sha256sum -c checksums.sha256)
  elif command -v shasum >/dev/null 2>&1; then
    (CDPATH= cd -- "$snapshot_dir" && shasum -a 256 -c checksums.sha256)
  else
    printf 'Neither sha256sum nor shasum is available.\n' >&2
    return 1
  fi
}

mount_results() {
  if ! command -v sshfs >/dev/null 2>&1; then
    printf 'sshfs is not installed or is not available in PATH.\n' >&2
    return 1
  fi
  mkdir -p "$mount_dir"
  chmod 700 "$mount_dir"
  if is_mounted; then
    show_status
    return 0
  fi
  if [ -n "$(find "$mount_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    printf 'Mount directory is not empty: %s\n' "$mount_dir" >&2
    printf 'Move its contents or set WACKYWACKY_REMOTE_RESULTS_MOUNT.\n' >&2
    return 1
  fi
  sshfs \
    "$remote_results" \
    "$mount_dir" \
    -o ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
  show_status
}

unmount_results() {
  if ! is_mounted; then
    printf 'Remote results are not mounted at %s\n' "$mount_dir"
    return 0
  fi
  umount "$mount_dir"
  printf 'Remote results unmounted from %s\n' "$mount_dir"
}

case "$command" in
  mount)
    mount_results
    ;;
  status)
    show_status
    ;;
  verify)
    verify_checksums
    ;;
  unmount)
    unmount_results
    ;;
  *)
    printf 'Usage: %s [mount|status|verify|unmount]\n' "$0" >&2
    exit 2
    ;;
esac
