#!/usr/bin/env bash
# One-time setup for the A100 machine: run a Rex server that joins the owner's Tailscale
# network as its own device, so the OVH server and firstmate can reach it with `rex -S a100 ...`.
# Runs as an ordinary user: no root, no tailscaled, no open ports. Safe to re-run.
#
#   scripts/a100_connect_back.sh --auth-key-file ~/ts-key.txt     # key from a file
#   printf '%s' "$KEY" | scripts/a100_connect_back.sh --auth-key-file -   # key from stdin
#   scripts/a100_connect_back.sh --dry-run ...                    # print every command, run none
#
# Options: --hostname NAME (default kahlus-a100), --auth-key-file FILE|-, --no-systemd, --dry-run.
# Access stays limited to the Tailscale account that owns the key: no --allow-anonymous, no tags.
# The key is never printed or logged. Its temp copy is deleted once the server has signed in.
set -euo pipefail

HOSTNAME_TS=kahlus-a100 KEY_SRC="" DRY=0 USE_SYSTEMD=auto
while (($#)); do
  case $1 in
    --hostname) HOSTNAME_TS=${2:?--hostname needs a name}; shift 2 ;;
    --auth-key-file) KEY_SRC=${2:?--auth-key-file needs a file or -}; shift 2 ;;
    --no-systemd) USE_SYSTEMD=no; shift ;;
    --dry-run) DRY=1; shift ;;
    -h | --help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

BIN=$HOME/.local/bin
REX=$BIN/rex
STATE=${XDG_STATE_HOME:-$HOME/.local/state}/kahlus-a100-connect
KEY_COPY=$STATE/authkey
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
UNIT=$UNIT_DIR/rex-a100.service
PIDFILE=$STATE/server.pid
LOG=$STATE/server.log

say() { printf '%s\n' "$*"; }
run() {
  if ((DRY)); then printf '+ %s\n' "$*"; else "$@"; fi
}

server_args() { # $1 = "with-key" adds the one-time key flag; the key path is never echoed
  local a="--tailscale --tailscale-hostname $HOSTNAME_TS"
  [[ $1 == with-key ]] && a+=" --tailscale-auth-key-file $KEY_COPY"
  printf '%s' "$a"
}

unit_text() {
  cat <<UNITEOF
[Unit]
Description=Rex server with built-in Tailscale (kahlus A100 connect-back)

[Service]
ExecStart=%h/.local/bin/rex server run $(server_args "$1")
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
UNITEOF
}

[[ $HOSTNAME_TS =~ ^[a-zA-Z0-9-]+$ ]] || { echo "invalid --hostname" >&2; exit 2; }

# 1. Install Rex for this user only.
export PATH=$BIN:$PATH
if [[ -x $REX ]]; then
  say "rex already installed, skipping install"
else
  run sh -c 'curl -fsSL https://pkgs.superlogical.com/rex/install.sh | sh -s -- --user'
fi

# 2. Stage the one-time auth key privately (never in the repo, never printed).
run mkdir -p "$STATE"
run chmod 700 "$STATE"
HAVE_KEY=0
if [[ -n $KEY_SRC ]]; then
  HAVE_KEY=1
  if ((DRY)); then
    say "+ stage auth key into private temp copy (path hidden, mode 600)"
  else
    umask 077
    if [[ $KEY_SRC == - ]]; then cat >"$KEY_COPY"; else cp -- "$KEY_SRC" "$KEY_COPY"; fi
    [[ -s $KEY_COPY ]] || { rm -f "$KEY_COPY"; echo "auth key is empty" >&2; exit 1; }
  fi
fi

if [[ $USE_SYSTEMD == auto ]]; then
  if systemctl --user show-environment >/dev/null 2>&1; then USE_SYSTEMD=yes; else USE_SYSTEMD=no; fi
fi
FLAVOR=$([[ $HAVE_KEY == 1 ]] && echo with-key || echo no-key)

# 3. Start the server.
if [[ $USE_SYSTEMD == yes ]]; then
  say "mode: systemd user service rex-a100.service"
  run mkdir -p "$UNIT_DIR"
  if ((DRY)); then
    say "+ write $UNIT:"
    unit_text "$FLAVOR" | sed 's/ --tailscale-auth-key-file .*/ --tailscale-auth-key-file <hidden>/; s/^/    /'
  else
    unit_text "$FLAVOR" >"$UNIT"
  fi
  # Rex's own installer may have enabled a plain rex.service that owns the socket.
  run systemctl --user disable --now rex.service 2>/dev/null || true
  run systemctl --user daemon-reload
  run systemctl --user enable rex-a100.service
  run systemctl --user restart rex-a100.service
  if ! run loginctl enable-linger "$USER"; then
    say "linger NOT enabled (not permitted). The server stops when you log out and"
    say "does not start at boot. Ask an admin to run: sudo loginctl enable-linger $USER"
    say "or keep a login session open (for example inside tmux) while the job runs."
  fi
else
  say "mode: nohup fallback (systemd user services unavailable). The server does NOT"
  say "restart after a reboot or crash. Re-run this script to start it again."
  if [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    say "server already running (pid $(cat "$PIDFILE")), not starting another"
  elif ((DRY)); then
    say "+ nohup $REX server run $(server_args "$FLAVOR" | sed 's/ --tailscale-auth-key-file .*/ --tailscale-auth-key-file <hidden>/') >$LOG 2>&1 & echo \$! >$PIDFILE"
  else
    # shellcheck disable=SC2046 # server_args is a fixed, space-separated flag list
    nohup "$REX" server run $(server_args "$FLAVOR") >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
  fi
fi

# 4. Wait for sign-in, then delete the key copy and drop it from the unit.
ADDR=""
if ((DRY)); then
  say "+ wait for 'rex server status' to show the tailnet address, then delete the key copy"
else
  for _ in $(seq 30); do
    ADDR=$("$REX" server status 2>&1 | grep -oE 'https://[A-Za-z0-9.-]+\.ts\.net' | head -n1 || true)
    [[ -n $ADDR ]] && break
    sleep 2
  done
  if [[ -n $ADDR ]]; then
    rm -f "$KEY_COPY"
    if [[ $USE_SYSTEMD == yes && $HAVE_KEY == 1 ]]; then
      unit_text no-key >"$UNIT"
      systemctl --user daemon-reload
    fi
  elif ((HAVE_KEY)); then
    say "WARNING: no tailnet address yet. Key copy kept in the private state dir; check 'rex server status'."
  fi
fi

# 5. Report.
run "$REX" server status || true
say "Device address: ${ADDR:-https://$HOSTNAME_TS.<tailnet>.ts.net}"
say "From the OVH server: rex hosts add a100 https://$HOSTNAME_TS.<tailnet>.ts.net && rex -S a100 ls"
