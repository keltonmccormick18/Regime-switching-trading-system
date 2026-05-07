#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Generate a self-signed TLS certificate for local HTTPS.
#  Output: certs/server.key  (private key)
#          certs/server.crt  (certificate, valid 825 days)
#
#  The cert covers:  localhost  127.0.0.1  ::1
#  Run once; re-run any time you want to rotate.
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="$SCRIPT_DIR/certs"
KEY="$CERT_DIR/server.key"
CRT="$CERT_DIR/server.crt"
CNF="$CERT_DIR/openssl.cnf"

mkdir -p "$CERT_DIR"

# Skip if certs already exist and are not expiring within 7 days
if [[ -f "$KEY" && -f "$CRT" ]]; then
  if openssl x509 -checkend 604800 -noout -in "$CRT" 2>/dev/null; then
    echo "  Certs already valid — skipping generation."
    echo "  Key: $KEY"
    echo "  Crt: $CRT"
    exit 0
  else
    echo "  Cert expiring soon — regenerating..."
  fi
fi

# Write a minimal openssl config with Subject Alternative Names
cat > "$CNF" <<EOF
[req]
default_bits       = 2048
prompt             = no
default_md         = sha256
distinguished_name = dn
x509_extensions    = v3_req

[dn]
C  = US
ST = Local
L  = Local
O  = QuantTradingSystem
CN = localhost

[v3_req]
subjectAltName = @alt_names
keyUsage       = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = localhost
IP.1  = 127.0.0.1
IP.2  = ::1
EOF

openssl req \
  -x509 -nodes \
  -days 825 \
  -newkey rsa:2048 \
  -keyout "$KEY" \
  -out    "$CRT" \
  -config "$CNF" \
  2>/dev/null

chmod 600 "$KEY"

echo ""
echo "  ✓ TLS certificate generated"
echo "    Key : $KEY"
echo "    Cert: $CRT"
echo ""
echo "  To trust this cert system-wide (removes browser warnings):"
echo "    macOS : sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain \"$CRT\""
echo "    Linux : sudo cp \"$CRT\" /usr/local/share/ca-certificates/quant-trading.crt && sudo update-ca-certificates"
echo ""
