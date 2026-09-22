#!/bin/sh
# Prints a Caddy bcrypt hash for basic-auth, e.g.:
#   deploy/hash-password.sh 'change-me'
# Put the output into deploy/.env as RADAR_AUTH_HASH.
set -eu
PASSWORD="${1:?usage: hash-password.sh <password>}"
docker run --rm caddy:2 caddy hash-password --plaintext "$PASSWORD"