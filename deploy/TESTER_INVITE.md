# Tester Invite Pack — RADAR Kenya CT Review Pilot

Everything needed to onboard testers: an admin checklist, a paste-ready invite
message, certificate-trust steps per OS, and feedback asks.

> **Never commit the pilot password.** It is shared out-of-band with each tester
> (the invite below uses a `<PASSWORD>` placeholder). The bcrypt hash lives in
> the git-ignored `deploy/.env`; rotate it any time with `deploy/hash-password.sh`.

---

## Part 1 — Admin checklist (before sending)

1. **Stack healthy:**
   ```bash
   docker compose -f deploy/docker-compose.yml ps      # webapp + caddy = healthy
   ```
2. **Reachable over LAN** (each address serves the auth gate → `401`):
   ```bash
   # list the host's current addresses first
   ip -4 -o addr show | grep -v ' lo \| docker0\| br-'
   # then check each one (replace the IPs with the current ones)
   docker compose -f deploy/docker-compose.yml exec -T caddy \
     wget -q -O /dev/null --no-check-certificate --server-response https://192.168.1.7/ 2>&1 | grep HTTP
   ```
3. **Decide** the feedback channel (email / Slack / WhatsApp) and the test window.
4. **Send each tester:** the message in Part 2 (fill the placeholders) and, to
   avoid the certificate warning, the file `deploy/tls/caddy-root.crt`.
5. **Attach** [`TESTERS.md`](TESTERS.md) for the detailed walkthrough (optional).
6. **Expect queues:** one GPU analysis at a time; a second tester waits.

**Addresses change** (DHCP, Wi-Fi vs ethernet, USB adapters), so re-check before
sending. Update `CADDY_SITE_ADDRESSES` in `deploy/.env` and run
`docker compose -f deploy/docker-compose.yml up -d --force-recreate caddy`
whenever they change. Checked on 2026-09-27:

| Interface | Address | Notes |
|---|---|---|
| `enx68e43b307bcd` (USB ethernet, default route) | **192.168.1.9** | primary |
| `wlo1` (Wi-Fi) | **192.168.1.7** | also works |
| `radar.localhost` | 127.0.0.1 | **only on the host itself** — it never resolves for other devices |

---

## Part 2 — Paste-ready invite message

> **Subject: Invitation — RADAR abdominal CT review pilot (research)**
>
> Hi <NAME>,
>
> You're invited to a **research pilot** of RADAR, an abdominal CT review
> workspace. It shows model finding scores, organ segmentation overlays, and a
> structured report draft so we can evaluate how useful it is for review.
>
> **Important:** this is a research prototype, **not a medical device**, and it
> is not for diagnosis. Please do **not** upload real patient data — use the
> built-in example case and the test scans we provide.
>
> ### Get set up (one time, ~1 minute)
>
> 1. Be on the same network as the pilot machine (or its VPN).
> 2. Open **https://192.168.1.9/** in your browser *(if that does not load, try
>    `https://192.168.1.7/`)*.
>    - The first visit shows a certificate warning (we use a private certificate
>      for this pilot). Either click **Advanced → Proceed**, or install the
>      attached `caddy-root.crt` once (steps in the follow-up note).
>    - *Optional:* add `192.168.1.9  radar.localhost` to your hosts file
>      (Windows: `C:\Windows\System32\drivers\etc\hosts` as Administrator;
>      macOS/Linux: `/etc/hosts` with `sudo`) and then use
>      **https://radar.localhost/** instead. `radar.localhost` only means
>      "this device", so it needs that entry on your machine.
> 3. Log in with the browser prompt:
>    - **User:** `radiologist`
>    - **Password:** `<PASSWORD>`
>
> ### What to try (~20 minutes)
>
> 1. Click **Open example case** — explore the three planes, the window presets
>    (Abdomen / Liver / Bone), the organ overlay, and slice stepping.
> 2. **Findings** tab — search a term (e.g. `cyst`), filter by anatomy, move the
>    score threshold, export the CSV.
> 3. **Review & export** — mark a finding `Likely present`, add clinical context
>    and a note, set the review status, and read the generated report draft
>    (Clinical context / Findings by organ / Impression / Review limitations).
>    Click **Save case to history**.
> 4. Switch **View → Case worklist** — find your case, open it, download the
>    report TXT, and export the worklist CSV.
> 5. **Validation** tab — paste a short radiologist report snippet,
>    **Extract findings**, confirm the present/absent list, and look at the
>    agreement metrics.
> 6. If you have a test scan (we'll provide one), upload it and click
>    **Analyze scan** (~30 seconds on the GPU).
>
> ### What to expect
>
> - Analysis takes ~30 s per scan; if someone else is analyzing, yours waits.
> - Model scores are **not disease probabilities** — they compare predefined
>   prompts and always need human judgement.
> - The viewer is for review support; orientation is not validated for
>   diagnostic use.
>
> ### Please tell us
>
> - Anything that **crashes** or gets stuck (include the case name).
> - **Wrong or surprising** organ labels, overlays, or scores.
> - Report text that reads **confusingly or is missing** something you'd expect.
> - What you'd need before you'd use this in a real workflow.
>
> Send feedback to **<FEEDBACK CHANNEL>** any time before **<DATE>**.
>
> Thanks!
> <YOUR NAME>

---

## Part 3 — Trusting the certificate (optional, removes the warning)

Send `deploy/tls/caddy-root.crt` with the invite. Per OS:

**Windows** — double-click the `.crt` → *Install Certificate…* → *Local Machine*
→ *Place all certificates in the following store* → *Trusted Root Certification
Authorities* → Finish. Restart the browser.

**macOS** — double-click to add to Keychain → find *Caddy Local Authority* →
*Get Info* → *Trust* → **Always Trust**. Restart the browser.

**Ubuntu/Debian (Chrome/Edge)** — uses the system store:
```bash
sudo cp caddy-root.crt /usr/local/share/ca-certificates/caddy-radar.crt
sudo update-ca-certificates
```
Then fully restart the browser (`chrome://restart`).

**Firefox (any OS)** — separate store: *Settings → Privacy & Security →
Certificates → View Certificates… → Authorities → Import…* → select the `.crt`
→ tick *Trust this CA to identify websites*.

Only trust this CA on machines you control. Deleting the `radar-ke_caddy-data`
volume regenerates a new CA (re-extract with the command in `deploy/README.md`).

---

## Part 4 — Addresses and hosts file (already handled)

LAN devices cannot use `radar.localhost` — that name is reserved to mean "this
device", so it only resolves on the host itself. Caddy is therefore configured
to answer on the host's LAN IPs too, via `CADDY_SITE_ADDRESSES` in
`deploy/.env`:

```
CADDY_SITE_ADDRESSES=radar.localhost, https://192.168.1.9, https://192.168.1.7
```

Testers just open one of the IP addresses (accepting the certificate warning
once); no hosts-file edit is required. If the host's addresses change (DHCP,
Wi-Fi vs ethernet, USB adapters), re-check with `ip -4 -o addr show`, update
`CADDY_SITE_ADDRESSES`, and run
`docker compose -f deploy/docker-compose.yml up -d --force-recreate caddy`.

For the cloud path (B), set `CADDY_SITE_ADDRESSES` to the real domain only and
delete the `tls internal` line in `deploy/Caddyfile`.
