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
2. **Reachable over LAN** (both interfaces serve the auth gate → `401`):
   ```bash
   curl -ks --resolve radar.localhost:443:192.168.1.5 -o /dev/null -w '%{http_code}\n' https://radar.localhost/   # 401
   curl -ks --resolve radar.localhost:443:192.168.1.7 -o /dev/null -w '%{http_code}\n' https://radar.localhost/   # 401
   ```
3. **Decide** the feedback channel (email / Slack / WhatsApp) and the test window.
4. **Send each tester:** the message in Part 2 (fill the placeholders) and, to
   avoid the certificate warning, the file `deploy/tls/caddy-root.crt`.
5. **Attach** [`TESTERS.md`](TESTERS.md) for the detailed walkthrough (optional).
6. **Expect queues:** one GPU analysis at a time; a second tester waits.

Current host addresses (verify with `ip -4 -o addr show`):

| Interface | Address | Notes |
|---|---|---|
| `eno1` (ethernet, default route) | **192.168.1.5** | primary — use this in the invite |
| `wlo1` (Wi-Fi) | 192.168.1.7 | works too; same LAN |

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
> ### Get set up (one time, ~3 minutes)
>
> 1. Be on the same network as the pilot machine (or its VPN).
> 2. Add this line to your hosts file so the address resolves:
>    ```
>    192.168.1.5   radar.localhost
>    ```
>    - Windows: `C:\Windows\System32\drivers\etc\hosts` (edit as Administrator)
>    - macOS/Linux: `/etc/hosts` (needs `sudo`)
>    - *(If the machine is on Wi-Fi instead, use `192.168.1.7`.)*
> 3. Open **https://radar.localhost/**
>    - The first visit shows a certificate warning (we use a private certificate
>      for this pilot). Either click **Advanced → Proceed**, or install the
>      attached `caddy-root.crt` once (steps in the follow-up note).
> 4. Log in with the browser prompt:
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

## Part 4 — Optional: remove the hosts-file requirement

Testers currently must add the `radar.localhost` hosts entry because the Caddy
site block is bound to that hostname (a bare-IP URL fails the TLS handshake).
To let testers open `https://192.168.1.5/` directly (clicking through the cert
warning, no hosts edit), add the LAN IP as a second site address in
`deploy/Caddyfile`:

```caddyfile
{$CADDY_DOMAIN}, https://192.168.1.5 {
  tls internal
  ...
}
```

Then `docker compose -f deploy/docker-compose.yml up -d caddy`. Ask the
maintainer to apply this if testers can't edit their hosts file.
