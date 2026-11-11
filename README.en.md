# bugnav — External Anti-Stuck Navigation for the Ecovacs T80s

English | [中文](README.md)

An external "brain" for the Ecovacs DEEBOT T80S (plumbed Omni dock
version). The robot itself is untouched: the program runs on an
always-on host and, acting as another app client under the user's own
account, talks to the Ecovacs cloud (via deebot-client) to read
position/state/battery and issue commands. Three layers:

- **Control layer (deterministic, live on every 2.5 s sample)**:
  anti-stuck detection, wall-following escape, backtracking — fully
  decoupled from the judgment layer and always on;
- **Judgment layer (LLM session supervisor + per-episode advisor)**:
  code only measures and does arithmetic (speed, area, battery,
  charge curves, projected finish time); every conclusion — is this
  reasonable, continue or wrap up, charge to how much — is drawn by
  the LLM from explicit judgment logic, never from hardcoded
  threshold rules;
- **Charge planning (deterministic solver)**: computes the balance
  point between "finish everything in one more outing" and "split
  across several charge cycles" by total elapsed time, for the
  supervisor to ratify, override, or veto.

Overall objective: **minimize total elapsed time of cleaning +
charging + dock overheads** — a task started at 07:00 must not still
be cleaning at 20:00.

**Full design document (hardware → software design → deployment →
operations → verification checklist → rollback), in Chinese:
[方案.md](方案.md).**

## Control layer: anti-stuck

- **Struggle-precursor detection**: the stock firmware keeps climbing
  onto an obstacle until a drive wheel hangs and it errors out. This
  layer catches the precursor instead — repeated back-and-forth in a
  small area, long path with ≈0 net displacement, repeated heading
  reversals — and takes over before any error code appears.
- **Wall-following exploration (bug algorithm)**: after takeover it
  traces the obstacle's contour, trying approach directions around a
  staging ring in 30° steps; it slips through any gap it finds and
  remembers the way in (breadcrumb trail), backtracking along it if
  it cannot get out. Attempts are counted by distinct approach
  direction — charging the same direction ten times is one attempt.
  Small obstacle fully circled with a gap found = success; large
  obstacle fully circled with no gap = give up for this session.
- **The way out is explored too**: backtracking along the breadcrumbs
  is only the baseline route — it is never assumed to still be clear.
  If a new obstacle blocks it en route (struggle, or a waypoint
  timing out with no progress), the robot switches into the same
  wall-following detour. The return objective is prioritised:
  **first choice is the original entrance** (the area exit at the
  far end of the trail); if half a circle of exploration still
  cannot regain it, the objective degrades to **leaving this area**
  — out through any opening on the far side of the obstacle, able
  to resume work, without fixating on the original exit. At any
  moment during the process, if the exit corridor can be rejoined
  (at *any* remaining breadcrumb), rejoining wins and the exit
  continues along the original path.
  Only when that exploration has circled completely and can neither
  regain the exit nor leave the area at all is the robot declared
  **trapped**: it stops and raises the alarm for a human rescue —
  the notification goes to phone/Mac *and the robot itself sounds
  its built-in alert tone three times*, so someone at home hears it
  without looking at a phone — and stays hands-off until a person
  visibly intervenes (robot moved by hand, or back at the dock).
- **Looping/loitering detection**: circling inside a small disc
  (long path, large cumulative turn, small net displacement) triggers
  takeover into wall-following on first occurrence, before it turns
  into a struggle.
- **Layered memory**: a surrendered spot is blacklisted only for the
  current session (TTL) and cleared when the session ends — chairs
  move, so nothing is ever hardened into a permanent no-go zone.
  Across sessions only grid statistics persist (where it gets stuck,
  which direction worked), never prohibitions.
- **Error codes**: 105 (trapped), 102/204 (drive wheel suspended),
  etc. feed the supervisor's view and counters directly.

## Adapter layer: talking to the real robot

- deebot-client does not know this hardware class (nazy54); at
  runtime it is aliased to a same-generation V2 capability set.
  Firmware 1.61.0 rejects `getCleanInfo` (20003), so work state is
  read via `getWorkState` instead (the command is implemented
  locally to the pinned deebot-client version's spec and registered
  in its message dispatch table), and the aliased capability set is
  patched accordingly.
- Subscribed: position/heading, work state + dock state, battery,
  water spray level + mop attachment, error stream. Commands:
  pause/stop/resume, return-to-dock, area cleaning (V2).
- Credentials are injected only via process/container environment
  variables, never written into code or config files; the device_id
  is fixed so repeated logins do not trigger new-device
  verification.

## Judgment layer: LLM supervision

### Advisor (inside an episode)

At the fixed strategy's weak points (an episode failing repeatedly,
just before surrendering a large obstacle) the controller consults
the LLM with a bounded plan vocabulary (staging radius/step/budget
adjustments). The LLM never supplies coordinates; on timeout the
fixed strategy resumes — the LLM can never stall the robot.

### Supervisor (whole session)

Reviews on a cadence (about every 20 min of cleaning, always when
docked for charging, sooner on anomalies), with a bounded verdict
vocabulary: `continue / notify / abandon_current / stop_session /
ask_user`, plus an optional resume-charge override and charge-model
calibration (below). In observe mode verdicts are logged as shadow
decisions only; in active mode they execute. `stop_session` means
stop **and** return to dock — the robot is never left sitting
wherever it stopped.

### What it is given (code measures; code does not conclude)

- **Rolling gauges**: drain rate (%/m²), progress pace (m²/min) and
  newly covered area over the last ~15 minutes; whole-bout averages
  as fallback.
- **Projected finish**: remaining area ÷ current pace, plus the
  charge cycles implied by the energy shortfall (each priced with
  the charge curve + per-cycle overhead), yielding a projected
  finish clock time, minutes remaining, and the **total session
  span** (elapsed + waiting + remaining).
- **Virtual water gauge**: the robot has no water level sensor —
  the gauge refills to 100% only after a real station stay with a
  wash (dock dwell past a threshold), then drains by spray level ×
  area mopped since.
- **Per-segment ledger**: every "undock → dock" leg is booked
  separately (minutes, area, start/end battery, whether it left
  after charging), making short-cycle charge thrashing (charge,
  resume, flat again in minutes, repeat) diagnosable: low departure
  charge = charged too little; high departure charge yet quickly
  flat = abnormal drain.
- **Pace-stall flag**: a rolling window with almost zero new
  coverage while "cleaning" — independent of total area, valid in
  any home, including an unfamiliar one.

### Its judgment logic (in the prompt: guidance, not dead rules)

- **Duration reasonableness is derived from area**: the yardstick
  is the total time the planned area *should* take at a reasonable
  speed; the ratio of actual+projected span to that expectation
  judges **efficiency**. No "N hours" cutoff — a large home taking
  long with a proportionate ratio is simply normal.
- **Complexity is priced in**: expectations come in an optimistic
  and a conservative flavour — the conservative one uses the pace
  this environment has actually demonstrated (which inherently
  includes obstacle detour costs). Area occupied by obstacles and
  never cleanable is deducted from the planned area first
  (cleanable area); coverage, remaining area and projections are
  all computed against the cleanable area. Time spent probing
  around obstacles is metered separately ("low-yield time share"):
  ≈0 progress while probing is legitimate, and the probing-free
  pace is never used to judge the whole.
- **The total-span bottom line outranks the ratio**: whatever the
  speed and however "justified" the slowness, total cleaning time
  must not be stretched by charge waiting to an unacceptable extent
  (spanning most of a day, pushed into the night) — complexity can
  explain slow; it cannot explain endless. If a session truly
  cannot finish because the environment is too complex, it must
  notify and say so: projected finish time, where the time went,
  and what it intends to do. No silent endless cleaning, no silent
  surrender either.
- **Charging is a balance, not a mandate**: one end is "finish all
  remaining area in one outing" (its required departure charge,
  computed from the drain estimate plus a safety margin, is called
  the guarantee figure — a price tag, **not** a floor that must be
  reached); the other end is splitting across cycles (each extra
  cycle costs one dock overhead). Every candidate charge level is
  priced: minutes of waiting, clock time it can leave, area it
  buys. Little remaining → one-shot usually wins; lots remaining →
  insisting on one-shot can blow up the total span. The only floor:
  never fragment into short-cycle thrash — every outing must clean
  a worthwhile area.
- **Full coverage is not owed**: coverage is judged against the
  cleanable area; after long cleaning with the bulk done, the
  scraps are not worth another charge cycle — wrapping up is the
  reasonable choice. Time already spent is sunk cost, not a reason
  to continue.
- **Daily routine**: resumption decisions weigh the current time
  against ordinary living hours; a morning task still resuming at
  16:00–17:00 is a strong wrap-up signal (unless the ratio proves
  the area genuinely needs that long).

### Charge parameters: continuously calibrated by the AI from raw history

Charging is nonlinear (fast at low SOC, slow in the taper — e.g.
30%→50% might take 10 minutes while 50%→80% takes 30), and the true
curve drifts. Code only records raw history: every dock stay is
stored as a trajectory (start SOC, duration, a mark every +2% SOC),
persisted long-term and included in every review digest. The
supervisor compares the measurements against the parameters in use
(segmented band rates, drain per m²) and, when they disagree,
returns corrected parameters in its verdict; the system validates
the shape, adopts them, and persists them. Code prescribes no
correction formula of its own.

### Map-cache consistency

The whole-home area baseline comes only from an *adopted* layout:
when a session's footprint mismatches the baseline (too much novel
area — new map, different rooms, taken to another home), the
baseline is flagged suspect and the digest **withholds** the derived
figures (remaining area, coverage %, projected finish) so no
conclusion is drawn from a wrong denominator; raw progress and
durations still flow. A mismatching layout is stashed as a
candidate and adopted as the new baseline only when it repeats for
a second session — one trip elsewhere never poisons the baseline.
Progress percentage exists only once a baseline is mature.

### Ask the user (ask_user)

When continuing vs wrapping up is a genuine toss-up, the supervisor
can hand the decision to the user: it pauses the robot (or keeps it
docked), sends a notification stating the situation and the
choices; within 5 minutes, the user resuming cleaning (physical
button or in the app) means *continue*, sending it home means
*end*; **no feedback within 5 minutes defaults to return-to-dock
and end**. (The firmware reports no button events; button/app
actions are recognised through state changes — either way it is the
user's own hand.)

### Home profile

Config may carry this home's cleaning scope (e.g. living room,
dining room, kitchen and one study — no bedrooms) and a normal
duration reference: the scope is used as fact; where the reference
duration conflicts with the area-derived expectation, the
area-derived expectation wins.

## Notifications

Supervisor notifications (notify verdicts, abnormal wrap-ups,
ask_user questions, baseline changes, …) are written as `NOTIFY`
lines into `data/events.jsonl` and can be forwarded via webhook.
The Mac host ships a notification bridge (`deploy/mac/`): it tails
the events file → macOS Notification Center + an iMessage to the
user (which also reaches their phone), autostarted via launchd,
with a state file so it neither replays history nor misses lines
while it was down.

## Deployment

- Image: prebuilt on GHCR (`ghcr.io/xnetsc/bugnav`); the host only
  pulls and runs the container and holds small config files — no
  code lives on the host. Mac + Docker Desktop is the current form;
  a systemd installer is also provided.
- Config: `config.json` (account short ID, control/supervisor
  parameters, home profile) + `.env` (Ecovacs credentials; LLM key /
  endpoint / model — any OpenAI-compatible endpoint works; Alibaba
  Qwen `qwen-plus` is verified). All secrets live only in `.env`.
- Data (`data/`): `samples.jsonl` (sample stream, with per-sample
  progress), `events.jsonl` (events and notifications),
  `home_cells.json` (adopted baseline), `home_candidate.json`
  (candidate layout), `charge_model.json` (charge parameters,
  including AI calibrations), `charge_history.json` (raw charge
  trajectories), `stats.json` (cross-session stuck statistics).
- Modes: observe by default (read-only, shadow verdicts); switch
  to active (actually issuing commands) only after the verification
  checklist passes.

Quick start:

```bash
# 1) Watch it work in the simulator (no robot, no account needed)
python3 -m bugnav.demo
python3 -m unittest discover -s tests

# 2) Deploy to the host (observe mode by default; go active only
#    after verification)
cp config.example.json config.json   # fill in the account short ID
sudo bash deploy/install.sh          # or: docker compose up -d
```

Layout: `bugnav/` (core logic + simulator + real-robot adapter),
`deploy/` (systemd and Mac deployment pieces), `tests/` (unit and
three-scenario end-to-end tests), `Dockerfile`, `docker-compose.yml`.

## Verification status (stated honestly)

- Verified on the real robot: cloud login and the position /
  (initial) state / battery / error streams; a full real scheduled
  clean observed end to end; supervisor verdicts served by the real
  LLM endpoint.
- Implemented, pending on-robot verification with the next image
  pull and restart: whether `getWorkState` is accepted by the
  firmware (the state-stream fix), the water-info reading in
  practice (the firmware answers; its customAmount payload shape
  is handled by a local tolerant parser), coordinate scale
  calibration (3 m
  two-point method) and sub-metre area commands, the playSound
  alert tone, the full active mode chain.
- Known ceilings: no custom voice/intercom exists in the protocol
  (built-in sounds only); the camera feed runs over a vendor-only
  session an external program cannot join; dirt detection is
  internal firmware behaviour and emits no events.
