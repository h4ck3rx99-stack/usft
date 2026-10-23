# Hardware testing procedure

> **Status: not yet performed.** None of the measurements below have been run on real
> hardware by the developers of this version. Every performance figure elsewhere in the
> repository comes from the channel *simulator* and does not prove acoustic compatibility
> of any particular laptop, microphone or speaker. Please fill in the tables and submit
> them with an issue or pull request.

## What you need

* Two computers (A and B) with Python 3.11–3.13 and this package installed
  (`pip install -e .`), each with a working speaker **and** microphone (built-in or USB).
  Avoid Bluetooth audio devices.
* A tape measure; a quiet room; optionally a source of moderate noise (fan, music at low
  level, a second person talking).
* Optional: a sound-level meter app.

## Preparation (both machines)

1. Windows: disable audio enhancements for the microphone and speaker
   (Settings → System → Sound → device → *Audio enhancements: Off*; or the Sound Control
   Panel → Properties → *Enhancements*/*Advanced*), set both devices to 48 000 Hz
   (Sound Control Panel → Properties → *Advanced* → *Default Format*), and allow desktop
   apps to use the microphone (Settings → Privacy & security → Microphone).
2. Close other apps that use audio. Set the system output volume to ~70 %.
3. `airdrop devices` — note the device indices and host API (prefer *Windows WASAPI*).
4. Run the single-machine self-check and keep its output:

   ```
   airdrop check --in <mic> --out <speaker>            # 48 kHz
   airdrop check --in <mic> --out <speaker> --rate 96000 --exclusive   # optional, for the US profile
   ```

   Record the verdict per profile, the response in the NU band, and whether a
   brick-wall drop was detected. Repeat with `--exclusive` if a brick wall is reported.

## Link measurements

For each distance (0.3 m, 1 m, 2 m; speakers facing each other, same table) and each
environment (quiet room; moderate noise):

1. Link test (20 s): on B `airdrop linktest receive`, on A `airdrop linktest send`.
   Record the median carrier SNR, MFSK SNR, ppm offset and recommended mode.
2. Per-mode frame error rate with the measurement harness:

   ```
   # B (receiver)
   python tools/measure_link.py receive --csv results.csv --label "1 m quiet, A->B, <devices>"
   # A (sender)
   python tools/measure_link.py send --modes 0,1,2,3,4,5 --frames 20
   ```

   The receiver prints a per-mode summary table and appends one row per detected frame
   to the CSV (columns: label, time, mode, frame, ok, snr_db, mfsk_snr_db, ppm, cfo_hz,
   detect_ratio, rs_corrected, erasures). Repeat with A and B swapped (the two directions
   often differ).
3. A real transfer: a 2 KB text file and a 20 KB compressed-poorly file (e.g. a JPEG):

   ```
   B: airdrop receive --out ./rx
   A: airdrop send small.txt -y        # note the estimated vs actual time
   ```

   Record goodput (shown at the end), retransmitted units, final mode, and confirm the
   received file's SHA-256 (the app verifies it; `sha256sum`/`Get-FileHash` to double-check).
4. Broadcast: `airdrop send photo.jpg --broadcast -y` with `airdrop receive --out ./rx
   --accept-all` on B; record which files completed.
5. Optional audibility: ask people of different ages whether they hear the NU profile at
   −6 dBFS and at −12 dBFS.

## Results template

### Devices

| Machine | Model | OS | Mic (host API) | Speaker (host API) | Enhancements off? | Exclusive mode? |
|---|---|---|---|---|---|---|
| A | | | | | | |
| B | | | | | | |

### Self-check

| Machine | Rate | NU verdict | NU response (dB vs 2–4 kHz) | NU band SNR (dB) | AU verdict | Brick wall (kHz) | Notes |
|---|---|---|---|---|---|---|---|
| A | 48k | | | | | | |
| B | 48k | | | | | | |

### Link test / measurement harness

| Direction | Distance | Environment | Profile | Median carrier SNR (dB) | MFSK SNR (dB) | ppm | Mode 0 FER | Mode 1 FER | Mode 2 FER | Mode 3 FER | Mode 4 FER | Mode 5 FER | Recommended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A→B | 0.3 m | quiet | nu | | | | | | | | | | |
| A→B | 1 m | quiet | nu | | | | | | | | | | |
| A→B | 2 m | quiet | nu | | | | | | | | | | |
| A→B | 1 m | noise | nu | | | | | | | | | | |
| B→A | 1 m | quiet | nu | | | | | | | | | | |

### Transfers

| Direction | Distance | Environment | File | Size | Estimated time | Actual time | Goodput (B/s) | Final mode | Retransmitted units | Verified? |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

## Interpreting results

* Mode 0 failing at short range almost always means the band is being filtered:
  re-run `airdrop check` (look for a brick wall), disable enhancements, try exclusive mode,
  or switch both sides to `--profile au`.
* OFDM modes (2–5) need a high direct-to-reverberant ratio. In simulation they fail at
  RT60 0.3 s when the reverberant energy is within ~6 dB of the direct sound; at short
  distances (higher direct-to-reverberant ratio) they should work. Expect MFSK at 1–2 m in
  normal rooms.
* Large asymmetries between directions usually come from different speaker/mic quality.
  Broadcast mode works when only one direction is usable.
