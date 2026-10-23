# Ultrasonic AirDrop

Transfer files between two nearby computers using **only sound**: speaker → air →
microphone. No Wi-Fi, Bluetooth, LAN, USB, cloud or internet is used at runtime.
Windows 11 is the primary target; Linux and macOS work too.

The app is a complete acoustic modem plus a reliable file-transfer protocol:
chirp-synchronised dual-subband MFSK (robust) and differential OFDM (fast)
physical layers, Reed–Solomon FEC with erasures and interleaving, a half-duplex
selective-repeat ARQ with receiver-driven rate adaptation, resume, a one-way
broadcast mode, optional authenticated encryption, a PySide6 GUI and a CLI.

> **Honest status.** Everything has been verified in a sample-accurate channel
> simulator (noise, reverberation, clock offset, frequency offset, frame loss,
> noise bursts, band-limiting, clipping). **It has not been tested between two real
> computers.** Simulation does not prove that a given laptop's speaker and microphone
> can carry 17.5–20 kHz; run the Hardware check and the Link test first, and please
> report results using [docs/HARDWARE_TESTING.md](docs/HARDWARE_TESTING.md).

---

## What to expect

* **It is slow.** Acoustic links between ordinary laptops carry tens to a few hundred
  bytes per second. In simulation the robust MFSK modes deliver about **13–18 B/s** and the
  OFDM modes **~95–225 B/s** (data phase) under good conditions — these are design targets
  for short distance, quiet room and good hardware, *not* measurements. A 100 KB file takes
  **1.5–2 hours** in robust mode and roughly 8–20 minutes in the fastest mode; multi-megabyte
  files take many hours. The app
  shows an airtime estimate before anything is transmitted. Text and other compressible
  files are zlib-compressed automatically, which helps a lot.
* **Distance:** designed for 0.3–2 m. In rooms with strong reverberation the OFDM
  modes stop working well before MFSK does; the link probe measures this and the
  protocol falls back to MFSK automatically.
* **Audibility:** the default band (≈17.5–20 kHz) is inaudible or barely audible to
  most adults, but **children and pets may hear it**, and laptop speakers can produce
  faint clicks/hiss. The *Audible* profile (3–5 kHz) is clearly audible.
* **Privacy:** sound is not private. Anyone nearby with a microphone can record a
  transfer. Use the optional encryption for confidentiality (see below).

## Install

Requires **Python 3.11–3.13**.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -e .[dev]             # or: pip install -e .   (runtime only)
```

On Linux install PortAudio and Qt's system libraries first, e.g.
`sudo apt install libportaudio2 libegl1 libxkbcommon0`. On Windows and macOS the
`sounddevice` wheel bundles PortAudio.

Launch the GUI with `airdrop-gui` or `python -m ultrasonic_airdrop`; the CLI is `airdrop`.
**The app never plays sound when it starts** — audio is produced only after you press
Send, Hardware check, Link test, or Test tone, each of which says what will be played.

### Windows one-click launcher: `AirDrop.bat`

Double-click **`AirDrop.bat`** in the project folder. On first run it finds Python 3.11–3.13,
creates `.venv` and installs everything (needs internet once). It then shows a menu:
open the GUI, **receive**, **send** (with a file picker), **estimate transfer time**,
hardware check, link test (listen/send), list devices, switch profile (near-ultrasonic /
audible), simulate, reinstall. You can also **drag files onto `AirDrop.bat`** to send them
straight away. Press Ctrl+C to stop listening, then answer `N` to "Terminate batch job?" to
return to the menu.

To see how long something will take before sending: `airdrop estimate FILE...`
(menu option 4) — it compresses/hashes the files and prints the time for every mode.

## First use (step by step)

1. **On both machines: Diagnostics → Hardware self-check.** It plays 1 s of silence and
   two 3-second sweeps and reports, for each profile band, the speaker→mic response
   relative to 2–4 kHz, the band SNR, clipping and a verdict (*supported / marginal /
   not supported*). If it reports a "brick wall" drop (e.g. at 16 or 18 kHz), follow the
   Windows audio setup below and run it again.
2. **Link test.** Machine B: Diagnostics → Link test → role *Receiver* → Start.
   Machine A: role *Sender* → Start. B shows live per-carrier SNR bars, frame error rate,
   the clock offset between the machines and the recommended mode. Do it in both
   directions.
3. **Send a small file.** B: *Receive* → choose a folder → *Start listening*.
   A: *Send* → add a small text file → *Send…* → confirm the airtime estimate.
   B sees the offer (sender name, file count, size, estimated time, encrypted?) and clicks
   *Accept*. Watch progress; the receiver verifies the SHA-256 before saving the file.

## Windows audio setup (important)

Windows audio processing is the most common reason the near-ultrasonic band fails:

* **Disable audio enhancements** for both the microphone and the speaker:
  Settings → System → Sound → *(device)* → **Audio enhancements: Off**, or Sound Control
  Panel → device → Properties → *Enhancements*/*Advanced* → disable all enhancements.
  Also turn off vendor suites (Dell MaxxAudio, Realtek/Waves, "Voice Clarity", noise
  suppression, beamforming).
* **Try WASAPI exclusive mode** (Diagnostics page or `--exclusive`): it bypasses the
  shared-mode effects chain on many drivers. If the driver refuses, the app falls back
  to shared mode and logs a warning.
* **Set both devices to 48 000 Hz** (Sound Control Panel → Properties → *Advanced* →
  *Default Format*); in shared mode Windows resamples, which can low-pass the signal.
* **Microphone permission:** Settings → Privacy & security → Microphone → *Let desktop
  apps access your microphone*.
* Close applications that hold the audio device (conferencing tools, browsers).
* **Avoid Bluetooth audio devices** — their codecs remove content above ~16 kHz.

## Profiles

| Profile | Band | Rate | When to use |
|---|---|---|---|
| Near-ultrasonic (default) | chirps 17–20 kHz, data 17.5–19.9 kHz | 48 kHz | Most laptops whose hardware check passes |
| Audible compatibility | 2.5–5.5 kHz | 48 kHz | Hardware that fails near-ultrasonic; audible |
| Ultrasonic (experimental) | 20.5–23.5 kHz | 96 kHz | Only if the 96 kHz hardware check passes (enabled in the GUI after a passing check) |
| **Fast close-range** (`fa`) | 1.5–20.6 kHz, 375 Hz carrier spacing, 4 ms symbols | 48 kHz | Laptops **side by side** (~10–30 cm). Loudly audible hiss. ~2 KB/s simulated: **1 MB in ~8 min** |

The fast profile spreads the same 52 carriers over ~19 kHz with 6.7× the symbol rate. Its 1.3 ms
guard interval only tolerates a strong direct path, so use it with the laptops next to each other
(sender's speakers facing the receiver's microphone); at room distances the probe falls back to
its MFSK modes. The receiver listens for all 48 kHz profiles at once, so nothing needs to be
selected on the receiving side.

Theory vs practice: a 48 kHz stream can represent up to 24 kHz, but DAC/ADC
anti-aliasing filters roll off around 20–22 kHz, laptop micro-speakers typically fall off
steeply above 16–18 kHz, and MEMS microphones vary widely above 18 kHz. True ultrasound
(> 20 kHz) therefore needs 96 kHz-capable hardware that actually passes the check. The
receiver listens for the near-ultrasonic and audible profiles simultaneously and
follows whichever the sender uses.

## How it works (summary)

Full specification: **[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

* **Frames:** up/down chirp preamble (detection, timing, frequency-offset estimate), a
  4-symbol sync word, a PHY header (mode, carrier mask, length; RS-protected, CRC-16), the
  body, and a postamble chirp whose position gives the sample-rate mismatch between the
  two machines (corrected by resampling each frame).
* **Modes:** 0–1 = 16-ary MFSK in two subbands (1 byte/symbol, 37.5 symbols/s) with
  RS(nsym 64/32); 2–5 = differential OFDM (DBPSK/DQPSK) on up to 52 carriers with pilots and
  a per-link carrier mask. Control frames always use mode 0.
* **Reliability:** RS with erasures and cross-codeword interleaving, CRC-32 on every link
  frame, SHA-256 on every file. Windows of frames are acknowledged with a 64-bit selective
  bitmap; lost frames are retransmitted first; a poll recovers lost ACKs.
* **Rate adaptation:** a PROBE frame measures per-carrier SNR; the receiver recommends a
  mode and carrier mask, and keeps updating its suggestion in every ACK. The sender steps
  down at once on > 25 % frame loss or two missed ACKs and steps up one mode after three
  clean windows if the receiver agrees. Thresholds come from simulated FER curves
  (**[docs/SNR_CURVES.md](docs/SNR_CURVES.md)**, `tools/snr_curves.py`).
* **Broadcast mode** (one-way, no ACKs) for when only one direction works: offers and
  manifest repeated, data sent in a carousel of N passes; a later broadcast resumes into the
  same partial files.
* **Resume:** partial data lives in `<destination>/.airdrop-partial/` with a sidecar per
  file (matched by SHA-256, never by name), so an interrupted transfer continues where it
  stopped.

### File safety

Received names are untrusted: Unicode-normalised, reduced to a single path component
(path syntax is rejected), stripped of control/reserved characters and bidi overrides,
Windows device names (CON, NUL, COM1…) are renamed, length-limited, and the final path is
checked to be inside the destination. Existing files are never overwritten
(`name (1).ext`…). Files are written to a partial folder and moved into place only after
the SHA-256 matches; a mismatch is reported as a failure and the data stays in the
partial folder as `.corrupt`. Received files are never opened or executed — "Open folder"
only opens the destination directory.

## Security model and encryption

Encryption is **off by default**. When enabled, both users type the same passphrase
(agree on it out of band). Key = Scrypt(passphrase, random per-session 16-byte salt,
N = 2¹⁵, r = 8, p = 1); each frame payload (manifest, data, ACK, END_ACK, RESUME_INFO) is
sealed with ChaCha20-Poly1305 using a nonce built from the session id, direction and
sequence number, with the frame header as associated data. A wrong passphrase stops the
transfer cleanly with "Passphrase mismatch or corrupted data".

Limitations — please read:

* Confidentiality depends **entirely on the passphrase strength**. Anyone who records the
  audio can try passphrases offline at leisure; use a long random passphrase.
* No forward secrecy: whoever learns the passphrase later can decrypt old recordings.
* Metadata is visible: sender name, file count, total size, frame sizes and timing.
  OFFER and ACCEPT are not encrypted (they carry the salt).
* Without encryption, anyone in hearing range can decode the files.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No detection at all (receiver never shows a transmitter) | Band filtered by enhancements/driver; mic permission; wrong device; volume too low | Hardware check on both sides; disable enhancements; try exclusive mode; check microphone privacy setting; raise volume/transmit level; try the Audible profile |
| Detects frames but they fail (CRC/RS errors) | Low SNR, noise, reverberation, clipping | Move closer and face the speakers to each other; reduce noise; lower the transmit level if the check reports clipping; use mode 0/1 |
| Works close but not far | Free-space loss and reverberation | Stay within ~1 m; reduce reverberation (soft furnishings); expect MFSK modes |
| "Brick wall" drop at 16 or 18 kHz in the hardware check | Audio enhancements/APOs, a driver low-pass, shared-mode resampling, Bluetooth device | Disable enhancements, set 48 kHz, try exclusive mode, use a USB mic/speaker, or the Audible profile |
| ACKs never arrive (sender keeps polling, "link lost") | The reverse direction is weaker (e.g. receiver's speaker) | Swap roles or use **Broadcast** mode |
| "Device busy" / "unsupported sample rate" | Another app holds the device, or it does not support 48/96 kHz | Close other audio apps; set the default format to 48 kHz; choose another device |
| Transfer interrupted | Silence for 90 s, cancel, or link lost | Start the same transfer again — it resumes from the partial data |
| "Passphrase mismatch or corrupted data" | Different passphrases | Re-type the passphrase on both sides |

Logs: GUI *Log* page, or the rotating file `%LOCALAPPDATA%\UltrasonicAirDrop\logs\airdrop.log`
(Linux: `~/.local/share/ultrasonic-airdrop/logs/`, macOS: `~/Library/Application Support/UltrasonicAirDrop/logs/`).

## CLI

```bash
airdrop devices                                   # list devices (indices, host API, 48k/96k support)
airdrop check --in 3 --out 5                      # hardware self-check (plays sweeps)
airdrop linktest receive                          # on machine B
airdrop linktest send --duration 20               # on machine A
airdrop receive --out ~/Downloads/rx              # asks before accepting each offer
airdrop receive --out ./rx --accept-all --keep-listening
airdrop send report.pdf notes.txt                 # shows estimate, asks to confirm
airdrop send photo.jpg --broadcast --passes 3 -y  # one-way
airdrop send secret.zip --encrypt                 # prompts for the passphrase
airdrop send big.bin --profile au                 # audible profile
airdrop simulate a.bin b.txt --snr 12 --reverb 0.3 --ppm 150 --loss 0.05
airdrop wav-encode a.bin --out sig.wav            # broadcast frames to a WAV file
airdrop wav-decode sig.wav --out ./rx             # decode a WAV/recording
airdrop estimate photo.jpg                        # time per mode, plays nothing
airdrop send photo.jpg --profile fa --shrink-images   # fast, side by side; photo re-encoded to ~0.9 MB
```

`simulate` runs a complete two-endpoint transfer in-process through the channel simulator,
prints goodput and frame statistics and verifies the SHA-256 of every file.
`wav-decode` also accepts recordings made with any audio recorder (any sample rate;
it is resampled to 48 kHz if needed).

## Simulated performance

**All numbers in this section are from the simulator, not from hardware.** "In-band SNR"
is the received MFSK reference power over the noise power in a 3 kHz band; 16 KB of random
(incompressible) data, AWGN, 150 ppm clock offset, no reverberation, fixed mode:

| Mode | PHY / FEC | Data-phase goodput | Whole-transfer goodput (incl. offer, probe, manifest, END) | Transfer time |
|---:|---|---:|---:|---:|
| 0 | MFSK, RS nsym 64 | 13.7 B/s | 13.4 B/s | 20 min 22 s |
| 1 | MFSK, RS nsym 32 | 18.2 B/s | 17.7 B/s | 15 min 25 s |
| 2 | OFDM DBPSK, nsym 64 | 96.0 B/s | 84.4 B/s | 3 min 14 s |
| 3 | OFDM DBPSK, nsym 32 | 111.8 B/s | 96.4 B/s | 2 min 50 s |
| 4 | OFDM DQPSK, nsym 64 | 187.9 B/s | 148.0 B/s | 1 min 51 s |
| 5 | OFDM DQPSK, nsym 32 | 224.1 B/s | 169.6 B/s | 1 min 37 s |

In-band SNR 25 dB for these runs (every mode error-free); goodput counts acknowledged
on-air bytes, so it includes FEC, framing, ACK turnarounds and the 300 ms reverberation waits.

Adaptive transfers (mode chosen by probe and ACK feedback):

| Scenario (simulated) | File | Result | Final mode | Goodput (whole transfer) | Notes |
|---|---|---|---|---:|---|
| SNR 25 dB, +150 ppm | 16 KB | verified | 5 | 155.9 B/s | probe chose mode 5 directly |
| SNR 18 dB, −200 ppm, RT60 0.3 s (DRR 6 dB) | 4 KB | verified | 1 | 15.6 B/s | the probe counts the reverberant tail as noise and stays on MFSK (conservative; DBPSK would have worked in this particular room, but fails in others) |
| SNR 8 dB, +150 ppm, RT60 0.3 s, 5 % frame loss | 4 KB | verified | 0 | 9.9 B/s | 18 % data-frame loss incl. bursts; 28 units retransmitted |
| SNR 15 dB, +100 ppm, encrypted | 4 KB | verified | 2 | 48.1 B/s | 79 B/s in the data phase |
| Broadcast, SNR 15 dB, 2 passes, mode 1 | 4 KB | verified | 1 | 10.2 B/s | both passes always sent |

Reproduce with e.g. `airdrop simulate file.bin --snr 18 --ppm -200 --reverb 0.3`.

Frame-error-rate curves per mode: [docs/SNR_CURVES.md](docs/SNR_CURVES.md). Summary of the
FER ≤ 2 % thresholds: MFSK modes work down to about −5 dB in-band SNR, DBPSK OFDM needs
about +11 dB and DQPSK OFDM +14…16 dB, i.e. the OFDM modes need ~16–21 dB more SNR for
~5–13× the raw rate. Reverberation (RT60 0.3 s with the reverberant energy only 6 dB below
the direct sound) is harmless to MFSK but prevents OFDM from working; OFDM tolerates the
same RT60 when the direct-to-reverberant ratio is ≥ ~12 dB (short distance).

## Testing

```bash
pytest                    # fast suite (~3 min): unit, channel simulations, integration, GUI smoke test
pytest -m "slow or not slow"   # everything incl. long simulations (~4 min): 100-frame operating point, 64 KB transfer
ruff check . && ruff format --check .
mypy                      # strict on the core packages, relaxed for gui/
python tools/snr_curves.py --frames 50 --step 1 --snr-min -10 --snr-max 22 --out docs/SNR_CURVES.md --csv docs/snr_curves.csv
```

No test needs audio hardware: transfers run between two simulated endpoints driven by a
sample clock (deterministic, seeded, much faster than real time). The GUI test runs with
`QT_QPA_PLATFORM=offscreen` and asserts that building the window opens no audio stream.

Hardware testing is manual: see [docs/HARDWARE_TESTING.md](docs/HARDWARE_TESTING.md) and
`tools/measure_link.py` (per-frame CSV + per-mode summary).

## Known limitations

* **Not validated on real hardware** (see above).
* Throughput is fundamentally low (13–225 B/s simulated); large files take hours.
* Half-duplex: a machine cannot hear while it transmits; every window costs an ACK
  turnaround (~5 s).
* The guard interval (5.3 ms) is short compared with room reverberation; OFDM modes need
  a strong direct path. MFSK copes with reverberation but is slow.
* The receiver must accept an offer within ~1 minute (the sender stops re-offering after
  5 attempts). Broadcast frames that arrive before acceptance are ignored (later passes fill
  them in).
* Masks/thresholds are derived from simulated AWGN curves; real devices may need different
  margins.
* The 96 kHz profile requires both machines to support 96 kHz capture and playback with
  real response above 20 kHz, which is rare.
* Encryption limitations listed above.

## Project layout

```
ultrasonic_airdrop/   config, cli, logging, errors
  phy/                numerology, chirps, detector, resampling, MFSK, OFDM, synthesis, quality
  fec/                Reed–Solomon (reedsolo/creedsolo), interleaver, CRCs
  link/               frame definitions, PHY header, codec (frames ↔ samples), receive pipeline
  transport/          sender/receiver state machines, ARQ, rate adaptation, engine, estimates,
                      simulation and offline (WAV) drivers
  files/              manifest, sanitiser, chunker, reassembly, resume, hashing
  crypto/             Scrypt + ChaCha20-Poly1305
  audio/              backend protocol, sounddevice/WASAPI backend, simulator backend, devices, diagnostics
  channel/            composable channel impairments
  gui/                PySide6 application (only package that imports Qt)
tools/                snr_curves.py, measure_link.py
docs/                 PROTOCOL.md, HARDWARE_TESTING.md, SNR_CURVES.md (+ CSV)
tests/                unit/, sim/, integration/
```

## Tested dependency versions

Tested on Python 3.13.16 (Linux x86_64) with numpy 2.5.4, scipy 1.18.1, sounddevice 0.5.6
(PortAudio 19.6), reedsolo 1.7.0 (pure Python; `creedsolo` is used automatically if
installed), cryptography 50.0.2, PySide6 6.12.0, pytest 9.1.1, pytest-timeout 2.4.0,
ruff 0.17.2, mypy 2.4.0, Pillow 12.3.0 (optional photo shrinking). The full environment is pinned in `requirements.lock.txt`;
`pyproject.toml` declares compatible ranges. Python 3.11/3.12 and Windows were not tested
in this environment.
