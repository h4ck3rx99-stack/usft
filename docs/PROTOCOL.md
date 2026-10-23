# Ultrasonic AirDrop — wire protocol, version 1

This document specifies the acoustic physical layer, the link frame format and the
session state machines in enough detail for an independent implementation to
interoperate. The reference implementation's single source of truth for the
link format is `ultrasonic_airdrop/link/frames.py`; the PHY lives in
`ultrasonic_airdrop/phy/` and `ultrasonic_airdrop/link/codec.py`.

All multi-byte integers are **little-endian**. "LE u16" = unsigned 16-bit little-endian.

---

## 1. Physical layer

### 1.1 Profiles

| | Near-ultrasonic `nu` (default) | Audible `au` | Ultrasonic experimental `us` | Fast close-range `fa` |
|---|---|---|---|---|
| Sample rate *fs* | 48 000 Hz | 48 000 Hz | 96 000 Hz | 48 000 Hz |
| FFT size *N* | 1024 | 1024 | 2048 | 128 |
| Bin spacing | 46.875 Hz | 46.875 Hz | 46.875 Hz | 375 Hz |
| Guard *G* | 256 | 256 | 512 | 64 |
| Symbol length *L = N + G* | 1280 (26.667 ms) | 1280 | 2560 (26.667 ms) | 192 (4 ms) |
| Taper *T* | 64 | 64 | 128 | 16 |
| Chirp band | 17.0–20.0 kHz | 2.5–5.5 kHz | 20.5–23.5 kHz | 1.5–20.5 kHz |
| Chirp length *C* | 2048 | 2048 | 4096 (same 42.7 ms) | 2048 |
| MFSK subband A bins | 376–391 | 64–79 | 448–463 | 8–23 |
| MFSK subband B bins | 400–415 | 88–103 | 472–487 | 28–43 |
| OFDM candidate bins | 374–425 (52) | 60–111 (52) | 444–495 (52) | 4–55 (52) |

A frequency "bin *k*" means *k · fs / N* Hz. All tones and carriers sit exactly
on bin centres. The `us` chirp length is 4096 samples (not 2048) so that its
duration equals the 48 kHz profiles'.

Derived constants used below (values for 48 kHz; scale by *fs*/48000 for `us`):
taper *T* as in the table, ramp = 5 ms (240 samples), chirp-pair tolerance = 64 samples,
probe silence = 0.25 s (12 000 samples).

### 1.2 Burst layout

Offsets are relative to the first sample of the up-chirp (*t₀*):

| Offset | Content |
|---|---|
| 0 | up-chirp, *C* samples |
| *C* | down-chirp, *C* samples |
| 2*C* | sync word: 4 MFSK symbols, bytes `5A C3 96 3C` |
| 2*C* + 4*L* | PHY header: 14 MFSK symbols (§1.6) |
| 2*C* + 18*L* | body (*B* samples, depends on mode, §1.7) |
| 2*C* + 18*L* + *B* | *G* samples of silence |
| 2*C* + 18*L* + *B* + *G* | postamble: up-chirp, *C* samples |
| … + *C* | *T* samples (tail of the last symbol taper), end of burst |

The whole burst is multiplied by a 5 ms raised-cosine (sin²) fade at both ends.
Peak amplitude is the user transmit level (default −6 dBFS, clamped to
[−20, −3] dBFS). Consecutive frames in one transmission are separated by 50 ms of
silence.

**Chirps.** Linear chirp with phase φ(t) = 2π(f_a t + (f_b − f_a) t² / (2D)),
D = C/fs, Tukey window α = 0.1, peak amplitude = transmit level. Up-chirp:
f_a = band low, f_b = band high; down-chirp: reversed.

### 1.3 MFSK ("R") modulation

One byte per symbol. High nibble *h* → subband A, low nibble *l* → subband B.
Nibble value *v* is placed at tone position *p* where `GRAY[p] = v`,
`GRAY[p] = p XOR (p >> 1)` (adjacent tone positions differ in one bit).
Tone bin = first bin of the subband + *p*.

Each symbol of index *s* (starting at burst offset *S_s* = start + *s·L*) is the sum
of the two tones, each with amplitude = peak/2, using an absolute phase reference
`cos(2π k n / N)` where *n* is the sample offset within the burst. Each symbol is
generated over *L + T* samples, windowed by a sin² rise over the first *T* samples
and a cos² fall over the last *T* samples (which overlap the next symbol's rise),
so the guard contains the cross-fade and the FFT window is unaffected.

**Demodulation:** FFT of *N* samples starting *G/2* samples into the symbol
(rectangular window); in each subband pick the bin with maximum energy.
Erasure flag: best/second-best energy ratio < 1.6 in either subband.

### 1.4 OFDM ("F1" DBPSK, "F2" DQPSK)

* **Carriers**: the candidate bins enabled by the carrier mask (§1.5). Enabled
  carriers are numbered 0..n−1 in ascending frequency; those with index ≡ 0 (mod 8)
  are **pilots**, all others carry data.
* **Reference symbol**: first body symbol; carrier *i* (of *n* enabled) has phase
  π i² / n (Newman phases).
* **Differential encoding**: symbol *s* ≥ 1 on each carrier = previous symbol ×
  differential phasor. DBPSK: bit 0 → +1, bit 1 → −1. DQPSK (2 bits, MSB first):
  `00`→+1, `01`→+j, `10`→−j, `11`→−1.
* **Pilot sequence**: differential phasors 1 − 2·b where b is the PRBS-7 bit stream
  (x⁷ + x⁶ + 1, state seed 0x5D, output bit = bit6 XOR bit5, shifted in at the LSB)
  read row by row: symbol 1 pilots 0..P−1, then symbol 2, …
* **Byte mapping (time-first)**: with *D* data carriers and *m* bits per carrier,
  coded byte *i* goes to carrier *i mod D* of symbol group *⌊i/D⌋*; a group is
  *8/m* consecutive data symbols; the byte's bits are sent MSB first, *m* bits per
  symbol. Unused slots of the last group are zero.
* **Synthesis**: per symbol, inverse real FFT of the carrier phasors, two iterations
  of clip-and-filter (clip at 8 dB above the symbol RMS, then zero all bins that are
  not enabled carriers), cyclic extension (CP of *G* samples plus a *T*-sample
  cyclic suffix), sin²/cos² taper as for MFSK. The OFDM section (also the probe
  training symbols) is scaled so its RMS is 10 dB below the peak level and hard-limited
  at the peak.
* **Demodulation** (informative): FFT window starts 7G/8 into each symbol;
  z = Y_s · conj(Y_{s−1}); remove a common phase and a linear phase slope across
  carriers estimated from the pilots; nearest-point decisions.

### 1.5 Carrier masks

A mask is a 52-bit set (bit *i* = candidate carrier *i*), serialised as **7 bytes
LE** (bits 52–55 must be 0). `mask_id` = 0 for the full mask, otherwise
`CRC16-CCITT-FALSE(mask bytes) mod 255 + 1`. A receiver remembers every mask it
has itself advertised (PROBE_REPORT, ACK) in the session and resolves `mask_id`
against that set; frames with an unknown `mask_id` cannot be decoded.

### 1.6 PHY header

6 plaintext bytes, RS-encoded with nsym = 8 into 14 bytes, sent as 14 MFSK symbols:

| Off | Size | Field |
|---|---|---|
| 0 | 1 | `mode_id` (0–5, or 0xFE = PROBE) |
| 1 | 1 | `mask_id` (0 for MFSK modes and PROBE) |
| 2 | 2 | `body_len` — link frame length in bytes before FEC (1–4096) |
| 4 | 2 | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, xorout 0) of bytes 0–3 |

### 1.7 FEC and body

Reed–Solomon over GF(2⁸): primitive polynomial 0x11D, generator α = 2, first
consecutive root 0 (the `reedsolo` defaults: `RSCodec(nsym, nsize=255, fcr=0,
prim=0x11D, generator=2)`), systematic, parity appended.

Body of *n* bytes with parity *nsym*: *k* = ⌈n / (255 − nsym)⌉ codewords,
*d* = ⌈n / k⌉ data bytes each (zero-padded to *k·d*), each encoded to *d + nsym*
bytes. **Interleaving**: byte *j* of codeword *i* is transmitted at position
*j·k + i*.

| mode_id | PHY | nsym | body samples *B* |
|---|---|---|---|
| 0 | MFSK | 64 | coded_len · L |
| 1 | MFSK | 32 | coded_len · L |
| 2 | OFDM DBPSK | 64 | (1 + data symbols) · L |
| 3 | OFDM DBPSK | 32 | (1 + data symbols) · L |
| 4 | OFDM DQPSK | 64 | (1 + data symbols) · L |
| 5 | OFDM DQPSK | 32 | (1 + data symbols) · L |
| 0xFE | PROBE: MFSK (nsym 64) + 4 training symbols + 0.25 s silence | 64 | coded_len · L + 4L + 0.25 s |

Data symbols = ⌈coded_len / D⌉ · (8/m).

**PROBE training symbols**: 4 OFDM symbols on all 52 candidate carriers, symbol *s*
carrier *i* with phase π (i + 7s)² / 52, synthesised as in §1.4 (no differential
encoding).

### 1.8 Receiver timing (informative)

* Probe SNR per carrier: signal = |mean channel estimate over the 4 training symbols|²;
  noise = max(background noise in the silence gap, energy of the reverberant tail in the
  first FFT window after the guard, scatter of the 4 per-symbol estimates).
* Detection: band-pass (Butterworth order 4, zero-phase, band ± 500 Hz), envelope
  correlation with analytic chirp templates; accept when peak ≥ 8 × median of the
  surrounding envelope (CFAR) **and** a down-chirp peak lies *C* ± 64 samples later;
  confirm with ≥ 3 of 4 sync bytes.
* *t₀* = mean of up- and down-chirp positions (cancels delay–Doppler coupling).
  Their difference gives the frequency offset Δf = (q − p − C) · k / (2 fs),
  k = chirp sweep rate in Hz/s.
* Sample-rate ratio *r* = (postamble position + Δf·fs/k − t₀) / nominal distance;
  rejected if |r − 1| > 500 ppm (then the session's running average is used).
  The frame is resampled by 1/r, re-timed, frequency-corrected by −Δf and demodulated.

---

## 2. Link frame

| Off | Size | Field |
|---|---|---|
| 0 | 1 | magic `0xA5` |
| 1 | 1 | version = 1 |
| 2 | 1 | type (§3) |
| 3 | 1 | flags: bit0 `dir` (0 = sender→receiver, 1 = receiver→sender); bit1 `enc`; bit2 `poll`; bits 3–7 reserved = 0 |
| 4 | 4 | `session_id` (random, chosen by the sender) |
| 8 | 4 | `seq` (per direction, starting at 0, +1 per transmitted frame incl. retransmissions) |
| 12 | 2 | `payload_len` |
| 14 | n | payload (ciphertext + 16-byte tag if `enc`) |
| 14+n | 4 | CRC-32 (zlib/IEEE) of bytes 0 … 14+n−1 |

The receiver discards a frame (silently, counting a frame error) on: wrong magic,
length mismatch, CRC failure, version ≠ 1, reserved flag bits set, a direction
flag that does not match the type's direction (§3), `enc` set on a type that is
never encrypted, unknown session (except OFFER), or a frame carrying its own `dir`
(self-echo). Unknown types with a valid CRC are ignored and logged. A CRC-valid
OFFER with version ≠ 1 is answered with CANCEL(reason = VERSION).

Only an RS-decoded body whose CRC-32 verifies is ever accepted.

## 3. Frame types and payloads

| Type | Name | Dir | Encrypted when session is encrypted |
|---|---|---|---|
| 0x01 | OFFER | S→R | no |
| 0x02 | ACCEPT | R→S | no |
| 0x03 | PROBE | S→R | no |
| 0x04 | PROBE_REPORT | R→S | no |
| 0x05 | MANIFEST | S→R | yes |
| 0x06 | DATA | S→R | yes |
| 0x07 | ACK | R→S | yes |
| 0x08 | END | S→R | no |
| 0x09 | END_ACK | R→S | yes |
| 0x0A | CANCEL | either | no |
| 0x0B | POLL | S→R | no |
| 0x0C | RESUME_INFO | R→S | yes |

A payload must be consumed exactly (trailing bytes are an error).

**OFFER** (≤ 100 bytes)

| Size | Field |
|---|---|
| 1 | protocol version (1) |
| 2 | capability bitmap: bits 0–5 = modes 0–5 supported, bit 8 = encryption, bit 9 = compression |
| 1 | transfer mode: 0 interactive, 1 broadcast |
| 2 | file count |
| 8 | total on-air bytes (Σ `tx_size`) |
| 4 | manifest blob length |
| 32 | SHA-256 of the manifest blob |
| 1 | encrypted (0/1) |
| 16 | Scrypt salt (zeros if not encrypted) |
| 1 + n | sender display name: length (≤ 32) + UTF-8 |

**ACCEPT** — code u8 (0 OK, 1 rejected by user, 2 too large, 3 busy, 4 unsupported);
capability bitmap u16; resume-available u8 (receiver has partial data recorded for this
manifest hash); max acceptable bytes u64.

**PROBE** — empty payload; sent with PHY `mode_id` 0xFE (§1.7).

**PROBE_REPORT** — 52 × u8 per-candidate-carrier SNR, value = round((SNR_dB + 10) · 4)
clamped to 0–255 (−10 … 53.75 dB, 0.25 dB steps); MFSK SNR i16 (0.1 dB, −32768 = unknown);
recommended mode u8 (0–5); carrier mask 7 bytes; measured clock offset i16 (ppm).

**MANIFEST** — part_index u16, part_count u16 (≥ 1, index < count), then bytes of the
manifest blob. The blob is split into equal parts of at most 115 bytes (99 if encrypted)
so that each frame fits the 6 s airtime cap in mode 0.

**DATA** — file_index u16, start_unit u32, unit_count u8 (≥ 1), data. Units are 32 bytes
of the file's on-air content; data length must equal the sum of the unit lengths (the
last unit of a file may be short).

**ACK** — base_seq u32; bitmap u64 (bit *i* ⇒ the receiver holds sender frame
`base_seq + i`); mean SNR i16 (0.1 dB) of frames in the window; frame-error count u8 in the
window; suggested mode u8; carrier mask (7 bytes); n u8 (≤ 4) then n × (file u16, start u32,
count u16) missing unit ranges (gaps below the highest unit received for that file).

**END** — empty.

**END_ACK** — total_files u16, first u16, n u16, n × status u8 for files
`first … first+n−1` (0 OK, 1 HASH_MISMATCH, 2 INCOMPLETE, 3 WRITE_ERROR, 4 SKIPPED);
then m u8 (≤ 8) missing ranges (file u16, start u32, count u16). Statuses for more than
100 files are split over several END_ACK frames sent back-to-back; missing ranges ride on
the last one.

**CANCEL** — reason u8 (0 user, 1 version, 2 authentication, 3 timeout, 4 error,
5 rejected), text length u8 (≤ 64) + UTF-8. Sent twice.

**POLL** — empty; always sent with the `poll` flag.

**RESUME_INFO** — file_index u16, n u8 (≤ 48 and within the frame airtime cap; 14 in
practice), n × (start_unit u32, count u32) ranges the receiver already holds.

## 4. Manifest

UTF-8 JSON, compressed with zlib (≤ 256 KB uncompressed, ≤ 1000 files):

```json
{"files":[{"compressed":false,"i":0,"mtime":1700000000,"name":"a.txt",
  "sha256":"<64 lowercase hex>","size":123,"tx_size":123}],"unit":32,"v":1}
```

Validation is strict: exactly these keys; `v` = 1; `unit` = 32; indices 0…n−1 in order;
integer types (JSON booleans are not integers); `size`, `tx_size` ≥ 0; `tx_size` = `size`
when not compressed; compressed files are non-empty; names pass the sanitiser in
strict mode (no `/`, `\`, `:`, `.`/`..`, drive letters, UNC paths); Σ `size` ≤ the
receiver's limit. `size`/`sha256` describe the original file; `tx_size` the on-air
bytes (zlib stream if `compressed`). Files with identical (sha256, tx_size, compressed)
are aliases: the sender transmits only the first, the receiver writes both names.

## 5. Encryption (optional)

* key = Scrypt(passphrase NFC-normalised UTF-8, salt from OFFER, N = 2¹⁵, r = 8, p = 1, 32 bytes)
* AEAD = ChaCha20-Poly1305
* nonce (12 bytes) = session_id (u32 LE) ‖ dir (1 byte) ‖ `00 00 00` ‖ seq (u32 LE)
* AAD = the 14-byte link header (its `payload_len` is the ciphertext length incl. tag)
* Encrypted types: MANIFEST, DATA, ACK, END_ACK, RESUME_INFO. In an encrypted session
  an unencrypted frame of these types is rejected.
* After 3 consecutive authentication failures a side sends CANCEL(reason = 2) twice and
  fails with "Passphrase mismatch or corrupted data".

## 6. Frame sizing and airtime

Frame airtime = burst length / fs (§1.2). Every frame is ≤ 6 s; OFDM bodies ≤ 4 s.
DATA frames carry the largest unit count (≤ 255) meeting the cap. For the `nu`
profile without encryption:

| mode | units / DATA frame | frame airtime | frames per window (≤ 20 s) | raw payload rate |
|---|---|---|---|---|
| 0 | 3 | 5.55 s | 3 | 17 B/s |
| 1 | 4 | 5.55 s | 3 | 23 B/s |
| 2 | 17 | 4.27 s | 4 | 128 B/s |
| 3 | 20 | 4.27 s | 4 | 150 B/s |
| 4 | 37 | 4.59 s | 4 | 258 B/s |
| 5 | 44 | 4.59 s | 4 | 307 B/s |

Frames per window = min(W, ⌊20 s / frame airtime⌋), W = 6 (MFSK) or 8 (OFDM).
For the `fa` profile modes 4 and 5 hit the 255-unit (8160-byte) DATA limit at 4.1 s / 3.6 s
per frame (≈2.0 / 2.3 KB/s raw).

## 7. Interactive session (half-duplex ARQ)

Turnaround rule (both sides): wait **300 ms** after the end of the last received frame
before transmitting (reverberation decay; also clears the peer's 250 ms half-duplex gate).
While transmitting, a station ignores its own capture for the playback interval + 250 ms.

```
Sender                                   Receiver
OFFER ─────────────────────────────────▶ (user accepts)
      ◀───────────────────────────────── ACCEPT
PROBE (mode 0xFE) ─────────────────────▶ measure per-carrier SNR
      ◀───────────────────────────────── PROBE_REPORT (mode, mask)
MANIFEST parts…(last has poll) ────────▶ verify SHA-256, validate
      ◀───────────────────────────────── [RESUME_INFO…] ACK
DATA × window (last has poll) ─────────▶ write units
      ◀───────────────────────────────── ACK (bitmap, SNR, suggested mode/mask)
… repeat …
END (poll) ────────────────────────────▶ decompress, SHA-256 verify, atomic rename
      ◀───────────────────────────────── END_ACK (per-file status)
```

**Sender**

| State | Action | Next |
|---|---|---|
| OFFERING | send OFFER; listen max(6 s, reply wait); up to 5 sends | ACCEPT(OK) → PROBING; ACCEPT(≠OK) → FAILED; 5 timeouts → FAILED |
| PROBING | send PROBE; wait reply | PROBE_REPORT → MANIFEST with mode = min(recommended, common caps) and mask; 3 timeouts → MANIFEST with mode 1, full mask |
| MANIFEST | send up to W parts not yet acknowledged (mode 0), last with poll; wait ACK | all parts acked → DATA |
| DATA | send a window of DATA frames (lowest unacknowledged units first, so retransmissions precede new data), last with poll; wait ACK | all units acked → ENDING |
| ENDING | send END (poll); wait END_ACK(s) | all OK → DONE; INCOMPLETE → un-ack missing ranges (whole file if none listed) → DATA; HASH_MISMATCH/WRITE_ERROR → FAILED; 6 timeouts → FAILED |

*Reply wait* = 0.6 s (detection latency) + 0.3 s + reply airtime + output latency
(0.15 s) + 2.5 s, extended while a frame is being received and by one ACK airtime + 2.5 s
whenever a valid frame from the receiver arrives.

*ACK timeout*: send POLL (mode 0) and wait again; after 6 consecutive POLLs without an
ACK → FAILED ("link lost"; receiver partials remain resumable).

*ACK processing*: frames of the window whose seq has its bitmap bit set are acknowledged;
all other frames of the window are lost and their units become eligible again. Missing
ranges in the ACK un-acknowledge those units.

*Rate adaptation*: after each data ACK: if lost/sent > 25 % → step down one mode now;
else after 3 consecutive loss-free windows step up one mode **only** if the receiver's
suggested mode is higher; the mode never exceeds the receiver's suggestion; two consecutive
missed ACKs → step down one mode. OFDM frames use the mask from the latest ACK.

**Receiver**

| State | Event | Action |
|---|---|---|
| LISTENING | OFFER (valid) | → OFFERED, show offer; auto-accept if configured |
| OFFERED | user accepts | derive key if encrypted; queue ACCEPT(OK); → RECEIVING |
| OFFERED | user rejects | queue ACCEPT(REJECTED); → CANCELLED |
| any | OFFER total > limit | ACCEPT(TOO_LARGE); → FAILED |
| RECEIVING | OFFER again (no manifest yet) | re-send ACCEPT |
| RECEIVING | PROBE | reply PROBE_REPORT |
| RECEIVING | MANIFEST/DATA/POLL with poll flag | reply [RESUME_INFO…] ACK |
| RECEIVING | MANIFEST/DATA without poll | arm window timeout: now + frame airtime + 3 s → ACK if nothing else arrives |
| RECEIVING | all MANIFEST parts | verify blob SHA-256 and length against OFFER; validate; create partial files; failure → CANCEL(ERROR) ×2, FAILED |
| RECEIVING | END | finalise every file; reply END_ACK; all OK → COMPLETE |
| COMPLETE | END/POLL again | re-send END_ACK/ACK for 30 s (linger) |
| any active | CANCEL | → CANCELLED (AUTH/VERSION → FAILED) |
| any active | 90 s without frames | → INTERRUPTED (partials kept) |
| any active | user cancel | CANCEL(USER) ×2 → CANCELLED (partials kept unless "discard") |

**ACK construction**: `base_seq` = max(0, highest seq received − 63); bitmap of received
seqs in [base, base + 63]; suggested mode and mask from the receiver's SNR estimates
(probe, then OFDM per-carrier EVM and MFSK SNR level changes): for each OFDM mode from
fastest to slowest, the mask is the carriers with SNR ≥ requirement + 3 dB margin − 3 dB
slack; the mode is chosen if the mask has ≥ 24 carriers and their median SNR ≥ requirement +
3 dB (`REQUIRED_SNR_DB` in `config.py`); otherwise mode 1 if the MFSK SNR ≥ its requirement +
3 dB, else mode 0; if > 25 % of the window's frames failed, the
suggestion is at most the last mode − 1.

**Resume**: partial files are `<dest>/.airdrop-partial/<sha256>.part` with a sidecar
`<sha256>.json` listing received unit ranges; they are matched by SHA-256 + `tx_size` +
compression flag only. When a manifest matches stored partials the receiver sends
RESUME_INFO frames (before the ACK) after the manifest phase; the sender marks those units
as acknowledged.

## 8. Broadcast session (one-way)

Sender: OFFER ×3, every MANIFEST part ×2 (mode 0), then `passes` (default 2) passes of
the data carousel (every unit of every non-alias file in order, user-selected mode,
default 1, full carrier mask), END after each pass and once more at the end. No frame
carries the poll flag and nothing is expected back.

Receiver: accepts per user decision/auto-accept (frames arriving before acceptance are
ignored), verifies and places each file as soon as all of its units are present, and on
END or 90 s of silence reports which files were verified and which are missing.
A later broadcast of the same content resumes into the same partials.

## 9. Integrity guarantees

* PHY header: CRC-16 after RS. Body: RS then CRC-32 — an RS "success" with CRC failure is
  a frame error.
* Files: SHA-256 of the original bytes (after decompression) must match the manifest
  before a file is moved (`os.replace`) to a never-overwriting name in the destination;
  otherwise it stays in the partial directory as `<sha256>.corrupt` and is reported as
  HASH_MISMATCH. Success is reported only after verification.
