"""Streaming receive chain: buffer → detect → sync/header → resample → demod → FEC.

The pipeline is fed contiguous blocks of captured samples (with absolute sample
indices) and returns :class:`RxFrame` objects.  It knows nothing about sessions
or link-frame semantics: ``RxFrame.body`` is the RS-decoded body (the link
frame bytes, still to be CRC-checked and parsed by :mod:`link.frames`), or
``None`` for a frame error.

Per frame:

1. CFAR detection of the up-chirp, confirmed by the down-chirp at the expected
   spacing (±64 samples at 48 kHz) and by the 4-symbol sync word (≥3 correct).
2. PHY header decode (14 MFSK symbols, RS nsym=8, CRC-16).
3. Wait until the whole frame is buffered, find the postamble, estimate the
   sample-rate ratio ``r`` and resample the frame by ``1/r`` (session average
   ratio is used if the postamble is missing).
4. Re-time on the corrected frame, estimate and remove residual carrier
   frequency offset from the up/down chirp pair, demodulate, RS-decode.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ultrasonic_airdrop.config import (
    MAX_PPM,
    PROBE_MODE_ID,
    SYNC_MIN_CORRECT,
    SYNC_WORD,
    Profile,
    get_mode,
)
from ultrasonic_airdrop.errors import FecError, FrameError
from ultrasonic_airdrop.fec.rs import decode_body
from ultrasonic_airdrop.link.codec import body_layout
from ultrasonic_airdrop.link.phy_header import PhyHeader
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.phy import mfsk, ofdm
from ultrasonic_airdrop.phy.detector import ChirpDetector
from ultrasonic_airdrop.phy.numerology import (
    HEADER_SYMBOLS,
    SYNC_SYMBOLS,
    FloatArray,
    numerology,
)
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.phy.resample import (
    frequency_shift,
    ratio_from_span,
    ratio_is_plausible,
    ratio_to_ppm,
    resample_by_ratio,
)

log = logging.getLogger(__name__)

PRE_WINDOW_S = 0.5
MIN_HOP_S = 0.1
CFO_CORRECT_MIN_HZ = 0.3
RATIO_EMA = 0.3
POSTAMBLE_RATIO_FACTOR = 0.5

MaskLookup = Callable[[int], CarrierMask | None]


@dataclass
class RxInfo:
    """Measurements attached to a received frame."""

    profile_key: str
    start_index: int
    end_index: int
    detect_ratio: float = 0.0
    header: PhyHeader | None = None
    mfsk_snr_db: float = float("nan")
    snr_db: float = float("nan")
    carrier_snr_db: FloatArray | None = None
    probe_snr_db: FloatArray | None = None
    bin_signal: FloatArray | None = None
    bin_noise: FloatArray | None = None
    ratio: float = 1.0
    postamble_found: bool = False
    cfo_hz: float = 0.0
    rs_corrected: int = 0
    erasures: int = 0

    @property
    def ppm(self) -> float:
        """Measured sample-rate offset in ppm."""
        return ratio_to_ppm(self.ratio)


@dataclass
class RxFrame:
    """One detected frame: decoded body bytes or ``None`` with an error string."""

    body: bytes | None
    info: RxInfo
    error: str = ""

    @property
    def ok(self) -> bool:
        """True if the body was decoded (link-level CRC still to be checked)."""
        return self.body is not None


@dataclass
class PipelineStats:
    """Counters for diagnostics."""

    detections: int = 0
    sync_failures: int = 0
    header_failures: int = 0
    body_failures: int = 0
    frames_ok: int = 0
    postamble_missing: int = 0
    last_detect_ratio: float = 0.0


@dataclass
class _Pending:
    t0: float
    header: PhyHeader
    detect_ratio: float
    frame_samples_nominal: int
    post_offset: int
    mfsk_snr_db: float
    need_end: int
    raw_cfo_hz: float = 0.0


class ReceiverPipeline:
    """Streaming receiver for one profile."""

    def __init__(
        self,
        profile: Profile,
        detect_ratio: float = 8.0,
        erasure_ratio: float = 1.6,
        mask_lookup: MaskLookup | None = None,
        capacity_s: float = 40.0,
    ) -> None:
        self.profile = profile
        self.num = numerology(profile)
        self.detector = ChirpDetector(profile, detect_ratio)
        self.erasure_ratio = erasure_ratio
        self.mask_lookup = mask_lookup
        self._cap = int(capacity_s * profile.sample_rate)
        self._buf = np.zeros(self._cap)
        self._len = 0
        self._buf_start = 0  # absolute index of _buf[0]
        self._scan_pos = 0
        self._pending: _Pending | None = None
        self._waiting = False
        self._mutes: list[tuple[int, int]] = []
        self.session_ratio = 1.0
        self.stats = PipelineStats()

    # ------------------------------------------------------------------ settings
    def set_detect_ratio(self, ratio: float) -> None:
        """Change the CFAR threshold."""
        self.detector.detect_ratio = ratio

    # ------------------------------------------------------------------ buffer
    @property
    def end_index(self) -> int:
        """Absolute index one past the newest buffered sample."""
        return self._buf_start + self._len

    def busy(self) -> bool:
        """True while a frame has been detected and is still being received/decoded."""
        return self._pending is not None or self._waiting

    def add_mute(self, start: int, end: int) -> None:
        """Ignore captured audio in ``[start, end)`` (half-duplex gating)."""
        if end <= start:
            return
        self._mutes.append((start, end))
        lo = max(start, self._buf_start)
        hi = min(end, self.end_index)
        if hi > lo:
            self._buf[lo - self._buf_start : hi - self._buf_start] = 0.0
        # A pending frame overlapping our own transmission cannot be trusted.
        if self._pending is not None and start < self._pending.need_end:
            self._pending = None
            self._waiting = False

    def _apply_mutes(self, start_abs: int, block: FloatArray) -> None:
        end_abs = start_abs + len(block)
        keep = []
        for s, e in self._mutes:
            if e <= self._buf_start - self._cap:
                continue
            keep.append((s, e))
            lo, hi = max(s, start_abs), min(e, end_abs)
            if hi > lo:
                block[lo - start_abs : hi - start_abs] = 0.0
        self._mutes = keep

    def _muted_mask(self, start_abs: int, n: int) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        for s, e in self._mutes:
            lo, hi = max(s, start_abs), min(e, start_abs + n)
            if hi > lo:
                mask[lo - start_abs : hi - start_abs] = True
        return mask

    def feed(self, samples: FloatArray, start_index: int | None = None) -> None:
        """Append captured samples.  Gaps (if ``start_index`` jumps ahead) are zero-filled."""
        block = np.asarray(samples, dtype=np.float64).reshape(-1).copy()
        if start_index is None:
            start_index = self.end_index
        if start_index < self.end_index:
            skip = self.end_index - start_index
            block = block[skip:]
            start_index = self.end_index
        gap = start_index - self.end_index
        if gap >= self._cap:
            self._reset(start_index)
        elif gap > 0:
            self._append(np.zeros(gap))
        if block.size:
            self._apply_mutes(start_index, block)
            self._append(block)

    def _reset(self, start_abs: int) -> None:
        self._buf_start = start_abs
        self._len = 0
        self._pending = None
        self._waiting = False
        self._scan_pos = start_abs

    def _append(self, block: FloatArray) -> None:
        n = len(block)
        if n >= self._cap:
            self._reset(self.end_index + n - self._cap)
            block = block[-self._cap :]
            n = self._cap
        if self._len + n > self._cap:
            keep_from = self._scan_pos - int(PRE_WINDOW_S * self.profile.sample_rate) - self.num.chirp_len
            if self._pending is not None:
                keep_from = min(keep_from, int(self._pending.t0) - self.num.chirp_len)
            drop = max(keep_from - self._buf_start, self._len + n - self._cap)
            drop = min(max(drop, 0), self._len)
            self._buf[: self._len - drop] = self._buf[drop : self._len]
            self._len -= drop
            self._buf_start += drop
            if self._scan_pos < self._buf_start:
                self._scan_pos = self._buf_start
            if self._pending is not None and int(self._pending.t0) < self._buf_start:
                self._pending = None
        self._buf[self._len : self._len + n] = block
        self._len += n

    def _slice(self, start_abs: int, end_abs: int) -> FloatArray:
        lo = start_abs - self._buf_start
        hi = end_abs - self._buf_start
        if lo < 0 or hi > self._len or hi < lo:
            raise IndexError("requested samples not buffered")
        return self._buf[lo:hi]

    def recent(self, n: int) -> FloatArray:
        """Copy of the newest ``n`` samples (fewer if not yet available)."""
        n = min(n, self._len)
        return self._buf[self._len - n : self._len].copy()

    # ------------------------------------------------------------------ processing
    def process(self) -> list[RxFrame]:
        """Run detection/decoding on everything buffered; return completed frames."""
        frames: list[RxFrame] = []
        for _ in range(64):
            if self._pending is not None:
                if self.end_index >= self._pending.need_end:
                    frames.append(self._decode_pending())
                    continue
                break
            if not self._scan():
                break
        return frames

    def _scan(self) -> bool:
        num = self.num
        fs = self.profile.sample_rate
        c = num.chirp_len
        tol = num.pair_tol
        if self._scan_pos < self._buf_start:
            self._scan_pos = self._buf_start
        hi = self.end_index - (2 * c + tol + 8)
        lo = self._scan_pos
        if hi - lo < int(MIN_HOP_S * fs):
            return False
        region_start = max(self._buf_start, lo - int(PRE_WINDOW_S * fs))
        region_end = hi + 2 * c + tol + 8
        x = self._slice(region_start, region_end)
        muted = self._muted_mask(region_start, len(x))
        env_up, env_dn = self.detector.envelopes(self.detector.bandpass(x))
        valid = ~muted[: len(env_up)]
        # also exclude correlation positions whose chirp window overlaps a mute
        if muted.any():
            cs = np.concatenate([[0], np.cumsum(muted)])
            overl = (cs[c : c + len(env_up)] - cs[: len(env_up)]) > 0
            valid &= ~overl
        if valid.sum() < c:
            self._scan_pos = hi
            self._waiting = False
            return False
        med_up = float(np.median(env_up[valid]))
        med_dn = float(np.median(env_dn[valid]))
        ratio = self.detector.detect_ratio
        i_lo, i_hi = lo - region_start, hi - region_start
        seg = env_up[i_lo:i_hi]
        above = np.flatnonzero((seg >= ratio * max(med_up, 1e-12)) & valid[i_lo:i_hi]) + i_lo
        half = c // 2
        checked_until = -1
        for i in (int(v) for v in above):
            if i <= checked_until:
                continue
            w0, w1 = max(0, i - half), min(len(env_up), i + half + 1)
            if env_up[i] < float(np.max(env_up[w0:w1])):
                continue
            checked_until = i + half
            if not self.detector.passes(float(env_up[i]), med_up):
                continue
            p = self.detector.first_arrival(env_up, int(i))
            d0, d1 = p + c - tol, p + c + tol + 1
            if d0 < 0 or d1 > len(env_dn):
                continue
            q = d0 + int(np.argmax(env_dn[d0:d1]))
            if not self.detector.passes(float(env_dn[q]), med_dn):
                continue
            q = self.detector.first_arrival(env_dn, q)
            pf = self.detector.refine(env_up, p)
            qf = self.detector.refine(env_dn, q)
            t0 = region_start + 0.5 * (pf + qf - c)
            t0i = round(t0)
            need = t0i + num.body_start + num.sym_len
            if self.end_index < need:
                self._scan_pos = region_start + max(0, int(i) - 1)
                self._waiting = True
                return False
            self._waiting = False
            det_ratio = float(env_up[p] / max(med_up, 1e-12))
            self.stats.detections += 1
            self.stats.last_detect_ratio = det_ratio
            pending = self._read_header(t0, det_ratio, self.detector.cfo_from_pair(pf, qf))
            if pending is None:
                continue
            self._pending = pending
            self._scan_pos = int(pending.t0) + pending.frame_samples_nominal - c
            return True
        self._waiting = False
        self._scan_pos = hi
        return False

    def _read_header(self, t0: float, det_ratio: float, raw_cfo: float) -> _Pending | None:
        num = self.num
        t0i = round(t0)
        starts = t0i + num.sync_start + np.arange(SYNC_SYMBOLS + HEADER_SYMBOLS) * num.sym_len
        x = self._slice(self._buf_start, self.end_index)
        res = mfsk.demodulate(x, starts - self._buf_start, num, self.erasure_ratio)
        sync = res.data[:SYNC_SYMBOLS]
        correct = sum(a == b for a, b in zip(sync, SYNC_WORD, strict=True))
        if correct < SYNC_MIN_CORRECT:
            self.stats.sync_failures += 1
            log.debug(kv("sync rejected", profile=self.profile.key, correct=correct, ratio=det_ratio))
            return None
        hdr_bytes = res.data[SYNC_SYMBOLS:]
        try:
            header = PhyHeader.decode(hdr_bytes, res.erasures[SYNC_SYMBOLS:], res.confidence[SYNC_SYMBOLS:])
        except FrameError as exc:
            self.stats.header_failures += 1
            log.info(kv("PHY header failed", profile=self.profile.key, error=exc.user_message))
            return None
        mask_count: int | None = None
        if header.mode_id != PROBE_MODE_ID and get_mode(header.mode_id).is_ofdm:
            mask = self._lookup_mask(header.mask_id)
            mask_count = mask.count if mask is not None else None
            if mask is None:
                self.stats.header_failures += 1
                log.warning(kv("unknown carrier mask id", mask_id=header.mask_id))
                return None
        lay = body_layout(self.profile, header.mode_id, header.body_len, mask_count)
        total = num.frame_length(lay.body_samples)
        post = num.postamble_start(lay.body_samples)
        slack = int(post * MAX_PPM * 1e-6) + num.pair_tol
        need_end = t0i + post + num.chirp_len + slack + num.taper + 16
        mfsk_snr = float(mfsk.snr_to_db(float(np.mean(res.symbol_snr))))
        return _Pending(
            t0=t0,
            header=header,
            detect_ratio=det_ratio,
            frame_samples_nominal=total,
            post_offset=post,
            mfsk_snr_db=mfsk_snr,
            need_end=need_end,
            raw_cfo_hz=raw_cfo,
        )

    def _lookup_mask(self, mask_id: int) -> CarrierMask | None:
        if mask_id == 0:
            return CarrierMask.all()
        if self.mask_lookup is None:
            return None
        return self.mask_lookup(mask_id)

    def _find_postamble(self, p: _Pending) -> float | None:
        num = self.num
        c = num.chirp_len
        expected = p.t0 + p.post_offset
        slack = int(p.post_offset * MAX_PPM * 1e-6) + num.pair_tol
        lo = int(expected) - slack
        hi = int(expected) + slack + 1
        ctx = int(0.1 * self.profile.sample_rate)
        r0 = max(self._buf_start, lo - ctx)
        r1 = min(self.end_index, hi + c + ctx)
        x = self._slice(r0, r1)
        env = self.detector.envelope_up(self.detector.bandpass(x))
        a, b = lo - r0, hi - r0
        if a < 0 or b > len(env) or b <= a:
            return None
        med = float(np.median(env))
        j = a + int(np.argmax(env[a:b]))
        if not self.detector.passes(float(env[j]), med, self.detector.detect_ratio * POSTAMBLE_RATIO_FACTOR):
            return None
        j = self.detector.first_arrival(env, j)
        # A lone up-chirp is displaced by -cfo/k (delay-Doppler coupling); undo it
        # using the frequency offset measured from the preamble up/down pair.
        coupling = p.raw_cfo_hz / num.chirp_rate_hz_per_s * self.profile.sample_rate
        return r0 + self.detector.refine(env, j) + coupling

    def _decode_pending(self) -> RxFrame:
        p = self._pending
        assert p is not None
        self._pending = None
        info = RxInfo(
            profile_key=self.profile.key,
            start_index=int(p.t0),
            end_index=int(p.t0) + p.frame_samples_nominal,
            detect_ratio=p.detect_ratio,
            header=p.header,
            mfsk_snr_db=p.mfsk_snr_db,
        )
        post = self._find_postamble(p)
        ratio = self.session_ratio
        if post is not None:
            r = ratio_from_span(post - p.t0, p.post_offset)
            if ratio_is_plausible(r):
                ratio = r
                info.postamble_found = True
                self.session_ratio = (1 - RATIO_EMA) * self.session_ratio + RATIO_EMA * r
        if not info.postamble_found:
            self.stats.postamble_missing += 1
        info.ratio = ratio
        info.end_index = int(p.t0 + p.frame_samples_nominal * ratio)
        try:
            body = self._demod_body(p, ratio, info)
        except (FecError, FrameError, IndexError) as exc:
            self.stats.body_failures += 1
            msg = exc.user_message if isinstance(exc, FecError | FrameError) else str(exc)
            log.info(kv("frame body failed", profile=self.profile.key, mode=p.header.mode_id, error=msg))
            return RxFrame(None, info, msg)
        self.stats.frames_ok += 1
        log.debug(
            kv(
                "frame decoded",
                mode=p.header.mode_id,
                body=p.header.body_len,
                snr=info.snr_db,
                ppm=info.ppm,
                cfo=info.cfo_hz,
                corrected=info.rs_corrected,
            )
        )
        return RxFrame(body, info)

    def _demod_body(self, p: _Pending, ratio: float, info: RxInfo) -> bytes:
        num = self.num
        c = num.chirp_len
        pad = c // 2
        t0i = int(np.floor(p.t0))
        seg_start = max(self._buf_start, t0i - pad)
        seg_end = min(self.end_index, t0i + int(np.ceil(p.frame_samples_nominal * ratio)) + pad)
        seg = self._slice(seg_start, seg_end).copy()
        seg = resample_by_ratio(seg, ratio)
        # Re-time on the corrected segment.
        approx = (p.t0 - seg_start) / ratio
        win0 = max(0, int(approx) - 3 * num.pair_tol)
        win1 = min(len(seg), int(approx) + 2 * c + 4 * num.pair_tol)
        env_up, env_dn = self.detector.envelopes(self.detector.bandpass(seg[win0:win1]))
        a = max(0, int(approx) - win0 - num.pair_tol)
        b = min(len(env_up), int(approx) - win0 + num.pair_tol + 1)
        if b > a:
            pi = self.detector.first_arrival(env_up, a + int(np.argmax(env_up[a:b])))
            d0, d1 = pi + c - num.pair_tol, min(len(env_dn), pi + c + num.pair_tol + 1)
            qi = self.detector.first_arrival(env_dn, d0 + int(np.argmax(env_dn[d0:d1])))
            pf = self.detector.refine(env_up, pi)
            qf = self.detector.refine(env_dn, qi)
            t0 = win0 + 0.5 * (pf + qf - c)
            cfo = self.detector.cfo_from_pair(pf, qf)
        else:
            t0 = approx
            cfo = 0.0
        info.cfo_hz = cfo
        if abs(cfo) >= CFO_CORRECT_MIN_HZ:
            seg = frequency_shift(seg, -cfo, self.profile.sample_rate)
        t0r = round(t0)
        hdr = p.header
        mode = get_mode(hdr.mode_id)
        body_start = t0r + num.body_start
        if hdr.mode_id == PROBE_MODE_ID or not mode.is_ofdm:
            lay = body_layout(self.profile, hdr.mode_id, hdr.body_len)
            starts = body_start + np.arange(lay.mfsk_symbols) * num.sym_len
            res = mfsk.demodulate(seg, starts, num, self.erasure_ratio)
            info.snr_db = res.snr_db
            info.mfsk_snr_db = res.snr_db
            info.bin_signal = res.bin_signal
            info.bin_noise = res.bin_noise
            if hdr.mode_id == PROBE_MODE_ID:
                train = body_start + lay.mfsk_symbols * num.sym_len
                silence = train + 4 * num.sym_len
                try:
                    info.probe_snr_db = ofdm.analyze_probe(seg, train, silence, num.probe_silence_len, num)
                except IndexError:
                    info.probe_snr_db = None
            body, st = decode_body(res.data, hdr.body_len, mode.nsym, res.erasures, res.confidence)
        else:
            mask = self._lookup_mask(hdr.mask_id)
            if mask is None:
                raise FrameError("unknown carrier mask")
            car = ofdm.layout(num, mask)
            lay = body_layout(self.profile, hdr.mode_id, hdr.body_len, mask.count)
            ores = ofdm.demodulate_body(seg, body_start, lay.coded_len, mode.bits_per_carrier, car, num)
            info.snr_db = ores.snr_db
            full = np.full(self.profile.n_candidate_carriers, np.nan)
            full[mask.as_array()] = ores.carrier_snr_db
            info.carrier_snr_db = full
            body, st = decode_body(ores.data, hdr.body_len, mode.nsym, ores.erasures, ores.confidence)
        info.rs_corrected = st.corrected
        info.erasures = st.erasures_used
        return body


class MultiPipeline:
    """Runs one :class:`ReceiverPipeline` per profile on the same capture stream."""

    def __init__(self, pipelines: list[ReceiverPipeline]) -> None:
        if not pipelines:
            raise ValueError("at least one pipeline required")
        rates = {p.profile.sample_rate for p in pipelines}
        if len(rates) != 1:
            raise ValueError("all pipelines must share the capture sample rate")
        self.pipelines = pipelines

    @property
    def sample_rate(self) -> int:
        """Capture sample rate."""
        return self.pipelines[0].profile.sample_rate

    @property
    def end_index(self) -> int:
        """Absolute index one past the newest sample."""
        return self.pipelines[0].end_index

    def feed(self, samples: FloatArray, start_index: int | None = None) -> None:
        """Feed every pipeline."""
        for p in self.pipelines:
            p.feed(samples, start_index)

    def add_mute(self, start: int, end: int) -> None:
        """Mute an interval on every pipeline."""
        for p in self.pipelines:
            p.add_mute(start, end)

    def process(self) -> list[RxFrame]:
        """Process all pipelines; frames are returned in start-time order."""
        out: list[RxFrame] = []
        for p in self.pipelines:
            out.extend(p.process())
        out.sort(key=lambda f: f.info.start_index)
        return out

    def busy(self) -> bool:
        """True if any pipeline is mid-frame."""
        return any(p.busy() for p in self.pipelines)

    def set_detect_ratio(self, ratio: float) -> None:
        """Change the CFAR threshold on all pipelines."""
        for p in self.pipelines:
            p.set_detect_ratio(ratio)

    def recent(self, n: int) -> FloatArray:
        """Newest captured samples."""
        return self.pipelines[0].recent(n)

    def restrict(self, profile_key: str) -> None:
        """Keep only the pipeline for ``profile_key`` (after a session locks a profile)."""
        keep = [p for p in self.pipelines if p.profile.key == profile_key]
        if keep:
            self.pipelines = keep
