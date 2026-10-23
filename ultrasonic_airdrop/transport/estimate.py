"""Frame sizing, window sizing and pre-transfer airtime estimates (all via ``frame_airtime``)."""

from __future__ import annotations

import math

from ultrasonic_airdrop.config import (
    CONTROL_MODE,
    MAX_FRAME_AIRTIME_S,
    MAX_OFDM_BODY_AIRTIME_S,
    UNIT_SIZE,
    WINDOW_AIRTIME_CAP_S,
    Profile,
    Settings,
    get_mode,
)
from ultrasonic_airdrop.crypto.aead import TAG_LEN
from ultrasonic_airdrop.link.codec import frame_airtime, max_body_len
from ultrasonic_airdrop.link.frames import DATA_HEADER_LEN, OVERHEAD, link_frame_len
from ultrasonic_airdrop.phy.numerology import numerology

ACK_PAYLOAD_MAX = 24 + 1 + 4 * 8
#: Allowance for detection latency on the far side before it can reply.
DETECT_LATENCY_S = 0.6


def frame_cap_s(profile: Profile, mode_id: int) -> float:
    """Maximum frame airtime for ``mode_id`` (6 s; OFDM bodies limited to 4 s)."""
    if not get_mode(mode_id).is_ofdm:
        return MAX_FRAME_AIRTIME_S
    num = numerology(profile)
    overhead = num.frame_length(0) / profile.sample_rate
    return min(MAX_FRAME_AIRTIME_S, MAX_OFDM_BODY_AIRTIME_S + overhead)


def max_units_per_frame(profile: Profile, mode_id: int, n_enabled: int | None, encrypted: bool) -> int:
    """Units (32 B) per DATA frame so that the frame respects the airtime cap (1..255)."""
    body = max_body_len(profile, mode_id, frame_cap_s(profile, mode_id), n_enabled)
    room = body - OVERHEAD - DATA_HEADER_LEN - (TAG_LEN if encrypted else 0)
    return max(1, min(255, room // UNIT_SIZE))


def max_control_payload(profile: Profile, encrypted: bool) -> int:
    """Largest mode-0 payload (bytes) within the frame airtime cap."""
    body = max_body_len(profile, CONTROL_MODE, MAX_FRAME_AIRTIME_S)
    return body - OVERHEAD - (TAG_LEN if encrypted else 0)


def data_frame_airtime(profile: Profile, mode_id: int, n_enabled: int | None, encrypted: bool, units: int) -> float:
    """Airtime of a DATA frame with ``units`` full units."""
    payload = DATA_HEADER_LEN + units * UNIT_SIZE + (TAG_LEN if encrypted else 0)
    return frame_airtime(profile, mode_id, link_frame_len(payload), n_enabled)


def window_frames(settings: Settings, profile: Profile, mode_id: int, n_enabled: int | None, encrypted: bool) -> int:
    """Frames per window: configured W limited so the window airtime stays <= 20 s."""
    w = settings.window_ofdm if get_mode(mode_id).is_ofdm else settings.window_mfsk
    units = max_units_per_frame(profile, mode_id, n_enabled, encrypted)
    air = data_frame_airtime(profile, mode_id, n_enabled, encrypted, units)
    return max(1, min(w, int(WINDOW_AIRTIME_CAP_S // air)))


def ack_airtime(profile: Profile, encrypted: bool) -> float:
    """Worst-case ACK airtime."""
    return frame_airtime(profile, CONTROL_MODE, link_frame_len(ACK_PAYLOAD_MAX + (TAG_LEN if encrypted else 0)))


def reply_wait_s(profile: Profile, settings: Settings, reply_airtime: float) -> float:
    """How long to wait after our transmission ends for a reply of ``reply_airtime``."""
    return DETECT_LATENCY_S + settings.reply_delay_s + reply_airtime + settings.output_latency_s + settings.ack_extra_s


def estimate_transfer_s(
    profile: Profile,
    tx_sizes: list[int],
    mode_id: int,
    settings: Settings,
    encrypted: bool = False,
    broadcast: bool = False,
    n_enabled: int | None = None,
    manifest_len: int = 200,
) -> float:
    """Airtime estimate for a whole transfer at a fixed mode, excluding retransmissions."""
    units_pf = max_units_per_frame(profile, mode_id, n_enabled, encrypted)
    full_air = data_frame_airtime(profile, mode_id, n_enabled, encrypted, units_pf)
    frames = 0
    data_air = 0.0
    for size in tx_sizes:
        units = math.ceil(size / UNIT_SIZE)
        n = math.ceil(units / units_pf) if units else 0
        frames += n
        data_air += n * full_air
    gap = 0.05
    offer_air = frame_airtime(profile, CONTROL_MODE, link_frame_len(100))
    part = max_control_payload(profile, encrypted) - 4
    parts = max(1, math.ceil(manifest_len / max(part, 1)))
    part_air = frame_airtime(profile, CONTROL_MODE, link_frame_len(part + 4 + (TAG_LEN if encrypted else 0)))
    end_air = frame_airtime(profile, CONTROL_MODE, link_frame_len(0))
    if broadcast:
        passes = settings.broadcast_passes
        return 3 * offer_air + 2 * parts * part_air + passes * (data_air + frames * gap) + 2 * end_air
    turnaround = settings.output_latency_s + DETECT_LATENCY_S + settings.reply_delay_s
    ack_air = ack_airtime(profile, encrypted)
    w = window_frames(settings, profile, mode_id, n_enabled, encrypted)
    windows = math.ceil(frames / w) if frames else 0
    m_windows = math.ceil(parts / settings.window_mfsk)
    probe_air = frame_airtime(profile, 0xFE, link_frame_len(0))
    report_air = frame_airtime(profile, CONTROL_MODE, link_frame_len(64))
    accept_air = frame_airtime(profile, CONTROL_MODE, link_frame_len(12))
    handshake = offer_air + accept_air + probe_air + report_air + 4 * turnaround
    manifest = parts * part_air + m_windows * (ack_air + 2 * turnaround)
    data = data_air + frames * gap + windows * (ack_air + 2 * turnaround)
    end = end_air + ack_air + 2 * turnaround
    return handshake + manifest + data + end
