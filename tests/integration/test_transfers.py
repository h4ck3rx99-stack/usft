"""End-to-end transfers between two simulated endpoints (sample-clock driven, deterministic)."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from ultrasonic_airdrop.channel.simulator import ChannelConfig
from ultrasonic_airdrop.config import Settings, TransferOptions
from ultrasonic_airdrop.transport.events import DoneEvent, FileEvent
from ultrasonic_airdrop.transport.simulate import run_transfer

FAST = Settings(linger_s=3.0)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ch(snr=18.0, ppm=0.0, seed=0, **kw) -> ChannelConfig:
    return ChannelConfig(snr_db=snr, ppm=ppm, seed=seed, **kw)


def files_in(dest: Path) -> dict[str, Path]:
    return {p.name: p for p in dest.iterdir() if p.is_file()}


def test_interactive_random_binary(tmp_path, make_file, rng):
    f = make_file("random.bin", rng.integers(0, 256, 1500, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    res = run_transfer(
        [f],
        dest,
        FAST,
        TransferOptions(compression=False),
        ch(10, 120, 1, rt60_s=0.2),
        ch(10, -120, 2, rt60_s=0.2),
        seed=3,
    )
    assert res.sender_success and res.receiver_success, (res.sender_message, res.receiver_message)
    assert sha(dest / "random.bin") == sha(f)
    assert res.sender_stats["goodput_bps"] > 5


def test_multi_file_unicode_empty_and_one_byte(tmp_path, make_file, rng):
    files = [
        make_file("empty.dat", b""),
        make_file("one.bin", b"\x7f"),
        make_file("spaces & ünïcødé 📄 (1).txt", ("Grüße " * 60).encode()),
        make_file("CON.txt", b"reserved name"),
        make_file("random.bin", rng.integers(0, 256, 300, dtype=np.uint8).tobytes()),
    ]
    dest = tmp_path / "dest"
    res = run_transfer(files, dest, FAST, TransferOptions(), ch(20, 50, 4), ch(20, -50, 5), seed=4)
    assert res.sender_success and res.receiver_success, (res.sender_message, res.receiver_message)
    got = files_in(dest)
    expected_names = {"empty.dat", "one.bin", "spaces & ünïcødé 📄 (1).txt", "_CON.txt", "random.bin"}
    assert set(got) == expected_names
    for src in files:
        name = "_CON.txt" if src.name == "CON.txt" else src.name
        assert sha(got[name]) == sha(src)
    assert not (dest / ".airdrop-partial").exists()


def test_streaming_8k_file_uses_ofdm(tmp_path, make_file, rng):
    f = make_file("big.bin", rng.integers(0, 256, 8 * 1024, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    res = run_transfer([f], dest, FAST, TransferOptions(compression=False), ch(26, 100, 6), ch(26, -100, 7), seed=5)
    assert res.sender_success and res.receiver_success
    assert sha(dest / "big.bin") == sha(f)
    assert res.sender is not None and max(res.sender.mode_history) >= 2


@pytest.mark.slow
def test_large_64k_file(tmp_path, make_file, rng):
    f = make_file("large.bin", rng.integers(0, 256, 64 * 1024, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    res = run_transfer(
        [f],
        dest,
        FAST,
        TransferOptions(compression=False),
        ch(26, 150, 8),
        ch(26, -150, 9),
        seed=6,
        max_seconds=4 * 3600,
    )
    assert res.sender_success and res.receiver_success
    assert sha(dest / "large.bin") == sha(f)


def test_recovery_from_lost_and_corrupted_frames(tmp_path, make_file, rng):
    f = make_file("lossy.bin", rng.integers(0, 256, 1200, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    res = run_transfer(
        [f],
        dest,
        FAST,
        TransferOptions(compression=False),
        ch(8, 100, 10, loss=0.15, burst_rate_hz=0.05),
        ch(15, -100, 11),
        seed=7,
    )
    assert res.sender_success and res.receiver_success, (res.sender_message, res.receiver_message)
    assert sha(dest / "lossy.bin") == sha(f)
    assert res.sender_stats["retransmitted_units"] > 0
    assert res.link is not None and res.link.ch_ab.frames_dropped > 0


def test_lost_acks_trigger_poll(tmp_path, make_file, rng):
    f = make_file("polled.bin", rng.integers(0, 256, 600, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    res = run_transfer(
        [f], dest, FAST, TransferOptions(compression=False), ch(18, 0, 12), ch(18, 0, 13, loss=0.4), seed=8
    )
    assert res.sender_success and res.receiver_success, (res.sender_message, res.receiver_message)
    assert res.sender_stats["polls"] > 0


def test_mode_steps_down_when_snr_degrades(tmp_path, make_file, rng):
    f = make_file("degrade.bin", rng.integers(0, 256, 14000, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    state = {"done": False}

    def degrade(sender, receiver) -> bool:
        if not state["done"] and sender.state.value == "data" and sender.units.acked_bytes() > 0:
            link = sender_link["link"]
            link.ch_ab.noise_sigma *= 10 ** (9 / 20)  # SNR drops by 9 dB mid-transfer
            state["done"] = True
        return False

    sender_link: dict = {}

    def hooks(link, sender, receiver) -> None:
        sender_link["link"] = link

    res = run_transfer(
        [f],
        dest,
        FAST,
        TransferOptions(compression=False),
        ch(24, 50, 14),
        ch(24, -50, 15),
        seed=9,
        hooks=hooks,
        stop_when=degrade,
    )
    assert state["done"]
    assert res.sender_success and res.receiver_success, (res.sender_message, res.receiver_message)
    hist = res.sender.mode_history
    assert max(hist) >= 2 and hist[-1] < max(hist)
    assert sha(dest / "degrade.bin") == sha(f)


def test_mode_steps_up_on_clean_channel(tmp_path, make_file, rng):
    f = make_file("climb.bin", rng.integers(0, 256, 3000, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"

    def hooks(link, sender, receiver) -> None:
        orig = sender._on_report

        def start_low(rep):
            orig(rep)
            sender.rate.mode = 1  # pretend the probe was pessimistic
            sender.mode_history.append(1)

        sender._on_report = start_low

    res = run_transfer(
        [f],
        dest,
        FAST,
        TransferOptions(compression=False),
        ch(28, 30, 16),
        ch(28, -30, 17),
        seed=10,
        hooks=hooks,
    )
    assert res.sender_success and res.receiver_success
    hist = res.sender.mode_history
    assert hist[0] == 1 or hist[1] == 1
    assert max(hist) > 1
    # step-ups are one mode at a time
    ups = [b - a for a, b in itertools.pairwise(hist) if b > a]
    assert all(u == 1 for u in ups)


def test_hash_mismatch_reported_as_failure(tmp_path, make_file, rng):
    f = make_file("tampered.bin", rng.integers(0, 256, 400, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"

    def hooks(link, sender, receiver) -> None:
        orig = sender.reader.read

        def corrupt(fi, start, count):
            data = bytearray(orig(fi, start, count))
            if start == 0:
                data[3] ^= 0xFF  # sender-side corruption after the manifest hash was computed
            return bytes(data)

        sender.reader.read = corrupt

    res = run_transfer(
        [f],
        dest,
        FAST,
        TransferOptions(compression=False),
        ch(20, 0, 18),
        ch(20, 0, 19),
        seed=11,
        hooks=hooks,
    )
    assert not res.sender_success and not res.receiver_success
    assert not (dest / "tampered.bin").exists()
    assert not any(isinstance(e, FileEvent) and e.status == "verified" for e in res.receiver_events)
    assert not any(isinstance(e, DoneEvent) and e.success for e in res.receiver_events + res.sender_events)
    assert list((dest / ".airdrop-partial").glob("*.corrupt"))


def test_cancel_from_sender(tmp_path, make_file, rng):
    f = make_file("c.bin", rng.integers(0, 256, 3000, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    fired = {"x": False}

    def stop_when(sender, receiver) -> bool:
        if not fired["x"] and receiver.sess is not None and receiver.sess.data_frames >= 2:
            sender.cancel()
            fired["x"] = True
        return fired["x"] and receiver.state.value in ("cancelled", "failed", "interrupted")

    res = run_transfer(
        [f],
        dest,
        FAST,
        TransferOptions(compression=False),
        ch(20, 0, 20),
        ch(20, 0, 21),
        seed=12,
        stop_when=stop_when,
    )
    assert res.sender.state.value == "cancelled"
    assert res.receiver.state.value == "cancelled"
    assert not (dest / "c.bin").exists()
    assert list((dest / ".airdrop-partial").glob("*.part"))  # partials kept for resume


def test_cancel_from_receiver(tmp_path, make_file, rng):
    f = make_file("r.bin", rng.integers(0, 256, 3000, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    fired = {"x": False}

    def stop_when(sender, receiver) -> bool:
        if not fired["x"] and receiver.sess is not None and receiver.sess.data_frames >= 2:
            receiver.cancel(discard=True)
            fired["x"] = True
        return fired["x"] and sender.done

    res = run_transfer(
        [f],
        dest,
        FAST,
        TransferOptions(compression=False),
        ch(20, 0, 22),
        ch(20, 0, 23),
        seed=13,
        stop_when=stop_when,
    )
    assert res.receiver.state.value == "cancelled"
    assert res.sender.state.value == "cancelled"
    assert not (dest / ".airdrop-partial").exists() or not list((dest / ".airdrop-partial").glob("*.part"))


def _interrupted_session(tmp_path: Path, f: Path, dest: Path, seed: int):
    settings = FAST.replace(rx_inactivity_s=20.0)
    links: dict = {}

    def hooks(link, sender, receiver) -> None:
        links["link"] = link

    def stop_when(sender, receiver) -> bool:
        if receiver.sess is not None and receiver.sess.data_frames >= 3:
            links["link"].a.muted_output = True  # sender's sound no longer reaches the receiver
        return receiver.state.value == "interrupted" and sender.done

    return run_transfer(
        [f],
        dest,
        settings,
        TransferOptions(compression=False, fixed_mode=1),
        ch(20, 0, seed),
        ch(20, 0, seed + 1),
        seed=seed,
        hooks=hooks,
        stop_when=stop_when,
        max_seconds=1800,
    )


def test_receiver_timeout_keeps_partials(tmp_path, make_file, rng):
    f = make_file("t.bin", rng.integers(0, 256, 2000, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    res = _interrupted_session(tmp_path, f, dest, 24)
    assert res.receiver.state.value == "interrupted"
    assert not res.sender_success and "Link lost" in res.sender_message
    assert list((dest / ".airdrop-partial").glob("*.part"))
    assert list((dest / ".airdrop-partial").glob("*.json"))


def test_resume_after_interruption(tmp_path, make_file, rng):
    f = make_file("resume.bin", rng.integers(0, 256, 2000, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    first = _interrupted_session(tmp_path, f, dest, 26)
    assert first.receiver.state.value == "interrupted"
    res = run_transfer([f], dest, FAST, TransferOptions(compression=False), ch(20, 0, 28), ch(20, 0, 29), seed=27)
    assert res.sender_success and res.receiver_success, (res.sender_message, res.receiver_message)
    assert sha(dest / "resume.bin") == sha(f)
    asm = res.receiver.sess.reassembly.assembler(0)
    assert asm.resumed_units > 0
    # the second session did not resend everything
    total_units = asm.n_units
    sent_units = sum(u.sum() for u in res.sender.units.sent_once)
    assert sent_units < total_units


def test_broadcast_with_loss_then_second_pass_completes(tmp_path, make_file, rng):
    f1 = make_file("b1.bin", rng.integers(0, 256, 700, dtype=np.uint8).tobytes())
    f2 = make_file("b2.txt", b"broadcast me " * 20)
    dest = tmp_path / "dest"
    settings = FAST.replace(broadcast_passes=1, rx_inactivity_s=15.0)
    opts = TransferOptions(broadcast=True, compression=False)
    first = run_transfer([f1, f2], dest, settings, opts, ch(15, 80, 30, loss=0.35), ch(15, 0, 31), seed=31)
    assert first.sender_success  # broadcast finished transmitting
    assert not first.receiver_success  # but frames were lost
    verified_first = {n for n, st, _ in first.receiver.results if st == "OK"}
    assert verified_first != {"b1.bin", "b2.txt"}
    second = run_transfer([f1, f2], dest, settings, opts, ch(15, 80, 32, loss=0.0), ch(15, 0, 33), seed=32)
    assert second.receiver_success, second.receiver_message
    for src in (f1, f2):
        assert sha(dest / src.name) == sha(src)
    resumed = sum(a.resumed_units for a in second.receiver.sess.reassembly.unique())
    assert resumed > 0 or verified_first


def test_encryption_roundtrip(tmp_path, make_file, rng):
    f = make_file("secret.bin", rng.integers(0, 256, 500, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    opts = TransferOptions(encrypt=True, passphrase="tr0ub4dor & 3", compression=True)
    res = run_transfer(
        [f], dest, FAST, opts, ch(20, 40, 34), ch(20, -40, 35), seed=33, receiver_passphrase="tr0ub4dor & 3"
    )
    assert res.sender_success and res.receiver_success, (res.sender_message, res.receiver_message)
    assert sha(dest / "secret.bin") == sha(f)


def test_wrong_passphrase_fails_cleanly(tmp_path, make_file, rng):
    f = make_file("secret.bin", rng.integers(0, 256, 500, dtype=np.uint8).tobytes())
    dest = tmp_path / "dest"
    opts = TransferOptions(encrypt=True, passphrase="right one")
    res = run_transfer(
        [f],
        dest,
        FAST,
        opts,
        ch(20, 0, 36),
        ch(20, 0, 37),
        seed=34,
        receiver_passphrase="wrong one",
        max_seconds=900,
    )
    assert not res.sender_success and not res.receiver_success
    assert "Passphrase mismatch" in res.receiver_message
    assert res.sender.state.value in ("cancelled", "failed")
    assert not dest.exists() or not files_in(dest)


def test_existing_destination_file_not_overwritten(tmp_path, make_file, rng):
    f = make_file("report.txt", b"new content " * 10)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "report.txt").write_bytes(b"precious original")
    res = run_transfer([f], dest, FAST, TransferOptions(), ch(20, 0, 38), ch(20, 0, 39), seed=35)
    assert res.receiver_success
    assert (dest / "report.txt").read_bytes() == b"precious original"
    assert sha(dest / "report (1).txt") == sha(f)
