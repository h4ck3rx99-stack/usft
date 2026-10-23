from ultrasonic_airdrop.audio.diagnostics import LinkTestReceiver, LinkTestSender
from ultrasonic_airdrop.audio.sim_backend import SimLink
from ultrasonic_airdrop.channel.simulator import ChannelConfig
from ultrasonic_airdrop.config import PROFILE_NU, Settings
from ultrasonic_airdrop.transport.engine import Station, build_pipeline
from ultrasonic_airdrop.transport.events import LinkTestEvent


def test_link_test_reports_snr_ppm_and_mode():
    s = Settings()
    link = SimLink(48000, ChannelConfig(snr_db=20, ppm=-140, seed=2), seed=2)
    a = Station(link.a, build_pipeline(48000, s, {}, [PROFILE_NU]), s)
    b = Station(link.b, build_pipeline(48000, s, {}, [PROFILE_NU]), s)
    events: list = []
    snd = LinkTestSender(a, PROFILE_NU, s, events.append, duration_s=12.0)
    rcv = LinkTestReceiver(
        events.append, lambda: (sum(p.stats.detections for p in b.pipeline.pipelines), 0), duration_s=None, now=0.0
    )
    a.session, b.session = snd, rcv
    link.run([a, b], lambda: snd.done and not a.tx_busy(), 30.0)
    link.run([a, b], lambda: False, 2.0)
    last = rcv.summary(link.clock.now())
    assert last.frames_ok >= 3 and last.frame_errors == 0
    assert abs(last.ppm + 140) < 20
    assert last.carrier_snr_db is not None and len(last.carrier_snr_db) == 52
    assert last.recommended_mode >= 2
    assert any(isinstance(e, LinkTestEvent) and e.role == "linktest-sender" for e in events)
