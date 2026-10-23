from ultrasonic_airdrop.cli import main


def test_estimate_command(make_file, capsys):
    f = make_file("photo.jpg", bytes(range(256)) * 400)
    assert main(["--log-level", "WARNING", "estimate", str(f)]) == 0
    out = capsys.readouterr().out
    assert "mode 0" in out and "mode 5" in out and "broadcast" in out


def test_help_lists_commands(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    for cmd in ("send", "receive", "estimate", "simulate", "check", "linktest", "devices"):
        assert cmd in out
