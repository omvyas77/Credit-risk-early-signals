import pytest

from cres import __main__ as cli
from cres import config


def test_help_lists_every_stage(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for stage in cli.STAGES:
        assert stage in out


def test_unknown_stage_is_rejected():
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["no-such-stage"])
    assert exit_info.value.code == 2


def test_default_config_has_one_seed():
    cfg = config.load()
    assert isinstance(cfg["seed"], int)
    assert cfg["label"]["received_from"] == "2017-05-01"


def test_settings_record_tracks_only_the_sections_a_stage_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAMPS", tmp_path / "stamps")
    cfg = {"seed": 42, "label": {"received_from": "2017-05-01"}, "scorer": {"digits": 7}}
    keys = ("seed", "label")
    current = config.settings(cfg, keys, tokenizer="abc")
    # No record yet: outputs built before records existed count as unchanged.
    assert not config.settings_changed("label", current)
    config.record_settings("label", current)
    assert not config.settings_changed("label", config.settings(cfg, keys, tokenizer="abc"))
    other = {**cfg, "scorer": {"digits": 3}}
    assert not config.settings_changed("label", config.settings(other, keys, tokenizer="abc"))
    assert config.settings_changed(
        "label", config.settings({**cfg, "seed": 1}, keys, tokenizer="abc")
    )
    assert config.settings_changed("label", config.settings(cfg, keys, tokenizer="xyz"))
