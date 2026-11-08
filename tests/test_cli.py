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
