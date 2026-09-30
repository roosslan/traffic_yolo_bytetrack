from datetime import datetime

import pytest

from traffic import app, storage


def test_defaults():
    args = app.build_parser().parse_args([])
    assert args.config == "config.toml"
    assert (args.headless, args.self_test, args.init_db, args.list) == (False, False, False, None)


def test_list_without_count_defaults_to_twenty():
    assert app.build_parser().parse_args(["--list"]).list == 20


def test_list_with_count_and_other_flags():
    args = app.build_parser().parse_args(["--list", "5", "--headless", "--config", "x.toml"])
    assert (args.list, args.headless, args.config) == (5, True, "x.toml")


def test_invalid_count_is_rejected():
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["--list", "много"])


def test_missing_config_exits_with_code_2(tmp_path, caplog):
    assert app.main(["--config", str(tmp_path / "missing.toml")]) == 2
    assert "config.example.toml" in caplog.text


def write_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[camera]\nrtsp_url = "x"\n[clickhouse]\ndatabase = "cams"\n', encoding="utf-8")
    return str(path)


class FakeClient:
    def __init__(self, rows=()):
        self.commands, self.rows, self.queries = [], list(rows), []

    def command(self, sql):
        self.commands.append(sql)

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters))
        return type("Result", (), {"result_rows": self.rows})()


def test_init_db_creates_the_schema(tmp_path, monkeypatch, capsys):
    client = FakeClient()
    monkeypatch.setattr(storage, "default_client_factory", lambda cfg: client)
    assert app.main(["--config", write_config(tmp_path), "--init-db"]) == 0
    assert len(client.commands) == 5
    assert "cams" in client.commands[0]
    assert "cams" in capsys.readouterr().out


def test_list_prints_crossings(tmp_path, monkeypatch, capsys):
    rows = [
        (datetime(2026, 9, 21, 14, 3, 27), "cam1", "west", "north", "car", 5, 0.8, "capture/a.jpg"),
        (datetime(2026, 9, 21, 14, 3, 40), "cam1", "side", "south", "headlights", 6, 0.9, ""),
    ]
    client = FakeClient(rows)
    monkeypatch.setattr(storage, "default_client_factory", lambda cfg: client)
    assert app.main(["--config", write_config(tmp_path), "--list", "2"]) == 0
    out = capsys.readouterr().out
    when = app.local(datetime(2026, 9, 21, 14, 3, 27))
    assert f"{when:%Y-%m-%d %H:%M:%S}" in out and "north" in out and "capture/a.jpg" in out
    assert "headlights #6" in out
    assert client.queries[0][1] == {"n": 2}


def test_counts_prints_cars_per_hour(tmp_path, monkeypatch, capsys):
    rows = [(datetime(2026, 9, 21, 14, 0), "cam1", "west", "north", 42)]
    client = FakeClient(rows)
    monkeypatch.setattr(storage, "default_client_factory", lambda cfg: client)
    assert app.main(["--config", write_config(tmp_path), "--counts", "6"]) == 0
    out = capsys.readouterr().out
    hour = app.local(datetime(2026, 9, 21, 14, 0))
    assert f"{hour:%Y-%m-%d %H:00}" in out and "west" in out and "north" in out and "42" in out
    assert client.queries[0][1] == {"h": 6}


def test_counts_without_hours_defaults_to_a_day():
    assert app.build_parser().parse_args(["--counts"]).counts == 24


def test_format_counts_lists_lines_and_directions_in_config_order():
    from traffic.config import parse_config

    lines = [
        {"name": "west", "points": [[0.3, 0], [0.3, 1]], "directions": ["a", "b"]},
        {"name": "side", "points": [[0.5, 0], [0.6, 0]], "directions": ["in", "out"]},
    ]
    config = parse_config({"camera": {"rtsp_url": "x"}, "lines": lines})
    counts = {("west", "b"): 2, ("side", "in"): 1}
    assert app.format_counts(counts, config) == "west: a 0, b 2; side: in 1, out 0"


def test_list_on_an_empty_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(storage, "default_client_factory", lambda cfg: FakeClient())
    assert app.main(["--config", write_config(tmp_path), "--list"]) == 0
    assert "нет записей" in capsys.readouterr().out


def test_unreachable_clickhouse_gives_exit_code_1(tmp_path, monkeypatch, caplog):
    def refuse(cfg):
        raise ConnectionError("нет связи")

    monkeypatch.setattr(storage, "default_client_factory", refuse)
    assert app.main(["--config", write_config(tmp_path), "--init-db"]) == 1
    assert "ClickHouse недоступен" in caplog.text


def test_self_test_uses_the_model_from_the_config(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(app, "run_self_test", lambda config: seen.append(config))
    path = tmp_path / "config.toml"
    path.write_text('[camera]\nrtsp_url = "x"\n[model]\ndevice = "cuda"\n', encoding="utf-8")
    assert app.main(["--config", str(path), "--self-test"]) == 0
    assert seen[0].model.device == "cuda"


def test_self_test_works_without_a_config(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(app, "run_self_test", lambda config: seen.append(config))
    assert app.main(["--config", str(tmp_path / "нет.toml"), "--self-test"]) == 0
    assert seen[0].model.weights == "models/yolov8n.pt"


def test_self_test_failure_gives_exit_code_1(tmp_path, monkeypatch, caplog):
    def broken(model_cfg):
        raise ImportError("нет torch")

    monkeypatch.setattr(app, "run_self_test", broken)
    assert app.main(["--config", str(tmp_path / "нет.toml"), "--self-test"]) == 1
    assert "Самопроверка не пройдена" in caplog.text


def test_local_treats_naive_time_as_utc():
    from datetime import UTC

    naive = datetime(2026, 9, 21, 14, 0)
    assert app.local(naive) == naive.replace(tzinfo=UTC)
