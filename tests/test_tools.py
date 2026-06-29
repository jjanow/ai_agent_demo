"""Tests for agent.tools."""

from types import SimpleNamespace

import pytest

from agent import tools


# --------------------------------------------------------------------------- #
# calculator
# --------------------------------------------------------------------------- #


def test_calculator_valid_expression():
    res = tools.calculator("2 * (3 + 4) ** 2")
    assert res["success"] is True
    assert res["result"]["value"] == 98


def test_calculator_with_math_function():
    res = tools.calculator("sqrt(16) + log(e)")
    assert res["success"] is True
    assert res["result"]["value"] == pytest.approx(5.0)


def test_calculator_division_by_zero():
    res = tools.calculator("1 / 0")
    assert res["success"] is False
    assert "zero" in res["error"].lower()


def test_calculator_injection_attempt():
    res = tools.calculator("__import__('os').system('ls')")
    assert res["success"] is False
    assert res["result"] is None


def test_calculator_empty_string():
    res = tools.calculator("")
    assert res["success"] is False
    assert "empty" in res["error"].lower()


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


def test_read_file_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    res = tools.read_file("hello.txt")
    assert res["success"] is True
    assert res["result"]["content"] == "hello world"


def test_read_file_path_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = tools.read_file("../../etc/passwd")
    assert res["success"] is False
    assert ".." in res["error"] or "traversal" in res["error"].lower()


def test_read_file_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = tools.read_file("/etc/passwd")
    assert res["success"] is False


def test_read_file_nonexistent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = tools.read_file("does_not_exist.txt")
    assert res["success"] is False
    assert "not found" in res["error"].lower()


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #


def test_write_file_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tools.write_file("out.txt", "first")["success"] is True
    # Second write without confirm should be refused.
    res = tools.write_file("out.txt", "second")
    assert res["success"] is False
    assert "exists" in res["error"].lower()
    # With confirmation it succeeds.
    assert tools.write_file("out.txt", "second", confirm_overwrite=True)["success"] is True
    assert (tmp_path / "out.txt").read_text() == "second"


# --------------------------------------------------------------------------- #
# get_datetime
# --------------------------------------------------------------------------- #


def test_get_datetime():
    res = tools.get_datetime()
    assert res["success"] is True
    assert res["result"]["timezone"] == "UTC"


# --------------------------------------------------------------------------- #
# web_search (mocked Tavily)
# --------------------------------------------------------------------------- #


class _FakeTavilyOK:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, query, max_results=3):
        return {
            "results": [
                {"title": "T1", "content": "snippet one", "url": "http://a"},
                {"title": "T2", "content": "snippet two", "url": "http://b"},
                {"title": "T3", "content": "snippet three", "url": "http://c"},
                {"title": "T4", "content": "snippet four", "url": "http://d"},
            ]
        }


class _FakeTavilyFail:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, query, max_results=3):
        raise RuntimeError("tavily down")


def test_web_search_mocked(monkeypatch):
    monkeypatch.setattr(tools, "settings", SimpleNamespace(tavily_api_key="x"))
    monkeypatch.setattr("tavily.TavilyClient", _FakeTavilyOK)
    res = tools.web_search("python news")
    assert res["success"] is True
    assert len(res["result"]["results"]) == 3
    assert res["result"]["results"][0]["title"] == "T1"


def test_web_search_api_failure(monkeypatch):
    monkeypatch.setattr(tools, "settings", SimpleNamespace(tavily_api_key="x"))
    monkeypatch.setattr("tavily.TavilyClient", _FakeTavilyFail)
    res = tools.web_search("python news")
    assert res["success"] is False
    assert "failed" in res["error"].lower()


def test_web_search_missing_key(monkeypatch):
    monkeypatch.setattr(tools, "settings", SimpleNamespace(tavily_api_key=""))
    res = tools.web_search("python news")
    assert res["success"] is False


def test_web_search_empty_query():
    res = tools.web_search("")
    assert res["success"] is False


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def test_dispatch_unknown_tool():
    res = tools.dispatch_tool("nope", {})
    assert res["success"] is False
    assert "unknown" in res["error"].lower()


def test_dispatch_bad_input_type():
    res = tools.dispatch_tool("calculator", "not a dict")
    assert res["success"] is False
