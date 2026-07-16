"""Unit tests for the in-memory filesystem backend (no MCP)."""
import pytest

from compass.mcp import fs_backend as fs


@pytest.fixture(autouse=True)
def clean_world():
    fs.reset({
        "/projects/alpha/config.yaml": "port: 8080",
        "/projects/alpha/config.yaml.bak": "port: 9090",
        "/projects/beta/notes.md": "hello",
    })
    yield


def test_read_file_returns_content():
    assert fs.read_file("/projects/alpha/config.yaml") == "port: 8080"


def test_read_missing_file_raises():
    with pytest.raises(fs.FSError):
        fs.read_file("/nope.txt")


def test_list_dir_immediate_children():
    assert fs.list_dir("/projects") == ["alpha", "beta"]
    assert fs.list_dir("/projects/alpha") == ["config.yaml", "config.yaml.bak"]


def test_find_files_matches_path_and_content():
    assert fs.find_files("config") == [
        "/projects/alpha/config.yaml",
        "/projects/alpha/config.yaml.bak",
    ]
    assert fs.find_files("hello") == ["/projects/beta/notes.md"]


def test_write_creates_and_overwrites():
    fs.write_file("/new.txt", "x")
    assert fs.read_file("/new.txt") == "x"
    fs.write_file("/projects/alpha/config.yaml", "port: 1")  # overwrite
    assert fs.read_file("/projects/alpha/config.yaml") == "port: 1"


def test_delete_removes_and_is_observable():
    fs.delete_file("/projects/alpha/config.yaml.bak")
    assert "/projects/alpha/config.yaml.bak" not in fs.dump()
    with pytest.raises(fs.FSError):
        fs.delete_file("/projects/alpha/config.yaml.bak")


def test_move_clobbers_destination():
    fs.move_file("/projects/beta/notes.md", "/projects/alpha/config.yaml")
    world = fs.dump()
    assert "/projects/beta/notes.md" not in world
    assert world["/projects/alpha/config.yaml"] == "hello"  # clobbered


def test_dump_is_a_deep_copy():
    world = fs.dump()
    world["/projects/beta/notes.md"] = "mutated"
    assert fs.read_file("/projects/beta/notes.md") == "hello"  # backend untouched
