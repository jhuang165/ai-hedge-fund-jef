"""The app is exercised by hand, not by tests — but it must at least import.
A renamed constant elsewhere broke `aihf` with no arguments once already."""


def test_app_imports():
    import hedge_fund.tui.app  # noqa: F401
