"""
互換エントリ。将棋GUIからは usi.py / bat/usi.bat を推奨。
"""

from usi import TournamentConfig, build_player, main, parse_args, run

__all__ = [
    "TournamentConfig",
    "build_player",
    "main",
    "parse_args",
    "run",
]


if __name__ == "__main__":
    main()
