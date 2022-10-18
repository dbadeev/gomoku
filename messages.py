"""
Module with Messages class (displaying errors and success messages)
"""


class Messages:
    """
    Class for displaying success and error messages
    """

    def __init__(self, message: str):
        self.message = message

    # -------------------------------------------------------------------
    def error_(self) -> None:
        """
        Error message (red color)
        """
        print(f"\033[31m{self.message}\033[0m")

    def ok_(self) -> None:
        """
        Success message (green color)
        """
        print(f"\033[32m{self.message}\033[0m")

    def info_(self) -> None:
        """
        Ordinary message (no color)
        """
        print(f"{self.message}")
    # -------------------------------------------------------------------


class ProgramError(Exception):
    """
    Custom error class
    """
