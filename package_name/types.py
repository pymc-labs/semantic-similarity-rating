"""
This module defines custom types.

These types are used to ensure type safety and provide clear structure for the
synthetic consumer generation and questionnaire processes.

When passing nested data structures through functions, consider creating a
type here and importing it in the module where it's used. This makes the code
more readable and maintainable.

Avoid:
    def my_function(data: dict[str, Union[list[str], str]]):
        pass

Prefer:
    from package_name.types import DataRaw
    def my_function(data: DataRaw):
        pass
"""

from typing import TypedDict


class DataRaw(TypedDict):
    series: list[str]
    index: str


class DataClean(TypedDict):
    series: list[int]
    index: int
