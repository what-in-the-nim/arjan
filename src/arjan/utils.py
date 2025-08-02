from pathlib import Path
from typing import Iterable, Optional


def list_files(
    directory: str | Path,
    white_exts: Optional[Iterable[str]] = None,
    black_exts: Optional[Iterable[str]] = None,
) -> list[Path]:
    """
    List all files in a directory

    Parameters
    ----------
    directory : str or Path
        The directory to search in.
    white_exts : Optional[Iterable[str]]
        List of allowed file extensions. Will return only files with these extensions.
        If None, all files are allowed.
    black_exts : Optional[Iterable[str]]
        List of disallowed file extensions. Will exclude files with these extensions.
        If None, no files are excluded.

    Returns
    -------
    list[Path]
        List of file paths that match the criteria.
    """
    if isinstance(directory, str):
        directory = Path(directory)

    files = [f for f in directory.rglob("*") if f.is_file()]

    if white_exts is not None:
        white_exts = {ext.lower() for ext in white_exts}
        return [f for f in files if f.suffix.lower() in white_exts]

    if black_exts is not None:
        black_exts = {ext.lower() for ext in black_exts}
        files = [f for f in files if f.suffix.lower() not in black_exts]

    return files
