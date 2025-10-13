"""
Misc utils
"""

import contextlib
import dataclasses
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import re
import sys
from typing import get_origin, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

NO_DEFAULT = object()
"""Default value for required fields in sub-class of `ValidatedDataclass`."""


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class ValidatedDataclass:
    """
    Enforces fields to have valid values:
        - correct type
        - if field is defined using: dataclasses.field(metadata=dict(choices=[...])),
          then field-val must be one of the choices.
    """

    optsdir: str = None
    """
    Path to the dir containing the options file, if loaded using ``from_json_file`. 
    """

    def __post_init__(self):
        """Enforce validate is called after instance is created."""
        self.validate()
        return

    def validate(self):
        """
        Raises an exception if a field value is invalid.
        """
        for fld in dataclasses.fields(self):
            fval = getattr(self, fld.name)
            if fval is not None:
                if fval is NO_DEFAULT:
                    raise TypeError(f"Missing required argument for: '{fld.name}'")
                elif (origin_type := get_origin(fld.type)) is None:
                    if not isinstance(fval, fld.type):
                        raise TypeError(f"Value of {fld.name}={fval} must be of type {fld.type}")
                else:
                    # Is a subscripted type like `List[str]`. Only checks the main type (`List` in the example).
                    if not isinstance(fval, origin_type):
                        raise TypeError(f"Value of {fld.name}={fval} must be of type {fld.type}")

            if choices := fld.metadata.get("choices"):
                if fval not in choices:
                    raise ValueError(f"Value of {fld.name}={fval} must be one of {choices}")

        return

    def assert_matches(self, other, skip_fields: List[str] = None):
        """
        Like `__eq__`, but raises informative ValueError if objects are unequal.
        :param other:
        :param skip_fields: IF provided THEN skip these fields when matching
        """
        assert self.__class__ is other.__class__, \
            ValueError(f"Classes do not match: {self.__class__.__name__} != {other.__class__.__name__}")

        if skip_fields is None:
            skip_fields = []

        for fld in dataclasses.fields(self):
            if fld.name in skip_fields:
                continue
            if getattr(self, fld.name) != getattr(other, fld.name):
                raise ValueError(f"Values of field '{self.__class__.__name__}.{fld.name}' do not match.")
        return True

    def to_json_file(self, json_file: str):
        with open(json_file, "w") as jf:
            json.dump(dataclasses.asdict(self), jf, indent=4)
        return

    @classmethod
    def from_json_file(cls, json_file: str):
        with open(json_file) as jf:
            jdict = json.load(jf)

        jdict["optsdir"] = os.path.split(json_file)[0]

        # noinspection PyArgumentList
        return cls(**jdict)
# /


class NpEncoder(json.JSONEncoder):
    """
    Usage: json.dump(f, data, cls=NpEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def check_output_file_dir(file_path: str,
                          *,
                          error_if_exists=False,
                          create_dir=True,
                          create_parents=True,
                          verbose=True):
    """
    Check if `dir_path` exists as a directory, and if not, create it, if requested.
    Otherwise, if the dir does not exist, raises an Error

    :param str file_path: Path to a file
    :param error_if_exists: Raise error if file exists
    :param create_dir: Create the dir if it doesn't exist?
    :param bool create_parents: If True, Then create all ancestors as needed.
    :param bool verbose:
    """
    path = Path(file_path)
    if path.exists():
        if error_if_exists:
            raise FileExistsError(file_path)
        elif verbose:
            print("The following file exists and will be overwritten:", file_path)
            print(flush=True)

        return

    path = path.parent

    if not path.exists():
        if create_dir:
            if verbose:
                print('Creating dir:', path)
                print(flush=True)
            path.mkdir(parents=create_parents)
        else:
            raise FileNotFoundError("Directory does not exist: " + str(path))

    elif not path.is_dir():
        raise NotADirectoryError("Path exists but is not a dir: " + str(path))

    return


def fn_name(fn):
    """Return str name of a function or method."""
    s = str(fn)
    if s.startswith('<function'):
        return 'fn:' + fn.__name__
    else:
        return ':'.join(s.split()[1:3])


def pp_funcargs(fn):
    arg_names = inspect.getfullargspec(fn).args
    print(fn_name(fn), "... args:")

    frame = inspect.currentframe()
    try:
        for i, name in enumerate(arg_names, start=1):
            if i == 1 and name == "self":
                continue

            val = frame.f_back.f_locals.get(name)
            if isinstance(val, str) and " " in val:
                val = f"'{val}'"
            print("   ", name, "=", val)

        print(flush=True)

    finally:
        # This is needed to ensure any reference cycles are deterministically removed as early as possible
        # see doc: https://docs.python.org/3/library/inspect.html#the-interpreter-stack
        del frame
    return


def print_cmd():
    import os
    import re
    import sys

    module = os.path.relpath(sys.argv[0], ".")
    module = module.replace("/", ".")
    module = re.sub(r"\.py$", "", module)

    args = [f"'{a}'" if " " in a else a
            for a in sys.argv[1:]]
    print("$>", "python -m", module, *args)
    print()
    return


def pp_arr(msg, arr: np.ndarray, print_value=False, indent: str = ""):
    dtype = arr.dtype
    print(indent, msg, ": type = ", type(arr), ", shape = ", arr.shape, ", dtype = ", dtype, sep="")
    if print_value:
        for line in str(arr).splitlines():
            print(indent, line, sep="")
    print()
    return


def pp_dict(d, msg=None, indent=4, precision=4):
    if msg:
        print(msg, '-' * len(msg), sep='\n')

    for k, v in d.items():
        if isinstance(v, Mapping):
            print('{:{indent}s}{}: {{'.format('', k, indent=indent))
            pp_dict(v, None, indent + 4)
            print('{:{indent}s}}}'.format('', indent=indent))
        else:
            if isinstance(v, (float, np.floating)):
                v = '{:.{w}f}'.format(v, w=precision)
            print('{:{indent}s}{}: {}'.format('', k, v, indent=indent))
    return


@contextlib.contextmanager
def buffered_stdout():
    """
    Use this to buffer print()'s until end of block.
    Useful when printing from processes, to keep entire print tnsrnm together.

    >>> with buffered_stdout():
    >>>     print("abc")
    >>>     print("123")
    """
    exc = None
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        try:
            yield
        except Exception as e:
            # Catch exception, so curr output can be printed before raising the exception
            exc = e

        msg = buf.getvalue()

    print(msg, flush=True)

    # Raise the caught exception, if any
    if exc is not None:
        raise exc

    return


@contextlib.contextmanager
def suppressed_stdout():
    """
    Use this to suppress all STDOUT.

    >>> with suppressed_stdout():
    >>>     # The following print stmts will not produce anything
    >>>     print("abc")
    >>>     print("123")
    """
    exc = None
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        try:
            yield
        except Exception as e:
            # Catch exception, so curr output can be printed before raising the exception
            exc = e

    # Raise the caught exception, if any
    if exc is not None:
        raise exc

    return


def get_longest(l: List[str]) -> Optional[str]:
    """
    Return the longest str (first one encountered if many qualify).
    This seems to be the fastest way to do this.
    """
    if not l:
        return None

    x, xl = l[0], len(l[0])
    for y in l[1:]:
        if (yl := len(y)) > xl:
            x, xl = y, yl

    return x


def is_interactive_mode():
    """
    Whether program is running in interactive mode.
    False means Batch Mode or in a Notebook.
    """
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def is_batch_mode():
    return not is_interactive_mode()


def pp_underlined_hdg(hdg, overline=False, linechar="-", file=None):
    if overline:
        print(linechar * len(hdg), file=file)
    print(hdg, linechar * len(hdg), sep="\n", file=file)
    print(file=file)
    return


def split_camel_case(txt: str) -> List[str]:
    """
    Splits `txt` on spaces, and Splits each component word into sub-words if its in camel-case.
    :return: List of words
    """
    if " " in txt:
        return list(itertools.chain.from_iterable(split_camel_case(ww) for ww in txt.split()))

    w = []
    m = 0
    for i in range(1, len(txt)):
        if txt[i].isupper() and txt[i-1].islower():
            w.append(txt[m:i])
            m = i
    if not w:
        return [txt]
    else:
        return w + [txt[m:]]


def capitalize_words(txt: str) -> str:
    """
    Friendlier version of `string.capwords()` which
    - does not capitalize words that are not pure alpha
    - only capitalizes words that are in lower-case (skips if word in mixed or upper case)
    """
    def capitalize(ww: str):
        if ww.isalpha() and ww.islower():
            return ww.capitalize()
        else:
            return ww

    return " ".join([capitalize(w)  for w in txt.split()])


def english_join(txt_list: List[str]) -> Optional[str]:
    """
    Join list of names into an English seq, e.g. "a, b and c"
    """
    if not txt_list:
        return None
    if len(txt_list) == 1:
        return txt_list[0]

    joined = ", ".join(txt_list[:-1])
    joined += " and " + txt_list[-1]

    return joined


def reset_df_index(df: pd.DataFrame, restart: int = 1) -> pd.DataFrame:
    """Convenience function"""
    df = df.reset_index(drop=True)
    if restart != 0:
        df.index += restart
    return df


def ppmd_counts_df(df: pd.DataFrame,
                   counts_col: Optional[str] = None,
                   floatfmt: Union[str, List[str]] = ".1%",
                   add_pct_total=False,
                   add_cum_pct_total=False,
                   sort_on_counts=False):
    """
    Prints df in markdown format, followed by an empty line.

    :param df: DataFrame
    :param counts_col: Column name that contains the 'counts' to be used below.
    :param add_pct_total: Adds the columns 'pctTotal'.
    :param add_cum_pct_total: Adds the columns 'pctTotal' and 'cumTotal'. Sorted on decreasing count.
    :param floatfmt: If different formats needed for each float column, then provide ordered list of formats.
    :param sort_on_counts: Sorted on decreasing count.
    """

    if add_cum_pct_total:
        add_pct_total = True

    if add_pct_total or sort_on_counts:
        assert counts_col in df.columns, "Must provide a valid `counts_col` when `add_pct_total` or `sort_on_counts`"

    if add_cum_pct_total or sort_on_counts:
        df = reset_df_index(df.sort_values(counts_col, ascending=False))

    if add_cum_pct_total or add_pct_total:
        df['cumTotal'] = df[counts_col].cumsum()

        if add_pct_total:
            df['pctTotal'] = df[counts_col] / df[counts_col].sum()

        df['cum_pct'] = df.pctTotal.cumsum()

    print(re.sub(r"( nan%? +)\|", lambda match: (" " * (len(match.group(0)) - 4)) + "-- |",
                 df.to_markdown(intfmt=',', floatfmt=floatfmt)))
    print()
    return
