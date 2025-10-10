"""
Text handling specific to LLM output
"""

import re
from typing import Iterable

from text.textutils import ALL_DASHES_TRANS_TABLE, standardize_chars_basic, translate_text


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

ABBREV_PATT = re.compile(r'\(((?:\w|-)+)\)')

# Seq of any non-word chars, including underscore ('_')
NONWORD_CHARS_PATT = re.compile(r"(?:\W|_)+")


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class NormalizedFirstIn:

    def __init__(self, name: str = None):

        if name is None:
            name = ""

        self.name = name

        self.excluded_names = [self.name]
        self.alt_names: set[str] = set()
        self.normalized_names: set[str] = {normalize_name(name)}

        return

    def exclude_name(self, excl_name: str):
        self.normalized_names.add(normalize_name(excl_name))
        return

    def exclude_from(self, excl_names: Iterable[str]):
        for nm in excl_names:
            self.exclude_name(nm)
        return

    def update_from(self, excl_names: Iterable[str], new_names: Iterable[str]):
        self.exclude_from(excl_names)
        self.add_from(new_names)
        return

    def add(self, new_name: str):
        normzd_new_name = normalize_name(new_name)
        if normzd_new_name in self.normalized_names:
            return
        self.normalized_names.add(normzd_new_name)
        self.alt_names.add(new_name)
        return

    def add_from(self, new_names: Iterable[str]):
        for nm in new_names:
            self.add(nm)
        return

    def is_empty(self):
        return len(self.alt_names) == 0

    def get_values(self) -> set[str]:
        return self.alt_names

    def get_values_list(self) -> list[str]:
        # sort to make it deterministic
        return sorted(self.alt_names)

# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def show_diff(s1, s2):
    if s1 == s2:
        print('They are the same')
        return

    print(f"'{s1}'")
    print(f"'{s2}'")

    if len(s1) != len(s2):
        print('Lengths are different')
        print()

    diff_chars = []
    print(' ', end='')

    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 == c2:
            print(' ', end='')
        else:
            print('x', end='')
            diff_chars.append(i)
    print()
    print()

    for i in diff_chars:
        c1 = s1[i]
        c2 = s2[i]
        print(f"[{i}]: '{c1}' (ord={ord(c1)}) != '{c2}' (ord={ord(c2)})")

    return


def standardize_name(name: str) -> str:
    name = standardize_chars_basic(name)
    # Replace any punctuation-dash chars with '-'
    name = name.translate(ALL_DASHES_TRANS_TABLE)
    return name


def parse_abbreviation(token: str) -> str | None:
    """
    IF token contains an abbreviation (see `ABBREV_PATT`) THEN return the abbreviation
    ELSE return None
    """
    if m := ABBREV_PATT.match(token):
        abbrev = m.group(1)
        return abbrev

    return None


def parse_examples(tokens: list[str] | str) -> tuple[list[str] | None, list[str] | None]:
    """
    Check if the tokens end with a list of examples in the form '(e.g. ex1, ..., ex_n)'
    :param tokens:
    :return:
        - List of examples
        - remaining tokens after removing examples text
    """

    if isinstance(tokens, str) == 1:
        tokens = tokens.split()

    if len(tokens) < 3 or not tokens[-1].endswith(")") or tokens[-1].startswith("("):
        return None, None

    # Look for token '(e.g.'
    ex_start = 0
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i] in ["(e.g.", "(e.g.,"]:
            ex_start = i
            break

    if ex_start == 0:
        return None, None

    # tokens[ex_start : ] are "(e.g.", "ex1,", ... "ex_n)"
    ex_text = " ".join(tokens[ex_start + 1:])
    if ex_text.endswith(")"):
        ex_text = ex_text[:-1]

    examples = re.split(r',\s+', ex_text)
    remaining_tokens = tokens[: ex_start]

    return examples, remaining_tokens


def parse_standardize_name(name: str) -> tuple[str, list[str], list[str]]:
    """
    Standardizes the Unicode characters where possible (see `standardize_name`).
    Parses the name, and
    :return: main_name, alt_names, examples
        - Main name
        - List of alternative names
        - List of examples

    Example 1:
        name = 'Peroxisome Proliferator‐Activated Receptor Gamma (PPAR‐γ) Agonists (PPARG-A) (e.g. AB1, CD ef-2x)'
        returned:   'Peroxisome Proliferator-Activated Receptor Gamma Agonists',
                    ['PPARG-A', 'PPAR-γ Agonists'],
                    ['AB1', 'CD ef-2x']

    Example 2:
        name = 'Just a simple Name'
        returned:   ('Just a simple Name', [], [])
    """

    name = standardize_name(name)

    alt_names = set()
    examples = None

    tokens = name.split()

    # Check for Examples and Abbreviation of whole name
    if len(tokens) > 1:
        examples, remaining_tokens = parse_examples(tokens)
        if remaining_tokens:
            tokens = remaining_tokens

        if abbrev := parse_abbreviation(tokens[-1]):
            alt_names.add(abbrev)
            tokens = tokens[:-1]

    # If there is an intermediate abbrev, then that adds an alternative name
    # To keep things simple, we chack for one abbrev

    for j, tkn in enumerate(tokens[1:-1]):
        i = j + 1
        if abbrev := parse_abbreviation(tkn):
            sfx = " ".join(tokens[i + 1:])
            alt_names.add(f"{abbrev} {sfx}")
            del tokens[i]
            break

    main_name = " ".join(tokens)

    # Use `sorted` to make output deterministic
    return main_name, sorted(alt_names), examples or []


def generate_alt_names(sttdized_name: str, is_sub_call=False) -> list[str]:
    """
    Call after `parse_standardize_name`, to generate a list of alternative names.
    All this does is replace any hyphen ('-') with SPACE or Empty String.

    The goal is to be comprehensive, so alt_names may not be always be valid names.

    :param sttdized_name: The name
    :param is_sub_call: For internal use only.
        Recursive calls set this to True.
    """

    hidx = sttdized_name.find('-', 1)

    if hidx < 0 or hidx == len(sttdized_name) - 1:
        return [sttdized_name] if is_sub_call else []

    alt_sfxs = generate_alt_names(sttdized_name[hidx + 1:], is_sub_call=True)

    pfx = sttdized_name[: hidx]

    alt_names = []
    for char in ['-', ' ', '']:
        for sfx in alt_sfxs:
            if char == '' and pfx[-1].islower():
                sfx = sfx[0].lower() + sfx[1:]
            alt_names.append(pfx + char + sfx)

    if not is_sub_call and alt_names[0] == sttdized_name:
        alt_names = alt_names[1:]

    return alt_names


def normalize_name(name: str, to_lower=True) -> str | None:
    """
    Normalizes a name:
        - Non-word chars, including '_', are considered equivalent to SPACE
        - Special non-ASCII chars are converted to ascii equivalent chars / words.
    """

    name = name.strip()

    if name is None:
        return None

    name = translate_text(standardize_name(name))

    if to_lower:
        name = name.casefold()

    tokens = NONWORD_CHARS_PATT.split(name)

    return " ".join(tokens)


def abbreviate_normalize_name(name: str, abbrev_max_len = 20) -> str:

    name = re.sub(r'\s+', ' ', name.strip())

    name_abbrev = normalize_name(name).replace(' ', '_')[:abbrev_max_len]
    if len(name) > abbrev_max_len:
        name_abbrev += "Z"

    return name_abbrev
