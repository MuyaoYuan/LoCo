import re

try:
    from word2number import w2n
except ImportError:
    w2n = None


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _fix_sqrt(string):
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def convert_word_number(text: str) -> str:
    if w2n is None:
        return text
    try:
        return str(w2n.word_to_num(text))
    except Exception:
        return text


unit_texts = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m",
    "deg", "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd",
    "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab",
    "way", "west", "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .",
    "t", "a", "h", "no change", "men", "soldier", "pie", "bc", "excess",
    "st", "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre",
    "rise", "a . m", "th", "π r 2", "sq", "mark", "l", "toy", "coin",
    "sq . m", "gallon", "° f", "profit", "minw", "yr", "women", "feet", "am",
    "pm", "hr", "cu cm", "square", "v â € ™", "are", "rupee", "rounds",
    "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day", "hour",
    "minute", "min", "second", "man", "woman", "sec", "cube", "mt",
    "sq inch", "mp", "∏ cm ³", "hectare", "more", "sec", "unit", "cu . m",
    "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m", "cm", "mm",
    "apple", "liter", "loss", "yard", "pure", "year", "increase", "decrease",
    "d", "less", "Surface", "litre", "pi sq m", "s .", "metre", "meter", "inch",
]
unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")

    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")

    stripped = re.sub(r"\\text{.*?}$", "", string).strip()
    if stripped and stripped != string:
        string = stripped

    for unit_text in unit_texts:
        stripped = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
        if stripped != "":
            string = stripped

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    string = convert_word_number(string)
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)

    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")
    string = string.replace("\\%", "").replace(r"\%", "").replace("%", "")
    string = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        "",
        string,
        flags=re.IGNORECASE,
    )
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if (
        string.startswith("{") and string.endswith("}") and string.isalnum()
        or string.startswith("(") and string.endswith(")") and string.isalnum()
        or string.startswith("[") and string.endswith("]") and string.isalnum()
    ):
        string = string[1:-1]

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string
