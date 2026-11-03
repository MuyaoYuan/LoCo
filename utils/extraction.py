import re


def extract_answer_llm(text: str) -> str:
    number_strings = re.findall(r"\S*\d+\S*", text)
    if not number_strings:
        return "INVALID_ANS"

    cleaned_number = "".join(
        char for char in number_strings[-1] if char.isdigit() or char in ".-"
    )

    if cleaned_number.count(".") > 1:
        first_dot_index = cleaned_number.index(".")
        cleaned_number = (
            cleaned_number[: first_dot_index + 1]
            + cleaned_number[first_dot_index + 1 :].replace(".", "")
        )

    if cleaned_number.startswith("-"):
        cleaned_number = "-" + cleaned_number[1:].replace("-", "")
    else:
        cleaned_number = cleaned_number.replace("-", "")

    return cleaned_number


def extract_all_boxed_content(text: str) -> list[str]:
    results = []
    start = 0

    while True:
        start = text.find(r"\boxed{", start)
        if start == -1:
            break

        brace_count = 0
        result = []
        i = start

        while i < len(text):
            char = text[i]
            result.append(char)

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

            if brace_count == 0 and result[-1] == "}":
                break

            i += 1

        results.append("".join(result))
        start = i + 1

    return results
