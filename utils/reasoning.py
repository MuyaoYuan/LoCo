import re

from utils.extraction import extract_all_boxed_content, extract_answer_llm
from utils.grader import check_is_correct


PROMPT_SUFFIX = r" Please reason step by step, and put your final answer within \boxed{}."
FALLBACK_PATTERNS = [
    r"<answer>(.*?)</answer>",
    r"</answer>(.*?)</answer>",
    r"<answer>(.*?)<answer>",
    r"\*\*Answer:\*\* ([\d\.]+)",
]


def build_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem + PROMPT_SUFFIX}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_prediction(response: str):
    boxed_matches = extract_all_boxed_content(response)
    if boxed_matches:
        prediction = boxed_matches[-1]
        if prediction and "\\boxed" in prediction:
            return prediction.replace("\\boxed{", "")[:-1]

    for pattern in FALLBACK_PATTERNS:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1]

    return None


def split_response(response: str) -> tuple[str, str]:
    if "</think>" not in response:
        return response, ""

    cot_response, final_response = response.split("</think>", 1)
    return cot_response + "</think>", final_response.strip()


def is_numeric_match(prediction: str, expected_answer: str) -> bool:
    pred_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
    expected_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", expected_answer)
    return bool(
        pred_numbers
        and expected_numbers
        and float(pred_numbers[0]) == float(expected_numbers[0])
    )


def answers_match(prediction: str | None, expected_answer: str | None) -> bool:
    if prediction is None or expected_answer is None:
        return False
    return check_is_correct(prediction, expected_answer) or is_numeric_match(
        prediction,
        expected_answer,
    )


def evaluate_prediction(response: str, expected_answer: str):
    cot_response, final_response = split_response(response)
    prediction_cot = extract_prediction(cot_response)
    prediction_final = extract_prediction(final_response)

    if prediction_cot is None:
        prediction_cot = extract_answer_llm(cot_response)
    if prediction_final is None and final_response:
        prediction_final = extract_answer_llm(final_response)

    has_boxed_answer_cot = len(extract_all_boxed_content(cot_response)) > 0
    has_boxed_answer_final = len(extract_all_boxed_content(final_response)) > 0
    accept = answers_match(prediction_cot, expected_answer)
    final_matches_cot = answers_match(prediction_final, prediction_cot)
    is_complete = has_boxed_answer_cot and has_boxed_answer_final

    return {
        "cot_response": cot_response,
        "final_response": final_response,
        "prediction_cot": prediction_cot,
        "prediction_final": prediction_final,
        "has_boxed_answer_cot": has_boxed_answer_cot,
        "has_boxed_answer_final": has_boxed_answer_final,
        "final_matches_cot": final_matches_cot,
        "is_complete": is_complete,
        "accept": accept,
    }


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))
