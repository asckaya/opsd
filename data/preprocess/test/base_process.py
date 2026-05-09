import random
import string
from collections.abc import Callable

HARD_MATH_SYSTEM = "You are a helpful assistant that solves difficult math problems.\nReason step by step.\nAt the end, output the final answer in the format \\boxed{answer}.\nThe last line of your response should be of the following format: '###Response \\boxed{answer}' (without quotes)."

MATH_SYSTEM = "You are a helpful assistant that solves math problems.\nReason step by step.\nAt the end, output the final answer in the format \\boxed{answer}.\nThe last line of your response should be of the following format: '###Response \\boxed{answer}' (without quotes)."

MCQ_PROMPT = "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER OR OPTION]' (without quotes). Think step by step before answering.\n\n{question}\n\n{choices}"


def _resolve(key_or_fn: str | Callable, example: dict):
    return key_or_fn(example) if callable(key_or_fn) else example[key_or_fn]


def make_math_process_fn(
    problem_key: str = "problem",
    answer_key: str | Callable = "answer",
    system_prompt: str = HARD_MATH_SYSTEM,
):
    """Returns a process_fn for math datasets with boxed answers."""

    def process_fn(example):
        answer = _resolve(answer_key, example)
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example[problem_key]},
            ],
            "label": str(answer),
            "metadata": {"rm_type": "deepscaler"},
        }

    return process_fn


def make_mcq_process_fn(
    question_key: str | Callable,
    options_key: str | Callable,
    answer_key: str | Callable,
    shuffle: bool = False,
    extra_metadata_fn: Callable | None = None,
):
    """Returns a process_fn for multiple-choice datasets.

    shuffle=False: answer_key should return the correct letter (e.g. "A").
    shuffle=True:  answer_key should return the correct option text; options
                   are shuffled before letters are assigned.
    """

    def process_fn(example):
        question = _resolve(question_key, example)
        options = list(_resolve(options_key, example))
        answer = _resolve(answer_key, example)

        if shuffle:
            correct_text = answer
            random.shuffle(options)
            correct_letter = string.ascii_uppercase[options.index(correct_text)]
        else:
            correct_letter = str(answer).strip().upper()

        choices_str = "\n".join(f"{string.ascii_uppercase[i]}. {opt}" for i, opt in enumerate(options))
        choices_dict = {string.ascii_uppercase[i]: opt for i, opt in enumerate(options)}

        metadata = {
            "choices": choices_dict,
            "correct_letter": correct_letter,
            "rm_type": "gpqa",
        }
        if extra_metadata_fn:
            metadata.update(extra_metadata_fn(example))

        return {
            "prompt": [
                {"role": "user", "content": MCQ_PROMPT.format(question=question, choices=choices_str)},
            ],
            "label": correct_letter,
            "metadata": metadata,
        }

    return process_fn
