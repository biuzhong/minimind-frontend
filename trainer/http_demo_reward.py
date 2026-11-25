"""Scoring shared by the HTTP teaching environment and held-out evaluation."""

import json


def answer_checks(response, expected):
    target = json.loads(expected)
    if type(target) is not dict or set(target) != {"code", "action"}:
        raise ValueError("Expected a JSON reference with code and action")
    raw = response.strip()
    try:
        answer = json.loads(raw)
    except ValueError:
        return False, False, False, False
    if type(answer) is not dict:
        return False, False, False, False
    correct_code = type(answer.get("code")) is int and answer["code"] == target["code"]
    correct_action = answer.get("action") == target["action"]
    proper_shape = ("\n" not in raw and "\r" not in raw and set(answer) == {"code", "action"}
                    and type(answer.get("code")) is int and type(answer.get("action")) is str)
    return True, correct_code, correct_action, proper_shape


def score_answer(response, expected):
    parsed, code, action, shape = answer_checks(response, expected)
    # Partial credit gives malformed and partially correct samples different rewards.
    raw = response.strip()
    return (
        0.1 * raw.startswith("{")
        + 0.1 * ('"code"' in raw)
        + 0.1 * ('"action"' in raw)
        + 0.4 * parsed
        + 1.0 * code
        + 0.7 * action
        + 0.4 * shape
    )
