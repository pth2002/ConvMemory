import json
from pathlib import Path

import numpy as np


def session_number(key):
    return int(key.split("_")[1])


def session_number_from_summary_key(key):
    parts = key.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


def dia_session_id(dia_id):
    if not str(dia_id).startswith("D") or ":" not in str(dia_id):
        return ""
    return "session_" + str(dia_id).split(":", 1)[0][1:]


def simple_token_count(text):
    return len(str(text).split())


def build_raw_turns(sample):
    conversation = sample["conversation"]
    session_keys = [
        key
        for key in conversation
        if key.startswith("session_")
        and not key.endswith("date_time")
        and isinstance(conversation[key], list)
    ]
    session_keys.sort(key=session_number)

    turns = []
    session_to_ids = {}
    for session_key in session_keys:
        for turn in conversation[session_key]:
            dia_id = str(turn["dia_id"])
            speaker = turn.get("speaker", "speaker")
            text = turn.get("text", "")
            session_to_ids.setdefault(session_key, []).append(dia_id)
            turns.append(
                {
                    "id": dia_id,
                    "text": f"{speaker}: {text}",
                    "source_ids": [dia_id],
                    "session_id": session_key,
                    "granularity": "turn",
                }
            )
    return turns, session_to_ids


def build_observation_notes(sample):
    notes = []
    observation = sample.get("observation", {})
    for session_key in sorted(observation, key=session_number_from_summary_key):
        session_id = session_key.replace("_observation", "")
        for speaker, speaker_notes in observation[session_key].items():
            for idx, item in enumerate(speaker_notes):
                if not item:
                    continue
                text = str(item[0])
                source_id = str(item[1]) if len(item) > 1 else ""
                source_ids = [source_id] if source_id else []
                notes.append(
                    {
                        "id": f"obs:{source_id or session_id}:{idx}",
                        "text": f"{speaker}: {text}",
                        "source_ids": source_ids,
                        "session_id": session_id,
                        "granularity": "observation",
                    }
                )
    return notes


def build_session_summaries(sample, session_to_ids):
    notes = []
    for key, value in sorted(
        sample.get("session_summary", {}).items(),
        key=lambda kv: session_number_from_summary_key(kv[0]),
    ):
        session_id = key.replace("_summary", "")
        notes.append(
            {
                "id": f"session_summary:{session_id}",
                "text": str(value),
                "source_ids": list(session_to_ids.get(session_id, [])),
                "session_id": session_id,
                "granularity": "session_summary",
            }
        )
    return notes


def build_event_notes(sample, session_to_ids):
    notes = []
    for key, value in sorted(
        sample.get("event_summary", {}).items(),
        key=lambda kv: session_number_from_summary_key(kv[0]),
    ):
        session_id = key.replace("events_", "")
        date = value.get("date", "")
        for speaker, events in value.items():
            if speaker == "date":
                continue
            for idx, event in enumerate(events):
                notes.append(
                    {
                        "id": f"event:{session_id}:{speaker}:{idx}",
                        "text": f"{date}. {speaker}: {event}",
                        "source_ids": list(session_to_ids.get(session_id, [])),
                        "session_id": session_id,
                        "granularity": "event_summary",
                    }
                )
    return notes


def load_locomo_memory_sets(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = {}
    questions = []
    for sample in data:
        sample_id = str(sample["sample_id"])
        raw_turns, session_to_ids = build_raw_turns(sample)
        dia_ids = {item["id"] for item in raw_turns}
        samples[sample_id] = {
            "raw_turns": raw_turns,
            "observation_notes": build_observation_notes(sample),
            "session_summaries": build_session_summaries(sample, session_to_ids),
            "event_notes": build_event_notes(sample, session_to_ids),
        }

        for qa_idx, qa in enumerate(sample["qa"]):
            evidence = [str(x) for x in qa.get("evidence", []) if str(x) in dia_ids]
            if not evidence:
                continue
            questions.append(
                {
                    "sample_id": sample_id,
                    "question_id": f"{sample_id}::qa{qa_idx}",
                    "question_type": f"category_{qa.get('category', 'unknown')}",
                    "query": str(qa["question"]),
                    "answer": str(qa.get("answer", "")),
                    "gold_ids": evidence,
                    "gold_sessions": sorted({dia_session_id(x) for x in evidence}),
                }
            )
    return samples, questions


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def encode_memory_bank(encoder, memories):
    if not memories:
        return np.zeros((0, 1), dtype=np.float32)
    return normalize_rows(encoder.transform([item["text"] for item in memories]))
