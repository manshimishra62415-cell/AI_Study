"""Smart study planner: weighted time allocation + SM-2 spaced repetition. Pure Python."""
from datetime import date, timedelta


def sm2(ease, interval, reps, quality):
    """SuperMemo-2 update. quality is 0-5. Returns (ease, interval_days, reps)."""
    if quality < 3:
        return max(1.3, ease - 0.2), 1, 0
    reps += 1
    interval = 1 if reps == 1 else 6 if reps == 2 else max(1, round(interval * ease))
    ease = max(1.3, ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return ease, interval, reps


def parse_topics(text):
    """'Normalization, 4' per line -> [{'name','difficulty'}]. Difficulty defaults to 3."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, diff = line.rpartition(",") if "," in line else (line, "", "")
        try:
            d = min(5, max(1, int(diff)))
        except ValueError:
            name, d = line, 3
        out.append({"name": name.strip(), "difficulty": d})
    return out


def generate_plan(topics, exam_date, hours_per_day, start=None, block=45):
    """topics: [{'name','difficulty'(1-5),'weakness'(0-1, optional)}]. Returns list of task dicts.
    Harder/weaker topics get more study blocks; the last days are reserved for revision."""
    start = start or date.today()
    days = (exam_date - start).days
    if days < 1:
        raise ValueError("Exam date must be in the future.")
    if not topics:
        raise ValueError("Add at least one topic.")
    cap = int(hours_per_day * 60)
    block = min(block, cap)
    if block < 15:
        raise ValueError("Need at least 15 minutes of study per day.")
    rev_days = min(2, days // 3)
    study_days = days - rev_days
    per_day = cap // block
    total_blocks = study_days * per_day
    if len(topics) > total_blocks:
        raise ValueError("Too many topics for the time available; add days or hours.")

    w = {t["name"]: t["difficulty"] * (1 + t.get("weakness", 0.0)) for t in topics}
    sw = sum(w.values())
    spare = total_blocks - len(topics)
    raw = {n: spare * v / sw for n, v in w.items()}
    alloc = {n: 1 + int(r) for n, r in raw.items()}
    leftover = total_blocks - sum(alloc.values())
    for n in sorted(raw, key=lambda n: raw[n] - int(raw[n]), reverse=True)[:leftover]:
        alloc[n] += 1

    order = sorted(w, key=w.get, reverse=True)
    seq = []
    while any(alloc.values()):  # round-robin interleave so topics are mixed, not blocked
        for n in order:
            if alloc[n]:
                seq.append(n)
                alloc[n] -= 1

    tasks = []
    for i, n in enumerate(seq):
        day = start + timedelta(days=i // per_day)
        tasks.append({"topic": n, "due": day.isoformat(), "minutes": block, "kind": "study"})

    rev_len, used, d = 20, 0, 0
    for n in order if rev_days else []:
        if used + rev_len > cap:
            d, used = d + 1, 0
        if d >= rev_days:
            break
        day = start + timedelta(days=study_days + d)
        tasks.append({"topic": n, "due": day.isoformat(), "minutes": rev_len, "kind": "revision"})
        used += rev_len
    return tasks
