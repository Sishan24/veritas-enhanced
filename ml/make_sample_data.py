"""Generate a small synthetic dataset so the whole pipeline runs without FEVER.

This is a SMOKE-TEST dataset (templated facts). Real results require FEVER; see data/README.md.
"""
import json
import random
from pathlib import Path

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo",
    "India": "New Delhi", "Egypt": "Cairo", "Canada": "Ottawa", "Brazil": "Brasilia", "Kenya": "Nairobi",
    "Norway": "Oslo", "Sweden": "Stockholm", "Greece": "Athens", "Portugal": "Lisbon", "Austria": "Vienna",
    "Poland": "Warsaw", "Peru": "Lima", "Chile": "Santiago", "Cuba": "Havana", "Turkey": "Ankara",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Nepal": "Kathmandu", "Iran": "Tehran", "Ireland": "Dublin",
}
FIRMS = ["Acme", "Borealis", "Cygnus", "Delphi", "Orion", "Pavo", "Quanta", "Rigel", "Sirius", "Tauri", "Vega", "Zephyr"]


def facts(rng):
    out = []
    for country, cap in CAPITALS.items():
        out.append(dict(claim=f"{cap} is the capital of {country}.", evidence=f"The capital city of {country} is {cap}.", subj=country, true=cap, kind="cap"))
    for f in FIRMS:
        y = rng.randint(1950, 2015)
        out.append(dict(claim=f"{f} Corporation was founded in {y}.", evidence=f"{f} Corporation was established in {y} by two engineers.", subj=f, true=y, kind="firm"))
    return out


def make_rows(rng, n):
    base, rows = facts(rng), []
    caps = list(CAPITALS.values())
    for _ in range(n):
        f = rng.choice(base)
        r = rng.random()
        if r < 0.34:
            rows.append(dict(claim=f["claim"], evidence=f["evidence"], label="SUPPORTED"))
        elif r < 0.67:
            if f["kind"] == "cap":
                wrong = rng.choice([c for c in caps if c != f["true"]])
                ev = f"The capital city of {f['subj']} is {wrong}."
            else:
                wrong = f["true"] + rng.choice([-9, -4, 3, 7, 12])
                ev = f"{f['subj']} Corporation was established in {wrong} by two engineers."
            rows.append(dict(claim=f["claim"], evidence=ev, label="REFUTED"))
        else:
            other = rng.choice([g for g in base if g["subj"] != f["subj"]])
            rows.append(dict(claim=f["claim"], evidence=other["evidence"], label="NOT_ENOUGH_INFO"))
    return rows


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    rng = random.Random(7)
    write("data/sample/train.jsonl", make_rows(rng, 4000))
    write("data/sample/dev.jsonl", make_rows(rng, 600))
    write("data/sample/test.jsonl", make_rows(rng, 600))
    ev = [dict(id=f"cap-{c}", text=f["evidence"]) for c, f in zip(CAPITALS, facts(rng)[: len(CAPITALS)])]
    ev += [dict(id="demo-1", text="The Eiffel Tower was completed in 1889 and stands in Paris."),
           dict(id="demo-2", text="Mount Everest is the highest mountain above sea level, at 8,849 metres.")]
    write("data/sample/evidence.jsonl", ev)
    print("Wrote data/sample/{train,dev,test,evidence}.jsonl")
