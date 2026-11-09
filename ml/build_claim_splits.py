"""Generate honest claim-level dev and test evaluation datasets for VERITAS."""
import json
import random
from pathlib import Path

from ml.make_sample_data import capital_facts, firm_facts, journalism_facts

DEV_REFUTED_VARIANTS = {
    "H5N1-US-2024": "No human cases of bird flu were detected in the United States in 2024.",
    "Meta-Llama": "Meta kept its Llama model proprietary, licensing it exclusively to enterprise customers.",
    "Champions-League-2024": "Borussia Dortmund defeated Real Madrid to win the Champions League final at Wembley.",
    "Turkey": "Istanbul is the capital of Turkey.",
    "US-China-chip-export": "The US government lifted export restrictions on advanced semiconductor chips to China.",
    "Sirius": "Sirius Corporation was founded in 2016.",
    "Polaris": "Polaris Corporation was founded in 1955.",
    "Israel-Rafah": "Israeli forces withdrew from Rafah in southern Gaza in May 2024 without conducting ground operations.",
    "Thailand": "Chiang Mai is the capital of Thailand.",
    "5G-COVID-disinfo": "Health authorities confirmed that 5G telecommunications infrastructure spreads COVID-19.",
    "semaglutide-approval": "Semaglutide failed to receive regulatory approval as an obesity treatment in the United States and Europe.",
    "France": "Marseille is the capital of France.",
    "EU-AI-Act": "The European Parliament rejected the proposed AI Act in 2024, leaving AI unregulated in the EU.",
    "ivermectin-COVID": "The WHO approved ivermectin as a recommended standard treatment for COVID-19.",
    "India-economy-rank": "Japan maintained its position ahead of India as the world's fourth-largest economy.",
    "wind-farm-cancer-disinfo": "Medical research systematic reviews confirmed that wind turbine electromagnetic fields cause cancer.",
    "WHO-stats-disinfo": "Independent audits revealed that WHO systematically manipulated COVID-19 mortality data to inflate pandemic numbers.",
    "Atlas": "Atlas Corporation was founded in 1963.",
    "Biden-withdrawal": "Joe Biden officially accepted the Democratic nomination to contest the 2024 presidential election.",
    "Pakistan-floods-2022": "Monsoon rains in Pakistan in 2022 caused only minor localized flooding displacing fewer than 5 000 people.",
    "hottest-year-2023": "Global temperature data confirmed that 2023 was cooler than historical 20th-century averages.",
}

DEV_INSUFFICIENT_CLAIMS = [
    "The infected dairy farm worker in Texas was treated with an experimental monoclonal antibody.",
    "Meta acquired an AI hardware startup based in London to optimize Llama training.",
    "Real Madrid announced plans to play exhibition matches in Australia following the Wembley victory.",
    "Ankara established a municipal light rail network operating across 40 underground stations.",
    "The US semiconductor export rules established a permanent subsidy fund for domestic fab construction.",
    "Sirius Corporation expanded its consumer retail operations into Scandinavian markets.",
    "Polaris Corporation merged with an industrial robotics supplier in Chicago.",
    "The Rafah border crossing was permanently closed to all humanitarian cargo by international observers.",
    "Bangkok completed an electric ferry transit project along the Chao Phraya River.",
    "Telecommunications regulators mandated that 5G base stations install specialized biological monitoring filters.",
    "The patent on semaglutide expired in 2024, enabling generic formulations across Europe.",
    "Paris banned all private passenger vehicles from its city center on weekdays.",
    "The European Parliament established a central enforcement headquarters for the AI Act in Brussels.",
    "The WHO sponsored clinical trials of ivermectin across forty hospitals in South America.",
    "The World Bank forecast that India's manufacturing sector will double its export output by 2026.",
    "Offshore wind farm installations require annual sound emission permits issued by coastal health authorities.",
    "The World Health Organization revised its excess mortality methodology during a summit in Geneva.",
    "Atlas Corporation launched an aerospace supply division based in Munich.",
    "Joe Biden delivered a televised farewell address from Philadelphia following his announcement.",
    "The Pakistani government received emergency reconstruction loans from the Asian Development Bank following the 2022 floods.",
    "Global meteorological stations recorded record oceanic surface salinity levels during the summer of 2023.",
]

TEST_REFUTED_VARIANTS = {
    "Australia-climate-emergency": "The Australian parliament rejected the motion declaring a national climate emergency.",
    "Argentina": "Cordoba is the capital of Argentina.",
    "FAO-food-prices": "The Food and Agriculture Organization reported that global food prices surged to record highs for three consecutive months.",
    "Caitlin-Clark-WNBA": "Caitlin Clark was selected tenth overall in the second round of the 2024 WNBA Draft.",
    "Apple-market-cap": "Microsoft became the first company to reach a three trillion dollar market capitalisation.",
    "Pakistan-elections": "Pakistan postponed its general elections indefinitely following an emergency decree.",
    "Pakistan": "Lahore is the capital of Pakistan.",
    "South Africa": "Johannesburg is the administrative capital of South Africa.",
    "Sweden": "Gothenburg is the capital of Sweden.",
    "Spain": "Barcelona is the capital of Spain.",
    "Brazil": "Rio de Janeiro is the capital of Brazil.",
    "Apple-Vision-Pro": "Apple cancelled the Vision Pro project before release due to manufacturing cost overruns.",
    "Nepal": "Pokhara is the capital of Nepal.",
    "Saudi Arabia": "Jeddah is the capital of Saudi Arabia.",
    "Acme": "Acme Corporation was founded in 1945.",
    "US-Russia-sanctions": "The United States lifted all bilateral sanctions on Russian banks and defence contractors.",
    "China-Taiwan-exercises": "China cancelled all naval and air exercises near Taiwan following diplomatic agreements.",
    "Fed-rates": "The Federal Reserve lowered benchmark interest rates to near-zero levels to stimulate lending.",
    "Egypt": "Alexandria is the capital of Egypt.",
    "Ukraine": "Odesa is the capital of Ukraine.",
}

TEST_INSUFFICIENT_CLAIMS = [
    "The Australian parliament established a carbon tariff on liquefied natural gas exports.",
    "Buenos Aires completed a modernization of its international container port facilities.",
    "The UN Food and Agriculture Organization declared emergency grain rationing across eleven nations.",
    "Caitlin Clark signed an exclusive multi-year sponsorship contract with a major European beverage brand.",
    "Apple opened an automated semiconductor testing laboratory in Austin, Texas.",
    "Pakistan implemented an electronic biometric voter registration system for all provinces.",
    "Islamabad hosted an international multilateral summit on South Asian renewable energy transmission.",
    "Pretoria introduced an automated solar metering rebate system for suburban households.",
    "Stockholm converted its municipal public bus fleet entirely to hydrogen fuel cells.",
    "Madrid completed an expansion of its underground thermal heating grid across forty districts.",
    "Brasilia opened a national satellite data archive center operated by the environmental ministry.",
    "Apple announced a specialized enterprise developer edition of the Vision Pro with upgraded memory.",
    "Kathmandu inaugurated a municipal environmental air filtering station in the central valley.",
    "Riyadh completed construction of an indoor winter sports arena in the diplomatic quarter.",
    "Acme Corporation licensed its core sensor patent portfolio to a consortium of automotive suppliers.",
    "The United States opened a diplomatic coordination office in Vilnius to monitor regional sanctions compliance.",
    "China established a joint maritime search and rescue command centre on Hainan Island.",
    "The Federal Reserve commissioned an independent academic study on the macroeconomic effects of quantitative tightening.",
    "Cairo launched a solar-powered irrigation project along agricultural land in the Nile Delta.",
    "Kyiv established an international reconstruction investment council with representatives from twenty nations.",
]

NEGATED_FORM_SUBJS = {
    "5G-COVID-disinfo",
    "ivermectin-COVID",
    "wind-farm-cancer-disinfo",
    "WHO-stats-disinfo",
}


def build_claim_splits(seed=42, out_dir="data/claim_level"):
    rng = random.Random(seed)
    cap = capital_facts()
    firm = firm_facts(random.Random(seed))
    jour = journalism_facts()
    all_facts = cap + firm + jour

    shuffled = all_facts[:]
    rng.shuffle(shuffled)
    n_train = round(len(all_facts) * 0.70)
    n_dev = round(len(all_facts) * 0.15)
    dev_facts = shuffled[n_train:n_train + n_dev]
    test_facts = shuffled[n_train + n_dev:]

    # Assert disjoint facts
    dev_subjs = {str(f.get("subj", f["true_claim"][:40])) for f in dev_facts}
    test_subjs = {str(f.get("subj", f["true_claim"][:40])) for f in test_facts}
    assert dev_subjs.isdisjoint(test_subjs), "Dev and test facts must be strictly disjoint!"

    dev_rows = []
    # 1. Dev SUPPORTED
    for f in dev_facts:
        ev_id = str(f.get("subj", f["true_claim"][:40]))
        row = {
            "claim": f["true_claim"],
            "verdict": "SUPPORTED",
            "justifying_evidence_ids": [ev_id],
            "source": "generator-true",
        }
        if ev_id in NEGATED_FORM_SUBJS or "No " in f["true_claim"] or " not " in f["true_claim"]:
            row["tags"] = ["negated_form"]
        dev_rows.append(row)

    # 2. Dev REFUTED
    for f in dev_facts:
        ev_id = str(f.get("subj", f["true_claim"][:40]))
        assert ev_id in DEV_REFUTED_VARIANTS, f"Missing refuted variant for {ev_id}"
        dev_rows.append({
            "claim": DEV_REFUTED_VARIANTS[ev_id],
            "verdict": "REFUTED",
            "justifying_evidence_ids": [ev_id],
            "source": "generator-false-variant",
        })

    # 3. Dev INSUFFICIENT
    assert len(DEV_INSUFFICIENT_CLAIMS) >= 20
    for claim in DEV_INSUFFICIENT_CLAIMS:
        dev_rows.append({
            "claim": claim,
            "verdict": "INSUFFICIENT",
            "justifying_evidence_ids": [],
            "source": "synthetic-out-of-corpus",
        })

    test_rows = []
    # 1. Test SUPPORTED
    for f in test_facts:
        ev_id = str(f.get("subj", f["true_claim"][:40]))
        row = {
            "claim": f["true_claim"],
            "verdict": "SUPPORTED",
            "justifying_evidence_ids": [ev_id],
            "source": "generator-true",
        }
        if ev_id in NEGATED_FORM_SUBJS or "No " in f["true_claim"] or " not " in f["true_claim"]:
            row["tags"] = ["negated_form"]
        test_rows.append(row)

    # 2. Test REFUTED
    for f in test_facts:
        ev_id = str(f.get("subj", f["true_claim"][:40]))
        assert ev_id in TEST_REFUTED_VARIANTS, f"Missing refuted variant for {ev_id}"
        test_rows.append({
            "claim": TEST_REFUTED_VARIANTS[ev_id],
            "verdict": "REFUTED",
            "justifying_evidence_ids": [ev_id],
            "source": "generator-false-variant",
        })

    # 3. Test INSUFFICIENT
    assert len(TEST_INSUFFICIENT_CLAIMS) >= 20
    for claim in TEST_INSUFFICIENT_CLAIMS:
        test_rows.append({
            "claim": claim,
            "verdict": "INSUFFICIENT",
            "justifying_evidence_ids": [],
            "source": "synthetic-out-of-corpus",
        })

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "dev_claims.jsonl", "w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_p / "test_claims.jsonl", "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    return dev_rows, test_rows


if __name__ == "__main__":
    dev, test = build_claim_splits()
    print(f"Generated dev_claims: {len(dev)} (SUPPORTED={sum(1 for r in dev if r['verdict']=='SUPPORTED')}, REFUTED={sum(1 for r in dev if r['verdict']=='REFUTED')}, INSUFFICIENT={sum(1 for r in dev if r['verdict']=='INSUFFICIENT')})")
    print(f"Generated test_claims: {len(test)} (SUPPORTED={sum(1 for r in test if r['verdict']=='SUPPORTED')}, REFUTED={sum(1 for r in test if r['verdict']=='REFUTED')}, INSUFFICIENT={sum(1 for r in test if r['verdict']=='INSUFFICIENT')})")
