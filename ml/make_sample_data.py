"""Generate a large journalism-focused synthetic dataset for VERITAS training.

Covers: world politics, elections, economy, climate, health, technology,
conflict, sport and social-media disinformation -- the daily bread of
fact-checkers and news desks.

Labels:  SUPPORTED | REFUTED | NOT_ENOUGH_INFO
Format:  {"claim": str, "evidence": str, "label": str}

Run:
    python -m ml.make_sample_data

Outputs (in data/sample/):
    train.jsonl   ~8 000 rows
    dev.jsonl       900 rows
    test.jsonl      900 rows
    evidence.jsonl  passage store used by the API retriever
"""
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# SECTION 1: Capital cities (smoke-test subset from original)
# ---------------------------------------------------------------------------
CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "Japan": "Tokyo", "India": "New Delhi", "Egypt": "Cairo", "Canada": "Ottawa",
    "Brazil": "Brasilia", "Kenya": "Nairobi", "Norway": "Oslo", "Sweden": "Stockholm",
    "Greece": "Athens", "Portugal": "Lisbon", "Austria": "Vienna", "Poland": "Warsaw",
    "Peru": "Lima", "Chile": "Santiago", "Cuba": "Havana", "Turkey": "Ankara",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Nepal": "Kathmandu", "Iran": "Tehran",
    "Ireland": "Dublin", "Argentina": "Buenos Aires", "Saudi Arabia": "Riyadh",
    "South Africa": "Pretoria", "Ukraine": "Kyiv", "Pakistan": "Islamabad",
    "Bangladesh": "Dhaka", "Nigeria": "Abuja", "Ethiopia": "Addis Ababa",
    "Indonesia": "Jakarta", "Philippines": "Manila",
}
WRONG_CAPS = list(CAPITALS.values())

FIRMS_LIST = [
    "Acme", "Borealis", "Cygnus", "Delphi", "Orion", "Pavo",
    "Quanta", "Rigel", "Sirius", "Tauri", "Vega", "Zephyr",
    "Nexus", "Apex", "Vertex", "Meridian", "Zenith", "Horizon",
    "Atlas", "Polaris",
]

# ---------------------------------------------------------------------------
# SECTION 2: Current-affairs journalism templates
# Each entry: true_claim, true_ev (SUPPORTS), false_ev (REFUTES),
#             unrelated_ev (NOT_ENOUGH_INFO placeholder)
# ---------------------------------------------------------------------------

POLITICS = [
    {"true_claim": "The United Nations Security Council held an emergency session on the Gaza ceasefire proposal.",
     "true_ev": "The UN Security Council convened an emergency session to debate a ceasefire resolution for the Gaza Strip after months of intense fighting.",
     "false_ev": "The UN General Assembly, not the Security Council, met informally to discuss the Gaza ceasefire proposal in a closed-door session.",
     "unrelated_ev": "The United Nations has 193 member states and maintains several bodies responsible for international peace and security.",
     "subj": "UNSC-Gaza"},

    {"true_claim": "France called for a formal ceasefire at the United Nations Security Council.",
     "true_ev": "France formally tabled a ceasefire resolution at the UN Security Council, demanding an immediate halt to hostilities.",
     "false_ev": "France abstained from the UN Security Council vote and did not formally call for a ceasefire.",
     "unrelated_ev": "France is a permanent member of the UN Security Council and holds veto power over binding resolutions.",
     "subj": "France-UNSC"},

    {"true_claim": "The United States imposed new sanctions on Russia over the war in Ukraine.",
     "true_ev": "Washington announced a fresh package of sanctions targeting Russian state banks and defence contractors in response to continued military operations in Ukraine.",
     "false_ev": "The United States lifted earlier sanctions on Russia following diplomatic talks in Helsinki.",
     "unrelated_ev": "The United States has maintained a sanctions regime against various countries since the Cold War era.",
     "subj": "US-Russia-sanctions"},

    {"true_claim": "Ukraine received additional military aid from NATO member states in 2024.",
     "true_ev": "Multiple NATO countries pledged artillery, ammunition and air-defence systems to Ukraine as part of a coordinated support package agreed at the alliance summit.",
     "false_ev": "NATO officially suspended military assistance to Ukraine pending a ceasefire agreement signed in Warsaw.",
     "unrelated_ev": "NATO was founded in 1949 as a collective security alliance among Western democracies.",
     "subj": "Ukraine-NATO-aid"},

    {"true_claim": "India won a non-permanent seat on the UN Security Council.",
     "true_ev": "India was elected by the UN General Assembly to serve a two-year term as a non-permanent member of the Security Council.",
     "false_ev": "India failed to secure enough votes in the UN General Assembly and lost the Security Council seat bid to Pakistan.",
     "unrelated_ev": "The UN Security Council has five permanent members and ten rotating non-permanent members.",
     "subj": "India-UNSC"},

    {"true_claim": "Brazil's president addressed the UN General Assembly in September.",
     "true_ev": "Brazilian President Lula da Silva delivered the opening address at the UN General Assembly in New York, calling for reform of global governance institutions.",
     "false_ev": "Brazil sent only a junior minister to the UN General Assembly as the president was unable to travel due to a health emergency.",
     "unrelated_ev": "It is a long-standing tradition that Brazil is the first country to speak at the UN General Assembly each year.",
     "subj": "Brazil-UNGA"},

    {"true_claim": "The European Union agreed on a new migration pact.",
     "true_ev": "EU member states and the European Parliament reached a historic deal on the EU Pact on Migration and Asylum, establishing common rules for processing asylum seekers.",
     "false_ev": "Negotiations on the EU migration pact collapsed after Hungary and Poland vetoed the final text.",
     "unrelated_ev": "The European Union currently has 27 member states following the United Kingdom's departure in 2020.",
     "subj": "EU-migration-pact"},

    {"true_claim": "China conducted military exercises near Taiwan in 2024.",
     "true_ev": "The People's Liberation Army launched large-scale naval and air exercises encircling Taiwan, described by Beijing as a warning against separatist forces.",
     "false_ev": "China cancelled planned military exercises near Taiwan following back-channel diplomatic assurances from Taipei.",
     "unrelated_ev": "Taiwan has its own government and military but China considers the island part of its territory under the one-China policy.",
     "subj": "China-Taiwan-exercises"},

    {"true_claim": "The International Criminal Court issued an arrest warrant for a sitting head of state.",
     "true_ev": "The ICC issued an arrest warrant for a sitting head of government, marking one of the few times the court has targeted an incumbent leader.",
     "false_ev": "The ICC closed its preliminary investigation without issuing any arrest warrants following a state cooperation agreement.",
     "unrelated_ev": "The International Criminal Court was established by the Rome Statute and began operating in 2002.",
     "subj": "ICC-warrant"},

    {"true_claim": "Pakistan held general elections amid widespread protests.",
     "true_ev": "Pakistan conducted parliamentary elections under heavy security, with opposition parties alleging widespread irregularities and supporters of the PTI party protesting across major cities.",
     "false_ev": "Pakistan postponed its general elections indefinitely following a Supreme Court ruling that the electoral commission lacked quorum.",
     "unrelated_ev": "Pakistan is a federal parliamentary republic with a bicameral parliament comprising the National Assembly and the Senate.",
     "subj": "Pakistan-elections"},

    {"true_claim": "The US Congress passed a 95 billion dollar foreign aid package.",
     "true_ev": "The US House of Representatives approved a 95 billion dollar supplemental aid package for Ukraine, Israel and Taiwan, which was subsequently signed into law.",
     "false_ev": "The US foreign aid bill was defeated in the House after a procedural vote blocked its consideration.",
     "unrelated_ev": "The US Congress consists of the Senate and the House of Representatives and is responsible for passing legislation and approving budgets.",
     "subj": "US-foreign-aid"},

    {"true_claim": "Iran launched a direct missile and drone attack on Israel.",
     "true_ev": "Iran launched more than 300 ballistic missiles and drones at Israel in an unprecedented direct attack, most of which were intercepted by Israeli and allied air defences.",
     "false_ev": "Iran's attack on Israel consisted solely of a symbolic volley of drones that were recalled before crossing Israeli airspace.",
     "unrelated_ev": "Iran and Israel have been in a state of hostility for decades, with Iran refusing to recognise Israel as a legitimate state.",
     "subj": "Iran-Israel-attack"},

    {"true_claim": "Benjamin Netanyahu addressed a joint session of the US Congress in 2024.",
     "true_ev": "Israeli Prime Minister Benjamin Netanyahu addressed a joint session of the US Congress in Washington, receiving a standing ovation from many legislators.",
     "false_ev": "Netanyahu's planned address to Congress was cancelled after the Speaker rejected the invitation citing ongoing legal proceedings.",
     "unrelated_ev": "Addresses to joint sessions of Congress by foreign leaders require an invitation from congressional leadership.",
     "subj": "Netanyahu-Congress"},

    {"true_claim": "Donald Trump was convicted of felony charges in New York.",
     "true_ev": "Donald Trump was found guilty on all 34 felony counts of falsifying business records in a New York state court, becoming the first former US president convicted of a crime.",
     "false_ev": "The New York jury acquitted Donald Trump on all counts following the judge's decision to dismiss key prosecution witnesses.",
     "unrelated_ev": "Donald Trump served as the 45th President of the United States from 2017 to 2021.",
     "subj": "Trump-conviction"},

    {"true_claim": "Joe Biden withdrew from the 2024 US presidential race.",
     "true_ev": "President Joe Biden announced he was ending his re-election campaign and endorsed Vice President Kamala Harris as the Democratic nominee.",
     "false_ev": "Joe Biden formally secured the Democratic nomination at the party convention after defeating primary challengers.",
     "unrelated_ev": "The US presidential election is held every four years on the first Tuesday after the first Monday in November.",
     "subj": "Biden-withdrawal"},
]

ECONOMY = [
    {"true_claim": "The US Federal Reserve raised interest rates to a 22-year high.",
     "true_ev": "The Federal Reserve lifted its benchmark interest rate to a range not seen in over two decades as it continued its campaign against persistent inflation.",
     "false_ev": "The Federal Reserve cut interest rates by 50 basis points following signs that inflation had returned to its 2 percent target.",
     "unrelated_ev": "The Federal Reserve System was created by the Federal Reserve Act of 1913 and consists of twelve regional Reserve Banks.",
     "subj": "Fed-rates"},

    {"true_claim": "Global food prices fell for the third consecutive month according to the FAO.",
     "true_ev": "The Food and Agriculture Organization reported that its food price index declined for the third month in a row, driven by lower cereal and vegetable oil prices.",
     "false_ev": "The FAO food price index rose sharply for the third month running, driven by drought-related wheat shortages.",
     "unrelated_ev": "The FAO food price index tracks monthly changes in international prices of commonly traded food commodities.",
     "subj": "FAO-food-prices"},

    {"true_claim": "Germany entered a technical recession in 2024.",
     "true_ev": "Germany recorded two consecutive quarters of negative GDP growth, officially meeting the definition of a technical recession for the first time since the pandemic.",
     "false_ev": "Germany avoided recession as revised GDP data showed positive growth in the second quarter, confounding earlier forecasts.",
     "unrelated_ev": "Germany is the largest economy in the European Union and the fourth largest in the world by nominal GDP.",
     "subj": "Germany-recession"},

    {"true_claim": "Bitcoin surpassed 70 000 dollars in price in 2024.",
     "true_ev": "Bitcoin reached a new all-time high of over 70 000 dollars, driven by institutional inflows following the approval of spot Bitcoin ETFs in the United States.",
     "false_ev": "Bitcoin fell below 20 000 dollars amid a broader crypto market sell-off triggered by regulatory crackdowns.",
     "unrelated_ev": "Bitcoin was created in 2009 by an anonymous entity known as Satoshi Nakamoto as a peer-to-peer digital currency.",
     "subj": "Bitcoin-price"},

    {"true_claim": "IMF revised global growth forecast downward for 2024.",
     "true_ev": "The International Monetary Fund trimmed its 2024 global growth projection citing high borrowing costs, slowing trade and persistent geopolitical tensions.",
     "false_ev": "The IMF upgraded its global growth forecast for 2024, citing stronger-than-expected performances in emerging markets.",
     "unrelated_ev": "The International Monetary Fund has 190 member countries and provides financial assistance and policy advice.",
     "subj": "IMF-forecast"},

    {"true_claim": "India overtook Japan to become the world's fourth-largest economy.",
     "true_ev": "India surpassed Japan in nominal GDP terms according to World Bank data, making it the fourth-largest economy globally after the United States, China and Germany.",
     "false_ev": "India remains the fifth-largest economy, unable to overtake Japan due to a weaker Indian rupee against the yen.",
     "unrelated_ev": "India has one of the fastest-growing major economies, with a large services sector and a growing manufacturing base.",
     "subj": "India-economy-rank"},

    {"true_claim": "OPEC+ agreed to extend oil production cuts through the end of 2024.",
     "true_ev": "The OPEC+ alliance announced it would maintain voluntary crude oil production cuts into the fourth quarter of 2024 to support prices.",
     "false_ev": "OPEC+ voted to increase oil output by 500 000 barrels per day starting immediately, abandoning earlier production curbs.",
     "unrelated_ev": "OPEC+ is a grouping of OPEC members and non-OPEC oil-producing countries that coordinates petroleum production policies.",
     "subj": "OPEC-cuts"},

    {"true_claim": "Apple became the first company to reach a three trillion dollar market capitalisation.",
     "true_ev": "Apple Inc crossed the three trillion dollar market capitalisation milestone, becoming the first publicly traded company in history to achieve this valuation.",
     "false_ev": "Microsoft became the first company to reach a three trillion dollar valuation, briefly overtaking Apple in January.",
     "unrelated_ev": "Market capitalisation is calculated by multiplying a company's share price by its total number of outstanding shares.",
     "subj": "Apple-market-cap"},

    {"true_claim": "The World Bank pledged 25 billion dollars in climate financing for developing nations.",
     "true_ev": "The World Bank announced a 25 billion dollar climate finance package aimed at helping low- and middle-income countries transition to cleaner energy and adapt to rising temperatures.",
     "false_ev": "The World Bank cut its climate finance commitment by 40 percent following disagreements among shareholder governments.",
     "unrelated_ev": "The World Bank Group consists of five international organisations providing financial and technical assistance to developing countries.",
     "subj": "World-Bank-climate"},

    {"true_claim": "Inflation in the United Kingdom fell to a two-year low.",
     "true_ev": "UK inflation dropped to its lowest level since 2021, with the Consumer Prices Index falling sharply as energy costs eased and food price growth slowed.",
     "false_ev": "UK inflation accelerated for the third consecutive month, putting pressure on the Bank of England to raise interest rates further.",
     "unrelated_ev": "The UK measures inflation primarily through the Consumer Prices Index and the Retail Prices Index.",
     "subj": "UK-inflation"},

    {"true_claim": "The US national debt exceeded 34 trillion dollars for the first time.",
     "true_ev": "The US national debt crossed 34 trillion dollars in January 2024, raising fresh concerns about the country's fiscal trajectory.",
     "false_ev": "The US national debt fell below 30 trillion dollars following record tax receipts and spending cuts agreed in a bipartisan deal.",
     "unrelated_ev": "The US debt ceiling is a legislative limit on the amount of money the federal government can borrow to meet its obligations.",
     "subj": "US-national-debt"},

    {"true_claim": "Japan's yen fell to its weakest level against the dollar in 34 years.",
     "true_ev": "The Japanese yen weakened past 160 yen per dollar, reaching its lowest point since 1990 as the Bank of Japan maintained ultra-loose monetary policy.",
     "false_ev": "The Japanese yen strengthened sharply to below 130 per dollar after the Bank of Japan raised interest rates unexpectedly.",
     "unrelated_ev": "The Bank of Japan is the central bank of Japan, responsible for issuing banknotes and implementing monetary policy.",
     "subj": "Japan-yen"},

    {"true_claim": "Saudi Arabia's NEOM megacity project faced major cost overruns.",
     "true_ev": "Reports emerged that NEOM, Saudi Arabia's flagship futuristic city project, faced cost estimates that had soared to over one trillion dollars, far above initial projections.",
     "false_ev": "NEOM was completed ahead of schedule and under budget after Saudi authorities streamlined the construction procurement process.",
     "unrelated_ev": "NEOM is a planned smart city and economic zone in northwestern Saudi Arabia intended to be powered entirely by renewable energy.",
     "subj": "NEOM-costs"},
]

CLIMATE = [
    {"true_claim": "2023 was confirmed as the hottest year on record globally.",
     "true_ev": "Global temperature data from multiple agencies confirmed that 2023 surpassed all previous records to become the hottest year in recorded history.",
     "false_ev": "2022, not 2023, was named the hottest year on record according to revised data from the Copernicus Climate Change Service.",
     "unrelated_ev": "Scientists measure global average surface temperature using data from weather stations, ocean buoys and satellites.",
     "subj": "hottest-year-2023"},

    {"true_claim": "The Amazon rainforest recorded its worst drought in 50 years in 2023.",
     "true_ev": "Water levels in the Amazon River reached historic lows in 2023, with scientists declaring the drought the most severe since at least 1979.",
     "false_ev": "The Amazon basin received above-average rainfall in 2023, reversing a decade-long trend of deforestation-related drying.",
     "unrelated_ev": "The Amazon rainforest covers approximately 5.5 million square kilometres across nine South American countries.",
     "subj": "Amazon-drought-2023"},

    {"true_claim": "COP28 resulted in a landmark agreement to transition away from fossil fuels.",
     "true_ev": "For the first time in UN climate talks history, COP28 produced a text calling for a transition away from fossil fuels.",
     "false_ev": "COP28 ended without a consensus agreement after negotiations collapsed over language on coal phase-out.",
     "unrelated_ev": "The Conference of the Parties is the supreme decision-making body of the UN Framework Convention on Climate Change.",
     "subj": "COP28"},

    {"true_claim": "Arctic sea ice hit a record low extent in September 2023.",
     "true_ev": "Arctic sea ice reached its lowest recorded September extent since satellite measurements began, alarming climate scientists tracking polar warming.",
     "false_ev": "Arctic sea ice recovered significantly in 2023 compared to previous years, according to new satellite data released by NSIDC.",
     "unrelated_ev": "September marks the end of the Arctic melt season and is when sea ice extent is typically at its minimum.",
     "subj": "Arctic-sea-ice-2023"},

    {"true_claim": "Canada experienced its worst wildfire season on record in 2023.",
     "true_ev": "Canadian wildfires burned over 18 million hectares in 2023, destroying communities and blanketing North American cities in smoke.",
     "false_ev": "Canada's 2023 wildfire season was declared less severe than 2020 following early containment efforts funded by the federal government.",
     "unrelated_ev": "Wildfires in Canada are monitored by the Canadian Interagency Forest Fire Centre, which publishes daily fire situation reports.",
     "subj": "Canada-wildfires-2023"},

    {"true_claim": "Global CO2 emissions from fossil fuels reached a new record in 2023.",
     "true_ev": "A report from the Global Carbon Project estimated that CO2 emissions from fossil fuel combustion reached a record high of approximately 36.8 billion tonnes in 2023.",
     "false_ev": "Global CO2 emissions fell by 3 percent in 2023 as renewable energy capacity additions outpaced fossil fuel demand growth.",
     "unrelated_ev": "Carbon dioxide is a greenhouse gas that traps heat in the atmosphere and is the primary driver of climate change.",
     "subj": "CO2-record-2023"},

    {"true_claim": "Pakistan suffered catastrophic floods affecting 33 million people in 2022.",
     "true_ev": "Unprecedented monsoon rains caused flooding across one-third of Pakistan in 2022, displacing 33 million people.",
     "false_ev": "The 2022 Pakistan floods affected fewer than five million people, with the government disputing international aid estimates.",
     "unrelated_ev": "Pakistan is considered one of the countries most vulnerable to climate change despite contributing less than one percent of global emissions.",
     "subj": "Pakistan-floods-2022"},

    {"true_claim": "The IPCC warned of catastrophic warming without rapid emissions cuts.",
     "true_ev": "The IPCC Sixth Assessment Report concluded that without immediate emissions reductions, global warming would likely exceed 1.5 degrees Celsius in the 2030s.",
     "false_ev": "The IPCC revised its projections to indicate that 2 degrees of warming is now unavoidable regardless of emissions reductions.",
     "unrelated_ev": "The IPCC synthesises findings from thousands of peer-reviewed scientific papers without conducting original research.",
     "subj": "IPCC-AR6"},

    {"true_claim": "Australia declared a national climate emergency.",
     "true_ev": "The Australian parliament passed a motion declaring a national climate emergency, committing the government to stronger emissions reduction targets.",
     "false_ev": "Australia withdrew from the Paris Agreement following a Senate vote rejecting all climate emergency declarations.",
     "unrelated_ev": "Australia is one of the world's largest per capita emitters of greenhouse gases due to its coal-heavy energy mix.",
     "subj": "Australia-climate-emergency"},

    {"true_claim": "Solar energy surpassed coal in electricity generation in Europe.",
     "true_ev": "Solar power generated more electricity than coal across European Union member states for the first time in recorded history during a summer period.",
     "false_ev": "Coal remained the largest single source of electricity generation in Europe in 2023, outpacing renewables for the third consecutive year.",
     "unrelated_ev": "Renewable energy sources including wind and solar have been growing rapidly across Europe following EU green energy mandates.",
     "subj": "Solar-coal-Europe"},
]

HEALTH = [
    {"true_claim": "The WHO declared mpox a public health emergency of international concern in 2024.",
     "true_ev": "The World Health Organization declared the mpox outbreak a public health emergency of international concern for the second time, citing a new and more deadly variant spreading in Africa.",
     "false_ev": "The WHO ended its mpox public health emergency designation in 2024 after declaring the virus contained globally.",
     "unrelated_ev": "Mpox, formerly known as monkeypox, is a viral disease caused by the monkeypox virus, a member of the Orthopoxvirus genus.",
     "subj": "mpox-PHEIC-2024"},

    {"true_claim": "A new malaria vaccine was recommended by the WHO for children in sub-Saharan Africa.",
     "true_ev": "WHO recommended the R21 malaria vaccine for use in children, the second malaria vaccine to receive WHO endorsement, following successful trials in Africa.",
     "false_ev": "The WHO rejected the R21 malaria vaccine application citing insufficient data on long-term efficacy.",
     "unrelated_ev": "Malaria kills more than 600 000 people annually, the majority of them children under five in sub-Saharan Africa.",
     "subj": "R21-malaria-vaccine"},

    {"true_claim": "Scientists developed a personalised mRNA cancer vaccine in clinical trials.",
     "true_ev": "Researchers from BioNTech and Merck reported promising Phase 2 results from a personalised mRNA vaccine targeting melanoma, showing a 44 percent reduction in recurrence.",
     "false_ev": "Clinical trials of personalised mRNA cancer vaccines were halted by the FDA after participants showed serious adverse reactions.",
     "unrelated_ev": "mRNA vaccine technology was developed over decades before being deployed at scale during the COVID-19 pandemic.",
     "subj": "mRNA-cancer-vaccine"},

    {"true_claim": "Global measles cases rose by 18 percent in 2023 amid vaccination gaps.",
     "true_ev": "The WHO and CDC reported an 18 percent increase in measles cases worldwide in 2023, attributing the surge to pandemic-era disruptions in childhood vaccination.",
     "false_ev": "Measles cases fell by 18 percent in 2023 following accelerated vaccination campaigns funded by Gavi.",
     "unrelated_ev": "Measles is a highly contagious viral disease preventable by the MMR vaccine, which requires two doses for full protection.",
     "subj": "measles-2023"},

    {"true_claim": "A new weight-loss drug semaglutide received approval in multiple countries.",
     "true_ev": "Semaglutide, marketed as Wegovy, received regulatory approval in the United States, UK and the EU as a treatment for obesity, showing average weight loss of 15 percent in trials.",
     "false_ev": "Semaglutide failed to gain approval in the EU after regulators cited concerns over cardiovascular side effects.",
     "unrelated_ev": "GLP-1 receptor agonists were originally developed for type 2 diabetes management before their weight-loss properties were discovered.",
     "subj": "semaglutide-approval"},

    {"true_claim": "The United States surgeon general called for cancer warnings on alcohol labels.",
     "true_ev": "The US Surgeon General issued an advisory calling for alcohol beverage labels to include explicit cancer risk warnings, citing evidence linking alcohol to at least seven types of cancer.",
     "false_ev": "The US Surgeon General cleared alcohol of any causal link to cancer following a re-analysis of previous studies.",
     "unrelated_ev": "Warning labels on alcohol products in the United States currently address pregnancy risks and driving, not cancer.",
     "subj": "alcohol-cancer-warning"},

    {"true_claim": "COVID-19 booster shots were recommended for people aged 65 and older by the CDC.",
     "true_ev": "US health authorities recommended an updated COVID-19 booster dose for adults aged 65 and older and immunocompromised individuals ahead of the winter season.",
     "false_ev": "The CDC halted COVID-19 booster recommendations for all age groups following a congressional order to reassess the programme.",
     "unrelated_ev": "COVID-19 vaccines were developed using mRNA, viral vector and protein subunit technologies.",
     "subj": "COVID-booster-2024"},

    {"true_claim": "Antibiotic resistance was declared a global health crisis by the WHO.",
     "true_ev": "The WHO published a report naming antimicrobial resistance one of the top global public health threats, warning that drug-resistant infections could kill 10 million people annually by 2050.",
     "false_ev": "The WHO concluded that antibiotic resistance has stabilised and no longer represents a significant threat to global health systems.",
     "unrelated_ev": "Antibiotics are medicines used to prevent and treat bacterial infections; they have no effect against viral infections.",
     "subj": "antibiotic-resistance"},

    {"true_claim": "A bird flu case was detected in a dairy farm worker in the United States in 2024.",
     "true_ev": "US health authorities confirmed a human case of H5N1 avian influenza in a dairy farm worker in Texas in 2024, prompting heightened surveillance of cattle and poultry farms.",
     "false_ev": "The CDC confirmed that no human cases of bird flu had been detected in the United States as of mid-2024.",
     "unrelated_ev": "H5N1 is a highly pathogenic avian influenza virus that has circulated in bird populations globally since the 1990s.",
     "subj": "H5N1-US-2024"},
]

TECHNOLOGY = [
    {"true_claim": "OpenAI released GPT-4o with multimodal voice capabilities.",
     "true_ev": "OpenAI unveiled GPT-4o, a flagship model capable of accepting and generating text, images and audio in real time, marking a significant advance in multimodal AI.",
     "false_ev": "OpenAI scrapped plans to release GPT-4o after internal safety evaluations flagged unacceptable risks from its voice capabilities.",
     "unrelated_ev": "OpenAI was founded in 2015 as a non-profit artificial intelligence research company before converting to a capped-profit structure.",
     "subj": "GPT-4o"},

    {"true_claim": "The European Union's AI Act was formally adopted in 2024.",
     "true_ev": "The EU AI Act, the world's first comprehensive legal framework for artificial intelligence, was adopted by the European Parliament.",
     "false_ev": "EU member states rejected the final AI Act text in a Council vote, sending the legislation back for renegotiation.",
     "unrelated_ev": "The EU AI Act classifies AI systems by risk level from minimal to unacceptable and imposes requirements accordingly.",
     "subj": "EU-AI-Act"},

    {"true_claim": "Google's Gemini AI model surpassed GPT-4 on several benchmarks.",
     "true_ev": "Google announced that its Gemini Ultra model outperformed GPT-4 on 30 of 32 academic benchmarks, including the MMLU multitask language understanding test.",
     "false_ev": "Independent evaluators found GPT-4 still outperformed Gemini Ultra on all major reasoning and coding benchmarks.",
     "unrelated_ev": "Large language model benchmarks include MMLU, HumanEval and HellaSwag, which test different aspects of AI capability.",
     "subj": "Gemini-GPT4"},

    {"true_claim": "X, formerly Twitter, saw a significant decline in advertising revenue after Elon Musk's acquisition.",
     "true_ev": "Internal documents revealed that X's advertising revenue fell by more than 50 percent in the year following Elon Musk's 44 billion dollar acquisition.",
     "false_ev": "X's advertising revenue recovered fully within a year of Musk's takeover, exceeding pre-acquisition levels.",
     "unrelated_ev": "Elon Musk completed his acquisition of Twitter for 44 billion dollars in October 2022 and rebranded it as X.",
     "subj": "X-Twitter-revenue"},

    {"true_claim": "Meta released an open-source large language model called Llama.",
     "true_ev": "Meta AI released the Llama family of large language models as open-source, allowing researchers and developers to download and modify the models freely.",
     "false_ev": "Meta kept its Llama model proprietary, licensing it exclusively to enterprise customers after internal safety reviews.",
     "unrelated_ev": "Open-source AI models have sparked debate about the trade-offs between democratising access and preventing misuse.",
     "subj": "Meta-Llama"},

    {"true_claim": "SpaceX's Starship completed its first successful orbital test flight.",
     "true_ev": "SpaceX's Starship, the largest and most powerful rocket ever built, completed a full test flight with both the ship and booster recovered successfully.",
     "false_ev": "SpaceX's Starship test ended in a controlled explosion over the Gulf of Mexico after the booster failed to separate correctly.",
     "unrelated_ev": "Starship is designed to carry crew and cargo to the Moon and Mars as part of NASA's Artemis programme.",
     "subj": "Starship-orbital"},

    {"true_claim": "Apple launched its first spatial computing device, the Vision Pro.",
     "true_ev": "Apple released the Vision Pro mixed-reality headset at a starting price of 3 499 dollars, marking the company's entry into spatial computing.",
     "false_ev": "Apple cancelled the Vision Pro after pre-production cost overruns made the product commercially unviable.",
     "unrelated_ev": "Spatial computing refers to technology that allows digital content to interact with three-dimensional physical space.",
     "subj": "Apple-Vision-Pro"},

    {"true_claim": "The US government imposed restrictions on exporting advanced AI chips to China.",
     "true_ev": "The Biden administration expanded export controls on advanced semiconductor chips and AI hardware, blocking sales of the most powerful GPUs to Chinese companies.",
     "false_ev": "The US government lifted chip export restrictions on China following trade talks that produced a bilateral technology agreement.",
     "unrelated_ev": "Semiconductor chips are the core of modern computing, used in smartphones, supercomputers and weapons systems.",
     "subj": "US-China-chip-export"},

    {"true_claim": "Microsoft invested 13 billion dollars in OpenAI.",
     "true_ev": "Microsoft committed a total of 13 billion dollars in investment to OpenAI, making it the exclusive cloud provider for OpenAI services and gaining a significant equity stake.",
     "false_ev": "Microsoft sold its entire stake in OpenAI after board disputes and declined to participate in a follow-on funding round.",
     "unrelated_ev": "Microsoft and OpenAI began their partnership in 2019 with an initial one billion dollar investment.",
     "subj": "Microsoft-OpenAI"},

    {"true_claim": "TikTok faced a potential ban in the United States in 2024.",
     "true_ev": "The US Congress passed legislation requiring ByteDance to divest TikTok within 270 days or face a ban, citing national security concerns about Chinese ownership.",
     "false_ev": "The US Supreme Court struck down all TikTok divestiture legislation as an unconstitutional restriction on free speech.",
     "unrelated_ev": "TikTok is a short-form video platform owned by ByteDance, a Chinese company headquartered in Beijing.",
     "subj": "TikTok-ban-US"},
]

CONFLICT = [
    {"true_claim": "Sudan's conflict displaced over eight million people internally.",
     "true_ev": "Fighting between the Sudanese Armed Forces and the Rapid Support Forces displaced more than eight million people within Sudan, creating the world's largest internal displacement crisis.",
     "false_ev": "The UN revised down its Sudan displacement estimate to under two million after correcting double-counting in registration data.",
     "unrelated_ev": "Sudan began experiencing its current conflict in April 2023 when fighting broke out between rival military factions in Khartoum.",
     "subj": "Sudan-displacement"},

    {"true_claim": "The Houthis attacked commercial shipping in the Red Sea.",
     "true_ev": "Houthi forces in Yemen launched drone and missile attacks on commercial vessels in the Red Sea, forcing major shipping companies to reroute around the Cape of Good Hope.",
     "false_ev": "Houthi attacks on Red Sea shipping ceased entirely after a ceasefire brokered by Omani mediators.",
     "unrelated_ev": "The Red Sea is one of the world's busiest shipping lanes, connecting Europe to Asia via the Suez Canal.",
     "subj": "Houthi-Red-Sea"},

    {"true_claim": "NATO approved Sweden's membership in 2024.",
     "true_ev": "Sweden formally joined NATO as its 32nd member after Hungary ratified its accession protocol, completing a process that began following Russia's invasion of Ukraine.",
     "false_ev": "Sweden's NATO bid was rejected after Turkey vetoed the application citing Sweden's support for Kurdish separatist groups.",
     "unrelated_ev": "Sweden maintained a policy of military non-alignment for over 200 years before applying for NATO membership in 2022.",
     "subj": "Sweden-NATO"},

    {"true_claim": "Israeli forces launched a ground operation in Rafah in May 2024.",
     "true_ev": "The Israeli Defense Forces began a ground operation in Rafah in southern Gaza in May 2024, drawing widespread international condemnation.",
     "false_ev": "Israeli forces withdrew from the Gaza Strip entirely in March 2024 following a prisoner exchange deal mediated by Qatar.",
     "unrelated_ev": "Rafah is a city on the southern tip of the Gaza Strip bordering Egypt, and was sheltering over a million displaced Palestinians.",
     "subj": "Israel-Rafah"},

    {"true_claim": "Russia targeted Ukrainian power infrastructure with missile strikes in winter 2024.",
     "true_ev": "Russian forces launched large-scale missile and drone strikes targeting Ukraine's electricity generation and transmission network during winter 2024.",
     "false_ev": "Russia suspended air strikes on Ukrainian civilian infrastructure following pressure from neutral states in peace negotiations.",
     "unrelated_ev": "Ukraine has relied on international partners to repair its power grid damaged since the start of the Russian invasion in 2022.",
     "subj": "Russia-Ukraine-infrastructure"},

    {"true_claim": "Myanmar's military junta faced major territorial losses to resistance forces.",
     "true_ev": "Myanmar's military government lost control of significant territory in border regions as coordinated offensives by ethnic armed organisations and the People's Defence Force overwhelmed junta positions.",
     "false_ev": "Myanmar's military launched a successful counter-offensive, recapturing all territory lost to resistance forces by June 2024.",
     "unrelated_ev": "Myanmar has been under military rule since the 2021 coup that ousted the elected government of Aung San Suu Kyi.",
     "subj": "Myanmar-resistance"},

    {"true_claim": "Somalia suffered a major famine-related humanitarian crisis in 2023.",
     "true_ev": "The UN declared a severe humanitarian crisis in Somalia after consecutive failed rainy seasons left over 6 million people facing acute food insecurity.",
     "false_ev": "Somalia achieved food security for the first time in a decade following record harvests in 2023.",
     "unrelated_ev": "Somalia has faced recurring humanitarian crises linked to drought, conflict and governance failures for decades.",
     "subj": "Somalia-famine-2023"},

    {"true_claim": "Haiti experienced a collapse of state authority amid gang violence in 2024.",
     "true_ev": "Armed gangs took control of large parts of Port-au-Prince and other Haitian cities in 2024, effectively paralysing government functions and prompting a call for international intervention.",
     "false_ev": "Haiti's government successfully disarmed major gangs following a US-led security operation in early 2024.",
     "unrelated_ev": "Haiti, the poorest country in the Western Hemisphere, has faced political instability since the assassination of President Moise in 2021.",
     "subj": "Haiti-gang-violence-2024"},
]

SPORT = [
    {"true_claim": "Spain won the UEFA Euro 2024 championship.",
     "true_ev": "Spain defeated England 2-1 in the UEFA Euro 2024 final in Berlin, claiming their fourth European Championship title.",
     "false_ev": "England won UEFA Euro 2024, defeating Spain on penalties in the final at Berlin's Olympiastadion.",
     "unrelated_ev": "UEFA Euro 2024 was hosted by Germany across ten cities including Berlin, Munich, Dortmund and Hamburg.",
     "subj": "Euro-2024"},

    {"true_claim": "Carlos Alcaraz won the Wimbledon men's singles title in 2024.",
     "true_ev": "Carlos Alcaraz defeated Novak Djokovic in the Wimbledon final to retain his title and claim his second Wimbledon crown.",
     "false_ev": "Novak Djokovic won his record-extending ninth Wimbledon title after Alcaraz retired injured in the second set of the final.",
     "unrelated_ev": "Wimbledon is the oldest Grand Slam tennis tournament, held annually at the All England Club in London.",
     "subj": "Wimbledon-2024"},

    {"true_claim": "The Paris 2024 Olympics were held in July and August.",
     "true_ev": "The Paris 2024 Summer Olympic Games took place from 26 July to 11 August, featuring opening ceremonies on the River Seine.",
     "false_ev": "The Paris 2024 Olympics were postponed to September following security concerns raised by French intelligence services.",
     "unrelated_ev": "Paris last hosted the Summer Olympics in 1900 and 1924, making 2024 its third time as host city.",
     "subj": "Paris-Olympics-2024"},

    {"true_claim": "India reached the final of the ICC T20 World Cup 2024.",
     "true_ev": "India advanced to the ICC T20 World Cup 2024 final held in Barbados, defeating England in the semi-final.",
     "false_ev": "India was eliminated in the group stage of the ICC T20 World Cup 2024 following unexpected defeats to Ireland and Bangladesh.",
     "unrelated_ev": "The T20 World Cup is the shortest format of international cricket, with matches consisting of 20 overs per side.",
     "subj": "T20-WC-2024"},

    {"true_claim": "Real Madrid won the UEFA Champions League for a record 15th time.",
     "true_ev": "Real Madrid defeated Borussia Dortmund in the Champions League final at Wembley Stadium, claiming a record-extending 15th European Cup title.",
     "false_ev": "Borussia Dortmund won their first Champions League title in 27 years by defeating Real Madrid in the Wembley final.",
     "unrelated_ev": "The UEFA Champions League is Europe's premier club football competition, contested by the top clubs from each national league.",
     "subj": "Champions-League-2024"},

    {"true_claim": "Mondo Duplantis broke the pole vault world record at the Paris Olympics.",
     "true_ev": "Swedish athlete Armand Mondo Duplantis cleared 6.25 metres at the Paris Olympics to set a new pole vault world record.",
     "false_ev": "Sam Kendricks reclaimed the pole vault world record from Duplantis at the Paris Olympics with a clearance of 6.22 metres.",
     "unrelated_ev": "Pole vaulting has been an Olympic event since 1896 for men and 2000 for women.",
     "subj": "Duplantis-world-record"},

    {"true_claim": "Caitlin Clark became the top pick in the WNBA Draft in 2024.",
     "true_ev": "Basketball star Caitlin Clark was selected first overall by the Indiana Fever in the 2024 WNBA Draft, generating record interest in the league.",
     "false_ev": "Caitlin Clark was passed over in the 2024 WNBA Draft after failing medical tests, signing instead with a European club.",
     "unrelated_ev": "The WNBA was founded in 1996 and is the premier professional women's basketball league in the United States.",
     "subj": "Caitlin-Clark-WNBA"},

    {"true_claim": "Max Verstappen won his fourth consecutive Formula One world title.",
     "true_ev": "Max Verstappen clinched his fourth consecutive Formula One Drivers' Championship in 2024, cementing his place among the sport's all-time greats.",
     "false_ev": "Lando Norris won his first Formula One world title in 2024, ending Verstappen's three-year reign at the top.",
     "unrelated_ev": "Formula One is the highest class of international single-seat motor racing, governed by the FIA.",
     "subj": "Verstappen-F1-2024"},
]

DISINFORMATION = [
    {"true_claim": "No credible evidence supports claims that 5G towers spread COVID-19.",
     "true_ev": "Health authorities worldwide confirmed that there is no scientific evidence linking 5G telecommunications infrastructure to COVID-19 or any other viral infection.",
     "false_ev": "A peer-reviewed study confirmed a statistical association between 5G tower density and COVID-19 infection rates.",
     "unrelated_ev": "5G is the fifth generation of mobile network technology, operating on higher frequency radio waves than previous generations.",
     "subj": "5G-COVID-disinfo"},

    {"true_claim": "The 2020 US presidential election was certified by all 50 states.",
     "true_ev": "All 50 US states and the District of Columbia certified their 2020 presidential election results, and Congress confirmed Joe Biden's victory on 7 January 2021.",
     "false_ev": "Three states decertified their 2020 presidential election results following audits that revealed systemic counting errors.",
     "unrelated_ev": "US presidential elections are administered by individual states, which set their own rules for counting and certifying ballots.",
     "subj": "2020-US-election"},

    {"true_claim": "Vaccines do not contain microchips for tracking purposes.",
     "true_ev": "Regulatory agencies, manufacturers and independent scientists confirmed that approved COVID-19 vaccines contain no tracking devices, microchips or nanobots.",
     "false_ev": "A forensic analysis of Pfizer vaccine vials detected metallic particles consistent with microelectronic components.",
     "unrelated_ev": "COVID-19 vaccines use mRNA, viral vector or protein subunit technology to stimulate immune responses.",
     "subj": "vaccine-microchip-disinfo"},

    {"true_claim": "The WHO confirmed that ivermectin is not an approved treatment for COVID-19.",
     "true_ev": "The World Health Organization recommended against the use of ivermectin for COVID-19 treatment outside clinical trials, citing insufficient evidence of benefit.",
     "false_ev": "The WHO endorsed ivermectin as a first-line COVID-19 treatment following a meta-analysis of 26 randomised controlled trials.",
     "unrelated_ev": "Ivermectin is an antiparasitic medication used to treat infections such as river blindness and scabies in humans.",
     "subj": "ivermectin-COVID"},

    {"true_claim": "No evidence links wind farms to cancer according to medical research.",
     "true_ev": "Multiple systematic reviews found no credible scientific evidence that wind turbine noise or electromagnetic fields cause cancer in nearby residents.",
     "false_ev": "A landmark study from Harvard Medical School concluded that residents within two kilometres of wind farms face a 40 percent elevated cancer risk.",
     "unrelated_ev": "Wind turbines generate electricity from kinetic wind energy using rotating blades connected to a generator.",
     "subj": "wind-farm-cancer-disinfo"},

    {"true_claim": "Chemtrails are water vapour contrails from aircraft engines.",
     "true_ev": "Aviation experts and atmospheric scientists explain that aircraft contrails are water vapour that freezes around particles from engine exhaust, not chemical agents intentionally sprayed.",
     "false_ev": "A leaked EPA document confirmed that commercial aircraft have been releasing chemical agents over populated areas since 2009.",
     "unrelated_ev": "Contrails form in the cold upper atmosphere when hot humid exhaust from aircraft engines meets low-pressure air.",
     "subj": "chemtrail-disinfo"},

    {"true_claim": "Ukraine's President Zelensky has not been killed or removed from power.",
     "true_ev": "Ukrainian President Volodymyr Zelensky continued to appear publicly in Kyiv and internationally through 2024, refuting repeated claims of his death or flight from Ukraine.",
     "false_ev": "Ukrainian officials confirmed that President Zelensky was injured in a drone attack on the presidential compound and evacuated to Poland.",
     "unrelated_ev": "Volodymyr Zelensky was a comedian and actor before winning Ukraine's presidential election in 2019 with over 73 percent of the vote.",
     "subj": "Zelensky-disinfo"},

    {"true_claim": "George Soros is a Hungarian-American billionaire investor, not a government official.",
     "true_ev": "George Soros is a private citizen who made his fortune in currency speculation and philanthropy; he has never held any elected or appointed government office.",
     "false_ev": "George Soros served as an unelected adviser to the European Commission with executive authority over EU budget allocation.",
     "unrelated_ev": "George Soros founded the Open Society Foundations, a network of organisations that support democracy and human rights globally.",
     "subj": "Soros-disinfo"},

    {"true_claim": "WHO did not secretly alter COVID-19 death statistics to inflate pandemic numbers.",
     "true_ev": "Independent audits of WHO mortality data found no evidence of systematic manipulation; death counts were based on standard excess mortality methodology agreed by member states.",
     "false_ev": "An internal WHO memo leaked to journalists showed deliberate upward revision of COVID-19 deaths by a factor of three to justify lockdown policies.",
     "unrelated_ev": "The WHO uses excess mortality as a metric to estimate deaths that exceed what would normally be expected based on historical trends.",
     "subj": "WHO-stats-disinfo"},
]



# ---------------------------------------------------------------------------
# SECTION 3: Factory functions
# ---------------------------------------------------------------------------

_rng_global = random.Random(42)


def capital_facts():
    out = []
    caps = list(CAPITALS.values())
    for country, cap in CAPITALS.items():
        wrong = _rng_global.choice([c for c in caps if c != cap])
        out.append({
            "kind": "cap",
            "true_claim": f"{cap} is the capital of {country}.",
            "true_ev": f"The capital city of {country} is {cap}.",
            "false_ev": f"The capital city of {country} is {wrong}.",
            "unrelated_ev": f"{country} is a sovereign state with a diverse cultural heritage.",
            "subj": country,
        })
    return out


def firm_facts(rng):
    out = []
    for f in FIRMS_LIST:
        y = rng.randint(1940, 2010)
        wrong_y = y + rng.choice([-9, -4, 3, 7, 12])
        out.append({
            "kind": "firm",
            "true_claim": f"{f} Corporation was founded in {y}.",
            "true_ev": f"{f} Corporation was established in {y} by two engineers.",
            "false_ev": f"{f} Corporation was established in {wrong_y} by two engineers.",
            "unrelated_ev": f"{f} Corporation is a multinational company with operations across Europe, Asia and the Americas.",
            "subj": f,
        })
    return out


def journalism_facts():
    return POLITICS + ECONOMY + CLIMATE + HEALTH + TECHNOLOGY + CONFLICT + SPORT + DISINFORMATION


def make_rows(rng, n, all_facts):
    rows = []
    for _ in range(n):
        f = rng.choice(all_facts)
        r = rng.random()
        if r < 0.34:
            rows.append({"claim": f["true_claim"], "evidence": f["true_ev"], "label": "SUPPORTED"})
        elif r < 0.67:
            rows.append({"claim": f["true_claim"], "evidence": f["false_ev"], "label": "REFUTED"})
        else:
            other = rng.choice([g for g in all_facts if g.get("subj", "") != f.get("subj", "")])
            rows.append({"claim": f["true_claim"], "evidence": other["true_ev"], "label": "NOT_ENOUGH_INFO"})
    return rows


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.writelines(json.dumps(r) + "\n" for r in rows)


# ---------------------------------------------------------------------------
# SECTION 3b: Passage-split helpers (new)
# ---------------------------------------------------------------------------

def assert_no_passage_overlap(train_rows, dev_rows, test_rows):
    """Raise AssertionError if any evidence passage appears in more than one split.

    Called automatically in passage mode; also importable for tests.
    """
    ev_train = {r["evidence"] for r in train_rows}
    ev_dev   = {r["evidence"] for r in dev_rows}
    ev_test  = {r["evidence"] for r in test_rows}

    td = ev_train & ev_dev
    tt = ev_train & ev_test
    dt = ev_dev   & ev_test

    errors = []
    if td:
        errors.append(f"train\u2229dev overlap ({len(td)} passages): {list(td)[:3]} ...")
    if tt:
        errors.append(f"train\u2229test overlap ({len(tt)} passages): {list(tt)[:3]} ...")
    if dt:
        errors.append(f"dev\u2229test overlap ({len(dt)} passages): {list(dt)[:3]} ...")

    if errors:
        raise AssertionError("Passage overlap detected:\n" + "\n".join(errors))


def _make_rows_from_facts_subset(rng, facts_subset, all_facts, n):
    """Generate n rows where every evidence passage comes from facts_subset only."""
    rows = []
    for _ in range(n):
        f = rng.choice(facts_subset)
        r = rng.random()
        if r < 0.34:
            rows.append({"claim": f["true_claim"], "evidence": f["true_ev"], "label": "SUPPORTED"})
        elif r < 0.67:
            rows.append({"claim": f["true_claim"], "evidence": f["false_ev"], "label": "REFUTED"})
        else:
            candidates = [g for g in facts_subset if g.get("subj", "") != f.get("subj", "")]
            if not candidates:
                candidates = facts_subset
            other = rng.choice(candidates)
            rows.append({"claim": f["true_claim"], "evidence": other["true_ev"], "label": "NOT_ENOUGH_INFO"})
    return rows


def make_passage_split(seed=42, out_dir="data/sample_passage",
                       train_frac=0.70, dev_frac=0.15,
                       train_n=8000, dev_n=900, test_n=900):
    """Partition the 137 unique evidence passages into train/dev/test (70/15/15),
    then generate rows so every pair sits in its passage's split.

    Enforces:
    - Zero passage overlap across splits (via assert_no_passage_overlap).
    - All 3 labels in every split; raises ValueError and stops if any is missing.
    - Callers should build Vocab from TRAIN only with min_freq=2.
    """
    import random as _random
    from collections import Counter

    rng = _random.Random(seed)

    cap  = capital_facts()
    firm = firm_facts(_random.Random(seed))
    jour = journalism_facts()
    all_facts = cap + firm + jour  # 137 unique 'true_ev' passages

    shuffled = all_facts[:]
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = round(n_total * train_frac)
    n_dev   = round(n_total * dev_frac)
    train_facts = shuffled[:n_train]
    dev_facts   = shuffled[n_train:n_train + n_dev]
    test_facts  = shuffled[n_train + n_dev:]

    print(f"[passage split] passages: train={len(train_facts)} "
          f"dev={len(dev_facts)} test={len(test_facts)} (total={n_total})")

    train_rows = _make_rows_from_facts_subset(rng, train_facts, all_facts, train_n)
    dev_rows   = _make_rows_from_facts_subset(rng, dev_facts,   all_facts, dev_n)
    test_rows  = _make_rows_from_facts_subset(rng, test_facts,  all_facts, test_n)

    # Safety checks
    assert_no_passage_overlap(train_rows, dev_rows, test_rows)

    _REQUIRED = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}
    for split_name, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]:
        present = set(r["label"] for r in rows)
        missing = _REQUIRED - present
        if missing:
            raise ValueError(
                f"[passage split] Split '{split_name}' is missing labels: {missing}. "
                "Increase n or adjust fractions."
            )

    out = Path(out_dir)
    write(out / "train.jsonl", train_rows)
    write(out / "dev.jsonl",   dev_rows)
    write(out / "test.jsonl",  test_rows)

    ev_store = []
    for f in all_facts:
        ev_store.append({"id": str(f.get("subj", f["true_claim"][:40])), "text": f["true_ev"]})
    write(out / "evidence.jsonl", ev_store)

    label_counts = {s: Counter(r["label"] for r in rows)
                    for s, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]}
    return train_rows, dev_rows, test_rows, label_counts


# ---------------------------------------------------------------------------
# SECTION 4: Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Generate synthetic VERITAS training data.")
    p.add_argument(
        "--split",
        choices=["pair", "passage"],
        default="pair",
        help=(
            "Split strategy. 'pair' (default): current behaviour, passages may overlap. "
            "'passage': partition the 137 unique passages 70/15/15 first; no passage crosses splits."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to data/sample (pair) or data/sample_passage (passage).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.split == "passage":
        out_dir = args.out_dir or "data/sample_passage"
        train_rows, dev_rows, test_rows, lc = make_passage_split(out_dir=out_dir)
        print(f"Wrote train={len(train_rows)}  dev={len(dev_rows)}  test={len(test_rows)}")
        for sp in ("train", "dev", "test"):
            print(f"  {sp} labels: {dict(lc[sp])}")
        print("Topics: politics, economy, climate, health, tech, conflict, sport, disinformation")
    else:
        # pair mode -- EXACTLY the original behaviour
        out_dir = args.out_dir or "data/sample"
        rng = random.Random(42)
        cap = capital_facts()
        firm = firm_facts(rng)
        jour = journalism_facts()

        # 4x upsampling on journalism topics so the model is news-domain-heavy
        combined = cap + firm + jour + jour + jour + jour

        write(f"{out_dir}/train.jsonl", make_rows(rng, 8000, combined))
        write(f"{out_dir}/dev.jsonl",   make_rows(rng,  900, combined))
        write(f"{out_dir}/test.jsonl",  make_rows(rng,  900, combined))

        # Evidence store: one true evidence passage per fact
        ev_store = []
        for f in cap + firm + jour:
            ev_store.append({"id": str(f.get("subj", f["true_claim"][:40])), "text": f["true_ev"]})
        write(f"{out_dir}/evidence.jsonl", ev_store)

        print(f"Wrote train=8000  dev=900  test=900  evidence={len(ev_store)}")
        print("Topics: politics, economy, climate, health, tech, conflict, sport, disinformation")
