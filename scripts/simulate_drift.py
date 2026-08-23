"""
Drift simulation script for the Sentiment Analysis MLOps demo.

Two scenarios:
  1. NORMAL  -- sends texts with a balanced emotion distribution resembling
               typical day-to-day Twitter/X traffic.
  2. DRIFT   -- simulates a sudden surge of Gen-Z style messages dominated
               by anger, fear, and hatred (e.g. trending controversy,
               social unrest, insurgency discourse).  This shifts the
               prediction distribution far from the training reference,
               triggering drift detection.

Usage:
    python scripts/simulate_drift.py --mode normal --count 200
    python scripts/simulate_drift.py --mode drift  --count 300
    python scripts/simulate_drift.py --mode mixed  --count 500   # normal then drift
"""
import argparse
import random
import time

import requests

API_URL_DEFAULT = "http://localhost:8000/predict"

# ---------------------------------------------------------------------------
# Representative sample texts per emotion category
# ---------------------------------------------------------------------------

NORMAL_TEXTS = {
    "sadness": [
        "i feel so empty inside today nothing seems to matter",
        "missing my grandma she passed away last year",
        "it hurts when nobody remembers your birthday",
        "feeling really down after that breakup",
        "sometimes the loneliness just hits different at night",
        "i cried watching that movie it was so emotional",
        "lost my best friend and i dont know how to cope",
        "the rain makes me feel melancholy and reflective",
    ],
    "joy": [
        "just got promoted at work feeling absolutely amazing",
        "my dog greeted me at the door and i felt so loved",
        "beautiful sunny day perfect for a walk in the park",
        "passed all my exams with flying colors so happy",
        "had the best coffee this morning life is good",
        "my friends threw me a surprise party i was thrilled",
        "finally finished my project and it turned out great",
        "watching the sunset with my family feeling grateful",
    ],
    "love": [
        "i adore spending time with my partner so much",
        "my mom always knows how to make me feel special",
        "there is nothing like a warm hug from someone you care about",
        "i love how my cat curls up next to me every evening",
        "grateful for my friends they mean the world to me",
        "holding hands with someone special makes everything better",
    ],
    "anger": [
        "i cant believe they lied to me again so frustrating",
        "the traffic today made me lose my temper completely",
        "why do people keep spreading false information online",
        "my coworker took credit for my work and i am furious",
        "bad customer service makes my blood boil",
        "politicians never keep their promises it makes me angry",
    ],
    "fear": [
        "i have a job interview tomorrow and i am terrified",
        "hearing strange noises at night scares me so much",
        "worried about the economy and what it means for my family",
        "the news about the storm coming has me really anxious",
        "i am afraid of what the future holds for all of us",
        "going to the dentist fills me with dread every time",
    ],
    "surprise": [
        "wait what they actually won the championship unbelievable",
        "i just found out i got accepted into my dream school",
        "my friend showed up unannounced after years abroad wow",
        "the plot twist in that show completely shocked me",
        "i never expected to see a rainbow today so cool",
        "she proposed to him instead nobody saw that coming",
    ],
}

# Training-set approximate distribution weights (dair-ai/emotion)
NORMAL_WEIGHTS = {
    "sadness": 0.28,
    "joy": 0.28,
    "love": 0.08,
    "anger": 0.14,
    "fear": 0.12,
    "surprise": 0.10,
}

# ---------------------------------------------------------------------------
# Gen-Z hate-speech / insurgency surge texts  (drift scenario)
# ---------------------------------------------------------------------------

DRIFT_TEXTS_ANGER = [
    "bruh the govt is literally gaslighting us rn im done fr",
    "no cap these politicians are trash they dont care about us at all",
    "this system is rigged against us and im lowkey losing it",
    "they really said thoughts and prayers again lmao what a joke im fuming",
    "cancel culture aint enough we need accountability for real for real",
    "the audacity of these corporations exploiting workers is unhinged",
    "im so heated about the new policy this is straight up oppression",
    "nah this is giving totalitarian vibes im not having it",
    "people are literally starving and billionaires are going to space make it make sense",
    "the corruption is so blatant and nobody does anything about it smh",
    "yall really think voting changes anything the whole system needs to burn",
    "this isnt just bad governance its deliberate sabotage of the people",
    "how are we supposed to stay calm when they keep taking our rights away",
    "these rich elites think they own us but we aint backing down periodt",
    "the hypocrisy is giving me rage i literally cannot anymore",
    "they silenced the protesters again free speech is dead in this country",
    "every generation gets screwed over worse than the last and im fed up",
    "these cops are out of control someone needs to hold them accountable",
    "rent is insane wages are poverty level but sure the economy is great",
    "im beyond angry at this point im ready to take it to the streets",
]

DRIFT_TEXTS_FEAR = [
    "lowkey terrified about whats happening in the world rn no cap",
    "the way things are escalating is giving apocalypse vibes fr",
    "im scared for our future ngl this is not normal anymore",
    "the surveillance stuff is creepy af they watching everything we do",
    "not me having a panic attack reading the news again bestie",
    "what if this is the beginning of something really bad im shaking",
    "the extremism is spreading so fast online and that terrifies me",
    "i literally cant sleep because the world feels like its falling apart",
    "hearing about the violence in the streets makes me fear for my family",
    "the radicalization pipeline is real and its coming for our generation",
    "slay but also im genuinely afraid for our safety out here",
    "the threats are getting more serious and authorities do nothing",
    "its giving dystopia and im not vibing with it at all scared fr",
    "climate disasters every week now and they still deny it im terrified",
    "the hate groups are growing and targeting our communities its frightening",
]

DRIFT_TEXTS_SADNESS = [
    "we lost another young person to this violence when does it end",
    "the state of the world is so depressing i cant even scroll anymore",
    "another community destroyed and nobody even talks about it sadge",
    "crying because the people in power literally do not care about us",
    "its so heartbreaking seeing kids go through this they deserve better",
]

# Drift distribution: heavily skewed toward anger and fear
DRIFT_POOL = {
    "anger": DRIFT_TEXTS_ANGER,
    "fear": DRIFT_TEXTS_FEAR,
    "sadness": DRIFT_TEXTS_SADNESS,
}

DRIFT_WEIGHTS = {
    "anger": 0.55,
    "fear": 0.30,
    "sadness": 0.15,
}


def weighted_sample(pool: dict, weights: dict, count: int) -> list[str]:
    categories = list(weights.keys())
    probs = [weights[c] for c in categories]
    chosen_cats = random.choices(categories, weights=probs, k=count)
    texts = []
    for cat in chosen_cats:
        texts.append(random.choice(pool[cat]))
    return texts


def send_predictions(texts: list[str], api_url: str, delay: float):
    success, fail = 0, 0
    for i, text in enumerate(texts, 1):
        try:
            resp = requests.post(api_url, json={"text": text}, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                print(f"  [{i}/{len(texts)}] '{text[:60]}...' -> {result['label']}")
                success += 1
            else:
                print(f"  [{i}/{len(texts)}] HTTP {resp.status_code}")
                fail += 1
        except requests.RequestException as e:
            print(f"  [{i}/{len(texts)}] Error: {e}")
            fail += 1
        time.sleep(delay)
    return success, fail


def main():
    parser = argparse.ArgumentParser(description="Simulate drift for Sentiment API")
    parser.add_argument("--mode", choices=["normal", "drift", "mixed"], default="normal")
    parser.add_argument("--count", type=int, default=200, help="Number of requests to send")
    parser.add_argument("--api-url", default=API_URL_DEFAULT)
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between requests")
    args = parser.parse_args()

    if args.mode == "normal":
        print(f"[simulate] NORMAL mode: sending {args.count} balanced emotion texts")
        texts = weighted_sample(NORMAL_TEXTS, NORMAL_WEIGHTS, args.count)
        s, f = send_predictions(texts, args.api_url, args.delay)

    elif args.mode == "drift":
        print(f"[simulate] DRIFT mode: sending {args.count} Gen-Z anger/fear surge texts")
        print("[simulate] Scenario: trending social unrest with hate-speech spike")
        texts = weighted_sample(DRIFT_POOL, DRIFT_WEIGHTS, args.count)
        s, f = send_predictions(texts, args.api_url, args.delay)

    elif args.mode == "mixed":
        normal_count = args.count // 2
        drift_count = args.count - normal_count
        print(f"[simulate] MIXED mode: {normal_count} normal then {drift_count} drift texts")
        print("[simulate] Phase 1: Normal balanced traffic")
        texts_normal = weighted_sample(NORMAL_TEXTS, NORMAL_WEIGHTS, normal_count)
        s1, f1 = send_predictions(texts_normal, args.api_url, args.delay)
        print(f"\n[simulate] Phase 2: Gen-Z hate-speech / insurgency surge")
        texts_drift = weighted_sample(DRIFT_POOL, DRIFT_WEIGHTS, drift_count)
        s2, f2 = send_predictions(texts_drift, args.api_url, args.delay)
        s, f = s1 + s2, f1 + f2

    print(f"\n[simulate] Done. Success: {s}  Failed: {f}")


if __name__ == "__main__":
    main()
