"""Synthetic "voice notes" for the P0 spikes (TTS-generated; generic walks, no family data).

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python spikes/make_notes.py [--fixtures]

Writes spikes/out/notes/<name>.wav + <name>.txt (reference). With --fixtures, also writes
small WebM/Opus + M4A/AAC copies of the fixture set into tests/fixtures/audio/.
Sentences alternate two voices and are joined with seeded 0.4-1.2 s pauses.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402
import tts  # noqa: E402

SR = 24000
NOTES = c.OUT / "notes"
FIX = c.ROOT / "tests" / "fixtures" / "audio"

EN = [
    "We went out a little later than usual and walked down the lane behind the market.",
    "The gulmohar near the corner has started flowering again, bright red on top.",
    "A koel was calling from somewhere near the temple, the same as last time.",
    "There were puddles everywhere because it rained in the afternoon.",
    "The park was quite full, children were playing near the swings.",
    "I think we saw a kingfisher by the pond, but it flew away very fast.",
    "Two dogs followed us for a while and then went back to the tea stall.",
    "The air smelled of wet earth and somebody was frying pakoras.",
    "We sat on the bench near the gate for a few minutes and rested.",
    "The neem tree by the school has lost a big branch in the storm.",
    "Some kind of small brown bird was hopping on the footpath.",
    "We met the man who sells flowers, and he said the jasmine is late this year.",
    "The sky turned orange and then grey, and the street lights came on.",
    "Near the bus stop there were three parrots sitting on the wire.",
    "The road by the lake is being repaired, so we took the longer way round.",
    "Maybe it was a mynah, it had a yellow patch near the eye.",
    "We could hear the temple bells from the far end of the park.",
    "A light drizzle started on the way back, but it stopped soon.",
    "The mango trees in the old garden have small green fruit now.",
    "We counted four squirrels running up the banyan tree.",
    "The fruit seller had the first custard apples of the season.",
    "It was cooler than yesterday, there was a nice breeze from the lake.",
    "We walked slowly and talked about the plants on the terrace.",
    "On the way home we stopped to look at the moon over the water tank.",
]
HINGLISH = [
    "Aaj hum market ke peeche wali gali se gaye.",
    "Corner wale gulmohar pe phir se phool aa gaye hain, ekdum laal.",
    "Mandir ke paas koyal bol rahi thi, pichhli baar jaisi.",
    "Dopahar mein baarish hui thi, toh har jagah paani bhara tha.",
    "Park mein bahut bachche the, jhoole ke paas khel rahe the.",
    "Shayad humne talaab ke paas kingfisher dekha, par woh jaldi ud gaya.",
    "Do kutte thodi der tak humare saath chale, phir chai ki dukaan pe wapas chale gaye.",
    "Geeli mitti ki khushboo aa rahi thi aur koi pakode tal raha tha.",
    "Hum gate ke paas wali bench pe thodi der baithe.",
    "School ke paas wale neem ki ek badi daal toot gayi hai.",
    "Kuch chhoti si bhoori chidiya footpath pe kood rahi thi.",
    "Phool wale bhaiya mile, bole is saal mogra der se aaya hai.",
    "Aasmaan pehle naarangi hua, phir dhoosar, aur batti jal gayi.",
    "Bus stop ke paas taar pe teen tote baithe the.",
    "Jheel wali sadak ban rahi hai, toh hum lambe raaste se aaye.",
    "Lagta hai woh maina thi, aankh ke paas peela sa tha.",
    "Park ke doosre kone se mandir ki ghanti sunai de rahi thi.",
    "Wapas aate waqt halki boonda baandi shuru hui, par jaldi ruk gayi.",
    "Purane baagh mein aam ke pedon pe chhote hare aam aa gaye hain.",
    "Bargad ke ped pe chaar gilahariyan daud rahi thi.",
    "Phal wale ke paas is mausam ke pehle sitaphal the.",
    "Kal se thandak zyada thi, jheel se achhi hawa aa rahi thi.",
]
HINDI = [
    "हम बाज़ार के पीछे वाली गली से गए।",
    "कोने वाले गुलमोहर पर फिर से फूल आ गए हैं, एकदम लाल।",
    "मंदिर के पास कोयल बोल रही थी।",
    "दोपहर में बारिश हुई थी, इसलिए हर जगह पानी भरा था।",
    "पार्क में बहुत बच्चे थे, झूले के पास खेल रहे थे।",
    "शायद हमने तालाब के पास किंगफिशर देखा, पर वह जल्दी उड़ गया।",
    "दो कुत्ते थोड़ी देर तक हमारे साथ चले, फिर चाय की दुकान पर लौट गए।",
    "गीली मिट्टी की खुशबू आ रही थी।",
    "हम गेट के पास वाली बेंच पर थोड़ी देर बैठे।",
    "स्कूल के पास वाले नीम की एक बड़ी डाल टूट गई है।",
    "कोई छोटी सी भूरी चिड़िया फुटपाथ पर कूद रही थी।",
    "फूल वाले भैया मिले, बोले इस साल मोगरा देर से आया है।",
    "बस स्टॉप के पास तार पर तीन तोते बैठे थे।",
    "झील वाली सड़क बन रही है, तो हम लंबे रास्ते से आए।",
    "लगता है वह मैना थी, आँख के पास पीला सा था।",
    "पार्क के दूसरे कोने से मंदिर की घंटी सुनाई दे रही थी।",
    "लौटते समय हल्की बूंदाबांदी शुरू हुई, पर जल्दी रुक गई।",
    "पुराने बाग में आम के पेड़ों पर छोटे हरे आम आ गए हैं।",
    "बरगद के पेड़ पर चार गिलहरियाँ दौड़ रही थीं।",
    "फल वाले के पास इस मौसम के पहले सीताफल थे।",
    "कल से ठंडक ज़्यादा थी, झील से अच्छी हवा आ रही थी।",
    "हमने छत के पौधों के बारे में बात की।",
]
# (pool, lang_code, voices alternated)
LANGS = {
    "en": (EN, "a", ["af_heart", "am_michael"]),
    "hinglish": (HINGLISH, "h", ["hf_beta", "hm_psi"]),
    "hi": (HINDI, "h", ["hf_alpha", "hm_omega"]),
}


def build(lang: str, target_s: float, seed: int) -> tuple[np.ndarray, str]:
    pool, code, voices = LANGS[lang]
    rng = np.random.default_rng(seed)
    pcm: list[np.ndarray] = [np.zeros(int(0.3 * SR), np.float32)]
    used: list[str] = []
    total, i = 0.3, 0
    while total < target_s:
        if i >= len(pool):
            raise SystemExit(f"{lang}: pool exhausted at {total:.1f}s for target {target_s}s")
        s = tts.render(pool[i], voices[i % 2], code)
        gap = np.zeros(int(rng.uniform(0.4, 1.2) * SR), np.float32)
        pcm += [s, gap]
        used.append(pool[i])
        total += (len(s) + len(gap)) / SR
        i += 1
    return np.concatenate(pcm), " ".join(used)


def save(name: str, pcm: np.ndarray, text: str) -> Path:
    NOTES.mkdir(parents=True, exist_ok=True)
    wav = NOTES / f"{name}.wav"
    c.write_wav(wav, pcm, SR)
    (NOTES / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
    print(f"{name:16} {len(pcm) / SR:6.1f}s  words={len(text.split())}")
    return wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    seed = 7
    for lang in LANGS:
        for tgt in (20, 30, 60, 90):
            pcm, text = build(lang, tgt, seed)
            seed += 1
            save(f"{lang}_{tgt}s", pcm, text)
    pcm, text = build("en", 88, 99)
    if len(pcm) > 95 * SR:  # reference must stay exact, so never trim mid-sentence
        raise SystemExit("en_95s overshoot; lower the build target")
    pcm = np.pad(pcm, (0, int(95 * SR) - len(pcm)))  # trailing silence to exactly 95 s
    save("en_95s", pcm, text)
    rng = np.random.default_rng(1)
    save("silent_5s", (rng.standard_normal(5 * SR) * 10 ** (-60 / 20)).astype(np.float32), "")
    short = tts.render("Koel.", "af_heart", "a")
    short = np.pad(short, (0, max(0, int(0.5 * SR) - len(short))))[: int(0.5 * SR)]
    save("short_0_5s", short.astype(np.float32), "Koel.")
    if a.fixtures:
        FIX.mkdir(parents=True, exist_ok=True)
        for name, fx in [
            ("en_20s", "en_20s"),
            ("hinglish_20s", "hinglish_20s"),
            ("silent_5s", "silent_5s"),
            ("short_0_5s", "short_0_5s"),
            ("en_95s", "en_95s"),
        ]:
            for kind in ("webm", "m4a"):
                c.encode(NOTES / f"{name}.wav", FIX / f"{fx}.{kind}", kind)
            shutil.copy(NOTES / f"{name}.txt", FIX / f"{fx}.txt")
        print("fixtures:", sorted(p.name for p in FIX.iterdir()))


if __name__ == "__main__":
    main()
