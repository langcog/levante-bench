#!/usr/bin/env python3
"""
Generate a new picture-grid vocabulary corpus in a format compatible with the
levante-bench asset layout.

What it creates
---------------
- assets/<version>/manifest.csv
- assets/<version>/visual/vocab/*.png
- assets/<version>/translations/item-bank-translations.csv
- assets/<version>/metadata/vocab_generation_report.json

Design goals
------------
- Produce 170 newly-authored vocab items spanning approximate age bands 3-12.
- Preserve the levante-bench vocab manifest fields used by VocabDataset:
  task, item_uid, answer, response_alternatives, prompt_phrase, full_prompt.
- Create simple original icon-style images from scratch, reducing reuse of
  public benchmark assets.
- Control difficulty by age_band and distractor hardness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

CANVAS = (768, 768)
BG = (250, 248, 244)
FG = (45, 45, 50)
ACCENTS = [
    (34, 111, 168),
    (220, 120, 40),
    (78, 153, 92),
    (153, 78, 120),
    (190, 60, 60),
    (103, 93, 173),
    (52, 140, 132),
    (173, 124, 52),
]

PROMPT_EN = "Which picture shows {word}?"
PROMPT_DE = "Welches Bild zeigt {word}?"
PROMPT_ES = "¿Qué imagen muestra {word}?"
FULL_PROMPT_TEMPLATE = (
    "You will see four pictures labeled A, B, C, and D. "
    "Choose the picture that best matches the word: {word}."
)


@dataclass
class LexItem:
    word: str
    category: str
    age_band: int
    tags: List[str]
    shape: str
    hardness: int


LEXICON: List[LexItem] = [
    LexItem("ball", "toy", 3, ["round", "play"], "circle", 1),
    LexItem("cup", "household", 3, ["drink", "container"], "cup", 1),
    LexItem("shoe", "clothing", 3, ["foot", "wear"], "shoe", 1),
    LexItem("apple", "food", 3, ["fruit", "round", "red"], "apple", 1),
    LexItem("dog", "animal", 3, ["pet", "tail"], "dog", 1),
    LexItem("cat", "animal", 3, ["pet", "whiskers"], "cat", 1),
    LexItem("car", "vehicle", 3, ["wheels", "drive"], "car", 1),
    LexItem("book", "household", 3, ["read", "pages"], "book", 1),
    LexItem("tree", "nature", 3, ["plant", "leaves"], "tree", 1),
    LexItem("fish", "animal", 3, ["water", "swim"], "fish", 1),
    LexItem("chair", "household", 4, ["sit", "furniture"], "chair", 1),
    LexItem("spoon", "household", 4, ["eat", "utensil"], "spoon", 1),
    LexItem("banana", "food", 4, ["fruit", "yellow"], "banana", 1),
    LexItem("train", "vehicle", 4, ["tracks", "travel"], "train", 1),
    LexItem("flower", "nature", 4, ["petals", "plant"], "flower", 1),
    LexItem("bird", "animal", 4, ["wings", "fly"], "bird", 1),
    LexItem("sock", "clothing", 4, ["foot", "wear"], "sock", 1),
    LexItem("bed", "household", 4, ["sleep", "furniture"], "bed", 1),
    LexItem("moon", "nature", 4, ["night", "sky"], "moon", 1),
    LexItem("boat", "vehicle", 4, ["water", "travel"], "boat", 1),
    LexItem("clock", "household", 5, ["time", "numbers"], "clock", 2),
    LexItem("ladder", "tool", 5, ["climb", "rungs"], "ladder", 2),
    LexItem("candle", "household", 5, ["flame", "wax"], "candle", 2),
    LexItem("envelope", "object", 5, ["mail", "paper"], "envelope", 2),
    LexItem("pillow", "household", 5, ["sleep", "soft"], "pillow", 2),
    LexItem("rabbit", "animal", 5, ["ears", "hop"], "rabbit", 2),
    LexItem("carrot", "food", 5, ["orange", "vegetable"], "carrot", 2),
    LexItem("button", "object", 5, ["small", "fasten"], "button", 2),
    LexItem("bridge", "place", 5, ["cross", "river"], "bridge", 2),
    LexItem("magnet", "tool", 5, ["metal", "attract"], "magnet", 2),
    LexItem("helmet", "clothing", 6, ["protect", "head"], "helmet", 2),
    LexItem("igloo", "place", 6, ["snow", "house"], "igloo", 2),
    LexItem("anchor", "object", 6, ["ship", "heavy"], "anchor", 2),
    LexItem("ladle", "household", 6, ["soup", "utensil"], "ladle", 2),
    LexItem("hammer", "tool", 6, ["tool", "hit"], "hammer", 2),
    LexItem("planet", "nature", 6, ["space", "round"], "planet", 2),
    LexItem("bicycle", "vehicle", 6, ["pedal", "wheels"], "bicycle", 2),
    LexItem("blanket", "household", 6, ["warm", "sleep"], "blanket", 2),
    LexItem("feather", "nature", 6, ["bird", "light"], "feather", 2),
    LexItem("castle", "place", 6, ["tower", "stone"], "castle", 2),
    LexItem("compass", "tool", 7, ["direction", "needle"], "compass", 3),
    LexItem("fountain", "place", 7, ["water", "spray"], "fountain", 3),
    LexItem("mitten", "clothing", 7, ["hand", "winter"], "mitten", 3),
    LexItem("lantern", "object", 7, ["light", "carry"], "lantern", 3),
    LexItem("telescope", "tool", 7, ["space", "look"], "telescope", 3),
    LexItem("cactus", "nature", 7, ["desert", "spines"], "cactus", 3),
    LexItem("parachute", "object", 7, ["fall", "air"], "parachute", 3),
    LexItem("saddle", "object", 7, ["horse", "ride"], "saddle", 3),
    LexItem("chimney", "household", 7, ["roof", "smoke"], "chimney", 3),
    LexItem("suitcase", "object", 8, ["travel", "handle"], "suitcase", 3),
    LexItem("whistle", "object", 8, ["sound", "blow"], "whistle", 3),
    LexItem("microscope", "tool", 8, ["science", "lens"], "microscope", 3),
    LexItem("thermometer", "tool", 8, ["temperature", "measure"], "thermometer", 3),
    LexItem("trumpet", "music", 8, ["instrument", "brass"], "trumpet", 3),
    LexItem("pyramid", "place", 8, ["triangle", "stone"], "pyramid", 3),
    LexItem("trophy", "object", 8, ["award", "win"], "trophy", 3),
    LexItem("beaver", "animal", 8, ["dam", "river"], "beaver", 3),
    LexItem("compost", "nature", 8, ["soil", "decay"], "compost", 3),
    LexItem("geyser", "nature", 8, ["steam", "eruption"], "geyser", 3),
    LexItem("harpoon", "tool", 9, ["spear", "fish"], "harpoon", 4),
    LexItem("anvil", "tool", 9, ["metal", "forge"], "anvil", 4),
    LexItem("turbine", "machine", 9, ["spin", "energy"], "turbine", 4),
    LexItem("mural", "art", 9, ["wall", "painting"], "mural", 4),
    LexItem("satchel", "object", 9, ["bag", "strap"], "satchel", 4),
    LexItem("cobweb", "nature", 9, ["spider", "web"], "cobweb", 4),
    LexItem("stethoscope", "tool", 9, ["doctor", "listen"], "stethoscope", 4),
    LexItem("crater", "nature", 9, ["hole", "volcano"], "crater", 4),
    LexItem("pavilion", "place", 9, ["roof", "open"], "pavilion", 4),
    LexItem("goblet", "household", 9, ["cup", "stem"], "goblet", 4),
    LexItem("tundra", "nature", 10, ["cold", "plain"], "tundra", 4),
    LexItem("tapestry", "art", 10, ["fabric", "wall"], "tapestry", 4),
    LexItem("periscope", "tool", 10, ["submarine", "look"], "periscope", 4),
    LexItem("hourglass", "object", 10, ["sand", "time"], "hourglass", 4),
    LexItem("trellis", "place", 10, ["garden", "climb"], "trellis", 4),
    LexItem("cauldron", "household", 10, ["pot", "brew"], "cauldron", 4),
    LexItem("gargoyle", "art", 10, ["stone", "roof"], "gargoyle", 4),
    LexItem("vineyard", "place", 10, ["grapes", "field"], "vineyard", 4),
    LexItem("quill", "object", 10, ["feather", "write"], "quill", 4),
    LexItem("reef", "nature", 10, ["coral", "sea"], "reef", 4),
    LexItem("monocle", "object", 11, ["eye", "lens"], "monocle", 5),
    LexItem("turnstile", "machine", 11, ["gate", "spin"], "turnstile", 5),
    LexItem("sextant", "tool", 11, ["navigation", "angle"], "sextant", 5),
    LexItem("catapult", "machine", 11, ["launch", "arm"], "catapult", 5),
    LexItem("aviary", "place", 11, ["birds", "enclosure"], "aviary", 5),
    LexItem("turret", "place", 11, ["tower", "castle"], "turret", 5),
    LexItem("kelp", "nature", 11, ["sea", "plant"], "kelp", 5),
    LexItem("foundry", "place", 11, ["metal", "factory"], "foundry", 5),
    LexItem("totem", "art", 11, ["pole", "symbol"], "totem", 5),
    LexItem("cistern", "object", 11, ["water", "tank"], "cistern", 5),
    LexItem("astrolabe", "tool", 12, ["stars", "navigation"], "astrolabe", 5),
    LexItem("obelisk", "art", 12, ["stone", "pillar"], "obelisk", 5),
    LexItem("hammock", "object", 12, ["swing", "rest"], "hammock", 5),
    LexItem("windlass", "machine", 12, ["crank", "rope"], "windlass", 5),
    LexItem("mezzanine", "place", 12, ["floor", "indoor"], "mezzanine", 5),
    LexItem("trowel", "tool", 12, ["garden", "dig"], "trowel", 5),
    LexItem("trident", "tool", 12, ["three prongs", "spear"], "trident", 5),
    LexItem("weir", "place", 12, ["river", "barrier"], "weir", 5),
    LexItem("filigree", "art", 12, ["ornament", "metal"], "filigree", 5),
    LexItem("armillary", "tool", 12, ["rings", "astronomy"], "armillary", 5),
    LexItem("wagon", "vehicle", 3, ["wheels", "pull"], "wagon", 1),
    LexItem("drum", "music", 4, ["instrument", "beat"], "drum", 1),
    LexItem("kite", "toy", 5, ["fly", "string"], "kite", 2),
    LexItem("nest", "nature", 5, ["bird", "eggs"], "nest", 2),
    LexItem("glove", "clothing", 5, ["hand", "wear"], "glove", 2),
    LexItem("tractor", "vehicle", 6, ["farm", "wheels"], "tractor", 2),
    LexItem("camel", "animal", 6, ["desert", "humps"], "camel", 2),
    LexItem("pocket", "clothing", 6, ["carry", "fabric"], "pocket", 2),
    LexItem("valley", "place", 6, ["between hills", "land"], "valley", 2),
    LexItem("accordion", "music", 7, ["instrument", "fold"], "accordion", 3),
    LexItem("backpack", "object", 7, ["bag", "school"], "backpack", 3),
    LexItem("cocoon", "nature", 7, ["insect", "wrap"], "cocoon", 3),
    LexItem("fossil", "nature", 7, ["stone", "old"], "fossil", 3),
    LexItem("binoculars", "tool", 8, ["look", "two lenses"], "bicycle", 3),
    LexItem("carousel", "place", 8, ["ride", "horses"], "castle", 3),
    LexItem("mushroom", "nature", 8, ["fungus", "cap"], "tree", 3),
    LexItem("pepper", "food", 8, ["vegetable", "bell"], "apple", 3),
    LexItem("forge", "place", 9, ["metal", "fire"], "castle", 4),
    LexItem("orchard", "place", 9, ["trees", "fruit"], "tree", 4),
    LexItem("pulley", "machine", 9, ["rope", "wheel"], "button", 4),
    LexItem("harvest", "nature", 10, ["crops", "gather"], "tree", 4),
    LexItem("sconce", "household", 10, ["wall", "lamp"], "lantern", 4),
    LexItem("thimble", "object", 10, ["sew", "finger"], "button", 4),
    LexItem("canopy", "object", 10, ["cover", "overhead"], "parachute", 4),
    LexItem("parlor", "place", 11, ["room", "sitting"], "house", 5),
    LexItem("viaduct", "place", 11, ["bridge", "arches"], "bridge", 5),
    LexItem("yoke", "object", 11, ["oxen", "wood"], "magnet", 5),
    LexItem("sluice", "machine", 11, ["water", "gate"], "anchor", 5),
    LexItem("aperture", "object", 12, ["opening", "camera"], "button", 5),
    LexItem("bracken", "nature", 12, ["fern", "plants"], "tree", 5),
    LexItem("carafe", "household", 12, ["glass", "pour"], "cup", 5),
    LexItem("lintel", "place", 12, ["door", "beam"], "bridge", 5),
    LexItem("croquet", "toy", 9, ["game", "mallet"], "hammer", 4),
    LexItem("spigot", "object", 10, ["faucet", "pour"], "cup", 4),
    LexItem("cliff", "nature", 5, ["rock", "edge"], "bridge", 2),
    LexItem("swan", "animal", 6, ["bird", "lake"], "fish", 2),
    LexItem("barn", "place", 4, ["farm", "building"], "castle", 1),
    LexItem("teapot", "household", 4, ["pour", "tea"], "cup", 1),
    LexItem("stool", "household", 5, ["sit", "seat"], "chair", 2),
    LexItem("crown", "clothing", 6, ["head", "royal"], "helmet", 2),
    LexItem("icicle", "nature", 7, ["ice", "hanging"], "feather", 3),
    LexItem("medal", "object", 7, ["award", "wear"], "button", 3),
    LexItem("pouch", "object", 6, ["bag", "small"], "suitcase", 2),
    LexItem("skillet", "household", 8, ["pan", "cook"], "cauldron", 3),
    LexItem("quiver", "object", 10, ["arrows", "carry"], "suitcase", 4),
    LexItem("spindle", "tool", 11, ["spin", "thread"], "hammer", 5),
    LexItem("dune", "nature", 7, ["sand", "hill"], "pyramid", 3),
    LexItem("easel", "art", 8, ["paint", "stand"], "ladder", 3),
    LexItem("flute", "music", 7, ["instrument", "wind"], "trumpet", 3),
    LexItem("silo", "place", 8, ["farm", "tower"], "castle", 3),
    LexItem("gondola", "vehicle", 10, ["boat", "canal"], "boat", 4),
    LexItem("trombone", "music", 10, ["instrument", "slide"], "trumpet", 4),
    LexItem("mosaic", "art", 9, ["tiles", "picture"], "book", 4),
    LexItem("pagoda", "place", 11, ["tower", "roof tiers"], "castle", 5),
    LexItem("cairn", "nature", 12, ["stone stack", "trail"], "button", 5),
    LexItem("urn", "object", 9, ["vase", "container"], "cup", 4),
    LexItem("lily", "nature", 5, ["flower", "petals"], "flower", 2),
    LexItem("otter", "animal", 6, ["river", "swim"], "fish", 2),
    LexItem("beehive", "nature", 6, ["bees", "home"], "button", 2),
    LexItem("scepter", "object", 11, ["royal", "staff"], "hammer", 5),
    LexItem("alcove", "place", 12, ["nook", "wall"], "house", 5),
    LexItem("wrench", "tool", 7, ["tool", "bolt"], "hammer", 3),
    LexItem("apron", "clothing", 5, ["kitchen", "wear"], "blanket", 2),
    LexItem("lighthouse", "place", 8, ["tower", "sea"], "castle", 3),
    LexItem("harbor", "place", 8, ["boats", "water"], "castle", 3),
    LexItem("violin", "music", 8, ["instrument", "strings"], "trumpet", 3),
    LexItem("porch", "place", 6, ["house", "steps"], "castle", 2),
    LexItem("acorn", "nature", 5, ["oak", "seed"], "apple", 2),
    LexItem("latch", "object", 9, ["door", "fasten"], "button", 4),
    LexItem("mantle", "household", 10, ["fireplace", "shelf"], "book", 4),
    LexItem("trench", "place", 11, ["ditch", "ground"], "bridge", 5),
    LexItem("orb", "object", 12, ["sphere", "round"], "circle", 5),
    LexItem("crutch", "object", 9, ["support", "walk"], "ladder", 4),
    LexItem("sundial", "object", 10, ["time", "sun"], "clock", 4),
]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def ensure_unique_lexicon(items: List[LexItem]) -> List[LexItem]:
    seen = {}
    out = []
    for item in items:
        key = item.word.lower()
        if key not in seen:
            out.append(item)
            seen[key] = 1
    return out


def choose_distractors(target: LexItem, pool: List[LexItem], rng: random.Random) -> List[LexItem]:
    same_cat = [x for x in pool if x.word != target.word and x.category == target.category]
    same_band = [x for x in pool if x.word != target.word and abs(x.age_band - target.age_band) <= 1]
    similar_shape = [x for x in pool if x.word != target.word and x.shape == target.shape]
    broad = [x for x in pool if x.word != target.word]

    distractors: List[LexItem] = []

    def add_from(cands):
        rng.shuffle(cands)
        for c in cands:
            if c.word != target.word and c.word not in {d.word for d in distractors}:
                distractors.append(c)
                if len(distractors) >= 3:
                    return

    if target.hardness <= 2:
        add_from([x for x in same_band if x.category != target.category])
        add_from(similar_shape)
        add_from(broad)
    elif target.hardness <= 4:
        add_from(same_cat)
        add_from(similar_shape)
        add_from(same_band)
    else:
        add_from(same_cat)
        add_from(same_cat)
        add_from(same_band)

    if len(distractors) < 3:
        add_from(broad)
    return distractors[:3]


def draw_word_image(word: str, shape: str, outpath: Path, seed: int) -> None:
    img = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(img)
    accent = ACCENTS[seed % len(ACCENTS)]
    accent2 = ACCENTS[(seed + 3) % len(ACCENTS)]

    cx, cy = CANVAS[0] // 2, CANVAS[1] // 2 - 35
    w, h = 320, 240

    def ellipse(box, fill=None, outline=FG, width=10):
        draw.ellipse(box, fill=fill, outline=outline, width=width)

    def rect(box, fill=None, outline=FG, width=10, radius=28):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def line(points, fill=FG, width=12):
        draw.line(points, fill=fill, width=width, joint="curve")

    sx = cx - w // 2
    sy = cy - h // 2
    ex = cx + w // 2
    ey = cy + h // 2

    if shape in {"circle", "apple", "planet", "button", "clock", "compass"}:
        ellipse((sx, sy, ex, ey), fill=accent)
        if shape == "apple":
            line([(cx, sy - 10), (cx + 10, sy - 45)], width=10)
            ellipse((cx + 10, sy - 55, cx + 70, sy - 5), fill=(90, 160, 90), width=6)
        if shape in {"clock", "compass"}:
            line([(cx, cy), (cx, cy - 70)], width=12)
            line([(cx, cy), (cx + 50, cy + 20)], width=10)
    elif shape in {"cup", "goblet", "carafe", "cauldron"}:
        rect((sx + 40, sy + 20, ex - 40, ey - 20), fill=accent)
        line([(ex - 45, cy - 35), (ex + 35, cy)], width=14)
        line([(ex + 35, cy), (ex - 45, cy + 35)], width=14)
    elif shape in {"book", "envelope", "suitcase", "satchel"}:
        rect((sx, sy, ex, ey), fill=accent)
        if shape == "envelope":
            line([(sx, sy), (cx, cy + 25), (ex, sy)], width=10)
        if shape in {"suitcase", "satchel"}:
            line([(cx - 55, sy), (cx - 35, sy - 35), (cx + 35, sy - 35), (cx + 55, sy)], width=12)
    elif shape in {"shoe", "boat", "banana"}:
        pts = [(sx, ey - 20), (sx + 80, sy + 60), (ex - 70, sy + 70), (ex, ey - 30), (sx + 110, ey)]
        draw.polygon(pts, fill=accent, outline=FG)
    elif shape in {"car", "wagon", "tractor", "train"}:
        rect((sx + 20, sy + 70, ex - 20, ey - 20), fill=accent)
        rect((sx + 80, sy + 10, ex - 100, sy + 100), fill=accent2)
        ellipse((sx + 60, ey - 40, sx + 140, ey + 40), fill=(80, 80, 80), width=8)
        ellipse((ex - 140, ey - 40, ex - 60, ey + 40), fill=(80, 80, 80), width=8)
    elif shape in {"tree", "flower", "cactus", "lily"}:
        line([(cx, ey + 20), (cx, cy)], width=18)
        ellipse((cx - 110, sy - 10, cx + 110, cy + 90), fill=accent)
        if shape == "flower":
            for ang in range(0, 360, 60):
                dx = int(math.cos(math.radians(ang)) * 70)
                dy = int(math.sin(math.radians(ang)) * 70)
                ellipse((cx + dx - 45, cy + dy - 45, cx + dx + 45, cy + dy + 45), fill=accent)
            ellipse((cx - 35, cy - 35, cx + 35, cy + 35), fill=(230, 190, 40), width=6)
    else:
        rect((sx, sy, ex, ey), fill=accent)
        ellipse((sx + 80, sy + 50, ex - 80, ey - 50), fill=accent2, width=8)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
    except Exception:
        font = ImageFont.load_default()
    label = word.title()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((CANVAS[0] - tw) / 2, CANVAS[1] - 110), label, fill=FG, font=font)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    img.save(outpath)


def build_manifest_rows(items: List[LexItem], rng: random.Random, max_items: int) -> Tuple[List[dict], Dict[str, dict]]:
    rows = []
    meta = {}
    selected = sorted(items, key=lambda x: (x.age_band, x.category, x.word))[:max_items]
    for item in selected:
        distractors = choose_distractors(item, items, rng)
        uid = f"vocab__{slugify(item.word)}"
        rows.append(
            {
                "task": "vocab",
                "item_uid": uid,
                "answer": item.word,
                "response_alternatives": ",".join([d.word for d in distractors]),
                "prompt_phrase": item.word,
                "full_prompt": FULL_PROMPT_TEMPLATE.format(word=item.word),
                "trial_type": "test",
                "age_band": item.age_band,
                "category": item.category,
                "hardness": item.hardness,
            }
        )
        meta[uid] = {
            "target": asdict(item),
            "distractors": [asdict(d) for d in distractors],
        }
    return rows, meta


def write_manifest(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "item_uid", "answer", "response_alternatives", "prompt_phrase", "full_prompt", "trial_type", "age_band", "category", "hardness"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_translations(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item_uid", "task", "language", "prompt_phrase", "full_prompt"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            for lang, tmpl in [("en", PROMPT_EN), ("de", PROMPT_DE), ("es", PROMPT_ES)]:
                writer.writerow(
                    {
                        "item_uid": r["item_uid"],
                        "task": "vocab",
                        "language": lang,
                        "prompt_phrase": r["answer"],
                        "full_prompt": tmpl.format(word=r["answer"]),
                    }
                )


def render_images(rows: List[dict], items_map: Dict[str, LexItem], visual_dir: Path) -> None:
    for row in rows:
        words = [row["answer"]] + [x.strip() for x in row["response_alternatives"].split(",")]
        for w in words:
            item = items_map[w]
            fn = visual_dir / f"{slugify(w)}.png"
            if not fn.exists():
                draw_word_image(w, item.shape, fn, seed=abs(hash(w)) % 10000)


def write_report(rows: List[dict], meta: Dict[str, dict], out_json: Path, seed: int) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    by_age = defaultdict(int)
    by_cat = defaultdict(int)
    for r in rows:
        by_age[str(r["age_band"])] += 1
        by_cat[r["category"]] += 1
    report = {
        "seed": seed,
        "n_items": len(rows),
        "age_band_counts": dict(sorted(by_age.items(), key=lambda kv: int(kv[0]))),
        "category_counts": dict(sorted(by_cat.items())),
        "notes": [
            "Manifest preserves the columns required by src/levante_bench/tasks/vocab.py.",
            "Images are generated de novo as simple icon-style PNGs named after the corresponding word.",
            "For strongest contamination protection, keep a held-out split private and manually review all items.",
        ],
        "sample_items": dict(list(meta.items())[:5]),
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="./new_vocab_assets")
    parser.add_argument("--version", default="new-vocab-2026-04-24")
    parser.add_argument("--n-items", type=int, default=170)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lexicon = ensure_unique_lexicon(LEXICON)
    if args.n_items > len(lexicon):
        raise SystemExit(f"Requested {args.n_items} items but only {len(lexicon)} unique lexicon entries are available.")

    out_root = Path(args.out_root)
    version_root = out_root / "assets" / args.version
    visual_dir = version_root / "visual" / "vocab"

    rows, meta = build_manifest_rows(lexicon, rng, args.n_items)
    items_map = {x.word: x for x in lexicon}

    write_manifest(rows, version_root / "manifest.csv")
    write_translations(rows, version_root / "translations" / "item-bank-translations.csv")
    render_images(rows, items_map, visual_dir)
    write_report(rows, meta, version_root / "metadata" / "vocab_generation_report.json", args.seed)

    print(f"Wrote {len(rows)} items to {version_root}")


if __name__ == "__main__":
    main()