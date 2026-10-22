"""Build the voice-intent task files from AmazonScience/massive (all_1.1).

    uv run python voice-intent/prepare.py [OUT_DIR]     # OUT_DIR defaults to voice-intent/

The source is a parallel corpus: every utterance `id` appears once per locale with
the same intent. Agents must not be able to regroup those translations, otherwise
they predict one locale and copy the answer to the other 51. So `id` is dropped,
and rows are shuffled (the source stores 52 locale blocks in identical id order,
so a sequential row_id would encode the group as row_id % 2974). Columns are
renamed and `partition` dropped so the files don't advertise the source dataset.
Only this script and dataset.toml name the source; agent-facing files must not.

Output is deterministic in content. Parquet bytes also depend on the pyarrow
version (it is recorded in the file), so compare reruns with pyarrow, not sha256.
GCS holds the published copy; input/, private/ and raw/ are not kept locally.

Writes (input/ is what gets published to the agent bucket):
    OUT_DIR/input/train.parquet            train + validation, shuffled, without id/partition
    OUT_DIR/input/test.parquet             row_id, locale, text (no labels), shuffled
    OUT_DIR/input/sample_submission.csv    row_id,intent
    OUT_DIR/input/intents.csv              intent,name
    OUT_DIR/input/scenarios.csv            scenario,name
    OUT_DIR/private/test_labels.csv        row_id,locale,intent
    OUT_DIR/raw/                           downloaded source parquets
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://huggingface.co/datasets/AmazonScience/massive/resolve/refs%2Fconvert%2Fparquet/all_1.1"
SPLITS = ["train", "validation", "test"]
SHA256 = {  # pinned so a changed upstream conversion fails loudly
    "train": "e57e656702b722d8880118c491e298c4b09f08d1d70c559f98a3b38933047c02",
    "validation": "e6dd096bc1be7837376887e0a714392b5274ebd90890500eda8d1afca6adf75d",
    "test": "4b0d0b70ba4103007aeb3c6b2218f691cbc9be38280ed0872d29facc189137d6",
}
# ClassLabel names from the dataset loader (massive.py); index = integer id.
SCENARIOS = [
    "social",
    "transport",
    "calendar",
    "play",
    "news",
    "datetime",
    "recommendation",
    "email",
    "iot",
    "general",
    "audio",
    "lists",
    "qa",
    "cooking",
    "takeaway",
    "music",
    "alarm",
    "weather",
]
INTENTS = [
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query",
]
SEED = 20261019
RENAME = {"utt": "text", "annot_utt": "annotated_text"}
DROP = ["id", "partition"]
TEST_COLUMNS = ["row_id", "locale", "text"]


def download(raw: Path) -> dict[str, pd.DataFrame]:
    raw.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in SPLITS:
        path = raw / f"massive_{split}.parquet"
        if not path.exists():
            urllib.request.urlretrieve(f"{BASE}/{split}/0000.parquet", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == SHA256[split], f"{path.name}: sha256 {digest} != pinned {SHA256[split]}"
        frames[split] = pd.read_parquet(path)
    return frames


def main(out: Path) -> None:
    frames = download(out / "raw")
    public, private = out / "input", out / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
    train = train.sample(frac=1, random_state=SEED).drop(columns=DROP).rename(columns=RENAME)
    train.to_parquet(public / "train.parquet", index=False)

    # Official test split, all rows kept; shuffled so row_id carries no group information.
    test = frames["test"]
    assert not test.duplicated(["id", "locale"]).any()
    test = test.sample(frac=1, random_state=SEED + 1).drop(columns=DROP).rename(columns=RENAME)
    test.insert(0, "row_id", range(len(test)))
    test[TEST_COLUMNS].to_parquet(public / "test.parquet", index=False)
    test[["row_id", "locale", "intent"]].to_csv(private / "test_labels.csv", index=False)

    pd.DataFrame({"intent": range(len(INTENTS)), "name": INTENTS}).to_csv(public / "intents.csv", index=False)
    pd.DataFrame({"scenario": range(len(SCENARIOS)), "name": SCENARIOS}).to_csv(
        public / "scenarios.csv", index=False
    )

    majority = int(train["intent"].mode()[0])
    test[["row_id"]].assign(intent=majority).to_csv(public / "sample_submission.csv", index=False)

    print(
        f"train {train.shape}  test {test.shape}  intents {train['intent'].nunique()}  "
        f"majority-class test accuracy {(test['intent'] == majority).mean():.4f}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
