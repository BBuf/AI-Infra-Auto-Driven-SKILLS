import importlib.util
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "sglang-explainer-video" / "scripts"
spec = importlib.util.spec_from_file_location("tone_take", SCRIPTS / "tone_take.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
QIE = T.tone_table()["切"]


def test_qie_is_qie1_unless_inside_a_listed_word():
    assert T.intended("DCP 也不切它。", "DCP 也不切它。".index("切"), QIE) == 1
    assert T.intended("把 KV 切成 8 份。", "把 KV 切成 8 份。".index("切"), QIE) == 1
    assert T.intended("一切都和单卡一样。", 1, QIE) == 4
    assert T.intended("两者关系密切。", "两者关系密切。".index("切"), QIE) == 4
    assert T.intended("切实可行", 0, QIE) == 4


def test_verdict_puts_the_take_next_to_the_closer_tone():
    # Slopes in semitones over the vowel: tone 1 stays level, tone 4 falls (DCP video, 「也不切它」 vs its controls).
    groups = {1: [1.1, 0.4, 0.9, 1.3], 4: [-3.4, -3.0, -4.1, -2.9]}
    heard, margin = T.verdict(-4.3, groups)
    assert heard == 4 and margin > 3
    heard, margin = T.verdict(0.2, groups)
    assert heard == 1 and margin > 2


def test_verdict_refuses_when_the_controls_do_not_separate():
    assert T.verdict(-1.0, {1: [-1.2, -0.8], 4: [-1.5, -1.1]}) == (None, 0.0)
    assert T.verdict(-1.0, {1: [-1.2]}) == (None, 0.0)


def test_mfa_transcript_renames_only_the_target_and_marks_latin_as_noise():
    base = {"也": ["j e˨˩˦"], "不": ["p u˥˩"], "切": ["tɕʰ e˥˩", "tɕʰ e˥"], "它": ["tʰ a˥"]}
    text = "DCP 也不切它。"
    lab, words = T.lab_and_words(text, text.index("切"), "zzt7", base)
    assert lab == "dcp 也 不 zzt7 它"
    assert words["zzt7"] == ["tɕʰ e˥˩", "tɕʰ e˥"]
    assert words["dcp"] == ["spn"] and words["不"] == ["p u˥˩"]


def test_mfa_transcript_keeps_other_occurrences_of_the_character():
    base = {"切": ["tɕʰ e˥"], "片": ["pʰ j ɛ n˥˩"], "开": ["kʰ aj˥"]}
    text = "切开切片"
    lab, words = T.lab_and_words(text, 2, "zzt0", base)
    assert lab == "切 开 zzt0 片"
    assert words["切"] == ["tɕʰ e˥"] and words["zzt0"] == ["tɕʰ e˥"]


def test_tone_table_entries_are_complete():
    data = json.load(open(SCRIPTS / "pronunciation.json", encoding="utf-8"))
    for ch, spec in T.tone_table().items():
        assert len(ch) == 1
        tones = set(spec["readings"])
        assert str(spec["default"]) in tones
        assert set(spec["controls"]) == tones and all(len(v) >= 2 for v in spec["controls"].values())
        assert set(spec.get("words", {})) <= tones
        for chars in spec["controls"].values():
            assert ch not in chars
    assert "_doc" in data["tones"]
