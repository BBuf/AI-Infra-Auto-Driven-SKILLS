import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "sglang-explainer-video" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import cover_kit as K  # noqa: E402


def _css(width, dy):
    return f"#cv{{width:{width}px}} #cv-in{{top:{dy}px}}"


def _body(prefix):
    return f'<div id="cv-in"><div id="cv-title">一串感叹号</div><img src="{prefix}assets/a.png"></div>'


def test_frame_lasts_the_narration_plus_the_crossfade(tmp_path):
    K.write_frame(str(tmp_path), _css, _body, 19.896, '        tl.fromTo("#cv-title", {opacity: 0}, {opacity: 1}, 0.2);', ["#cv-title"])
    html = (tmp_path / "compositions" / "frames" / "00-cover.html").read_text()
    assert html.startswith("<template>")
    assert 'data-composition-id="00-cover" data-start="0" data-duration="20.396"' in html
    assert "#cv{width:1920px} #cv-in{top:0px}" in html  # the frame is not shifted
    assert 'src="assets/a.png"' in html  # project-relative assets
    assert 'tl.to(["#cv-title"], { opacity: 0, duration: 0.25, ease: "power2.in" }, 19.696);' in html
    assert 'window.__timelines["00-cover"] = tl;' in html


def test_stills_use_the_full_cjk_fonts(tmp_path, monkeypatch):
    monkeypatch.setenv("SEV_WS", "/ws")
    faces = K.still_faces("/proj")
    assert 'url("file:///ws/fonts/NotoSerifSC.ttf")' in faces
    assert 'url("file:///proj/assets/fonts/JetBrainsMono-700.woff2")' in faces
    assert "Subset" not in faces
