"""The fill's answer says how long it took and where it ran, in Hyimg's words for a file's json (owner 2026-10-11: «затраченное время на
генерацию или рендеринг, обязательный параметр ... либо compute power какой-то условный»): "seconds" and "compute" beside the old "ms",
so the editor keeps them on the layer the fill makes."""
import pytest

from test_inpaint_routes import call, fake_lama, picture, png_url  # noqa: F401  (fixtures)

Image = pytest.importorskip("PIL.Image")


def test_inpaint_answers_its_seconds_and_where_it_ran(routes, fake_lama, picture):
    mask = Image.new("L", (6, 4), 0)
    code, d = call(routes.inpaint, {"image": png_url(picture), "mask": png_url(mask)})
    assert code == 200 and isinstance(d["seconds"], float) and d["seconds"] == round(d["ms"] / 1000, 3)
    assert d["compute"] == "local · LaMa ONNX · CPU"
