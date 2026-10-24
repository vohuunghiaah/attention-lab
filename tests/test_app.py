"""Exercise real Streamlit controls without requiring a graphical browser."""

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from attention_lab.playground import token_projections


def test_token_mapping() -> None:
    a = token_projections(["Tôi", "học", "Tôi"], 8, 42)
    b = token_projections(["Tôi", "học", "Tôi"], 8, 42)
    for first, second in zip(a, b, strict=True):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(a[0][0], a[0][2])
    assert not np.array_equal(a[0], token_projections(["Tôi", "học", "Tôi"], 8, 43)[0])


def test_app_controls() -> None:
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app" / "main.py"), default_timeout=30
    ).run()
    assert not app.exception
    for table in app.dataframe[:3]:
        assert table.value.index.name is None
        assert list(table.value.index) == [0, 1, 2]
        assert list(table.value.columns) == [0, 1]
    initial = app.dataframe[5].value.copy()
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert not app.dataframe[5].value.equals(initial)
    np.testing.assert_array_equal(app.dataframe[5].value.to_numpy()[0], [1, 0, 0])
    app.selectbox[0].set_value(2).run()
    app.slider[0].set_value(1).run()
    app.selectbox[1].set_value(12).run()
    app.selectbox[2].set_value(3).run()
    app.number_input[0].set_value(123).run()
    assert not app.exception
    app.selectbox[1].set_value(8).run()
    assert len(app.error) == 1  # 8 is not divisible by 3
    app.selectbox[2].set_value(2).run()
    app.radio[0].set_value("Chuỗi token tùy chỉnh").run()
    app.text_input[0].set_value("một hai ba bốn").run()
    app.select_slider[0].set_value(16).run()
    assert not app.exception
    assert app.dataframe[0].value.shape == (4, 16)
    app.text_input[0].set_value("").run()
    assert len(app.error) == 1 and not app.exception
    app.text_input[0].set_value("token " * 25).run()
    assert len(app.error) == 1 and not app.exception
