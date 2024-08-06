import xarray as xr
from tests.data import input_data, raw_prediction  # noqa # Might want to put these in conftest.py (see https://stackoverflow.com/questions/73191533/using-conftest-py-vs-importing-fixtures-from-dedicate-modules)
from ocean_emulators.postprocessing import post_processor


def test_post_processor(input_data, raw_prediction):
    ds_input = input_data
    ds_raw = raw_prediction
    ds = post_processor(ds_raw, ds_input)
    assert set(ds.data_vars) == set(["so", "thetao", "zos", "uo", "vo"])
    assert ds.sizes == {"time": 3, "x": 360, "y": 180, "lev": 19}
    assert set(["x", "y", "time"]) == set(ds["zos"].dims)
    for co in ds.coords:
        xr.testing.assert_equal(ds[co], ds_input[co])


def test_prediction_data_test():
    pass
    # TODO: Check each test in there with a failcase
