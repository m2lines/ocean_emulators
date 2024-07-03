import xarray as xr
from ocean_emulators.postprocessing import post_processor
from data import input_data, raw_prediction

def test_post_processor(input_data, raw_prediction):
    ds_input = input_data
    ds_raw = raw_prediction
    ds = post_processor(ds_raw, ds_input)
    assert set(ds.data_vars) == set(["so", "thetao", "zos", "uo", "vo"])
    assert ds.sizes == {"time": 3, "x": 360, "y": 180, "lev": 19}
    for co in ds.coords:
        xr.testing.assert_equal(ds[co], ds_input[co])

def test_prediction_data_test():
    pass
    #TODO: Check each test in there with a failcase