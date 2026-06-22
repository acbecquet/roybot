from roybot import config

def test_constants_present_and_sane():
    assert config.CONTROL_HZ == 50
    assert config.SIM_TIMESTEP == 0.002
    # 50 Hz control over a 2 ms sim step = 10 substeps, exactly
    assert config.N_SUBSTEPS == 10
    assert config.BAND_MIN < config.BAND_MAX
    assert 0.0 < config.WILLING_THRESHOLD < 1.0
    assert config.WHEEL_RADIUS > 0 and config.WHEEL_BASE > 0
    assert set(config.REWARD_WEIGHTS) >= {
        "engage", "juke", "give_space", "pester", "energy", "action_rate", "tip"
    }
