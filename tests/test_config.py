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

def test_phase15_constants_present():
    assert config.N_STACK >= 2
    assert "anticipate" in config.REWARD_WEIGHTS
    assert config.ANTICIPATE_HORIZON > 0
    for k in ("CAT_STALK_SPEED", "CAT_POUNCE_SPEED", "CAT_BAT_SPEED",
              "CAT_DART_SPEED", "CAT_MODE_MIN_S", "CAT_MODE_MAX_S"):
        assert hasattr(config, k)
    assert config.CAT_MODE_MIN_S < config.CAT_MODE_MAX_S
    assert config.DR_MASS[0] < 1.0 < config.DR_MASS[1]
    assert config.DIFFICULTY_RANGE[0] <= config.DIFFICULTY_RANGE[1]
