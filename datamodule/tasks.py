import numpy as np

# return a tuple, first is whether the regression task, second is the filter function
oct_task_list = {
    "Healthy": (False, None),
    "EarlyIntermediate": (False, lambda s: s["Healthy"] | s["EarlyIntermediate"]),
    "Late": (False, lambda s: s["Late"] | s["EarlyIntermediate"]),
    "VALogMAR": (True, lambda s: (~np.isnan(s["VALogMAR"]))),
    "Dry": (False, lambda s: s["Wet"] | s["Dry"]),
    "CurrentAge_normalised": (True, None),
    "Sex": (False, None),
    "Time_of_day_normalised": (True, None),
    "Conversion-CNV_250_1000": (True, lambda s: s["Conversion-CNV_250_1000"] > 0),
    "Conversion-250_1000": (True, lambda s: s["Conversion-250_1000"] > 0),
    "Converts_to_CNV_within_0.5_years": (False, lambda s: (~np.isnan(s["Converts_to_CNV_within_0.5_years"]))),
    "Converts_to_CNV_within_1_years": (False, lambda s: (~np.isnan(s["Converts_to_CNV_within_1_years"]))),
    "Converts_to_CNV_within_3_years": (False, lambda s: (~np.isnan(s["Converts_to_CNV_within_3_years"]))),
    "Converts_to_cRORA of 250 um_within_2_years": (
        False,
        lambda s: (~np.isnan(s["Converts_to_cRORA of 250 um_within_2_years"])),
    ),
    "Converts_to_cRORA of 1000 um_within_3_years": (
        False,
        lambda s: (~np.isnan(s["Converts_to_cRORA of 1000 um_within_3_years"])),
    ),
    "Converts_to_Scar_within_3_years": (False, lambda s: (~np.isnan(s["Converts_to_Scar_within_3_years"]))),
}


cardiac_task_is_regression = {
    "Hypertension": False,
    "CAD_broad": False,
    "Diabetes": False,
    "Fibrillation": False,
    "LVEF": True,
    "LVSV": True,
    "LVEDV": True,
    "LVESV": True,
    "CIndex": True,
    "CO": True,
    "LVM": True,
}
