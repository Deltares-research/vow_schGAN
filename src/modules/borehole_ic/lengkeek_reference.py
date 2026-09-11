"""Reference CPT Lengkeek classification, reused as-is from `geolib_plus`.

This repository has no existing Lengkeek classifier; `geolib_plus` (the venv's
"VOW modified build") already provides one end-to-end:
    `BroXmlCpt` (reads the CPT XML directly, no GEF needed)
    -> `RobertsonCptInterpretation(interpretation_method=LENGKEEK_2024)`
    -> `NEN_classification(interpretation_method=2)` for the Dutch main/qualifier names.
Nothing about the Lengkeek classification itself is reimplemented here.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from geolib_plus.bro_xml_cpt import BroXmlCpt
from geolib_plus.cpt_utils.NEN9997 import NEN_classification
from geolib_plus.robertson_cpt_interpretation import InterpretationMethod, RobertsonCptInterpretation

LENGKEEK_IMPLEMENTATION_NOTE = (
    "geolib_plus (VOW modified build) BroXmlCpt + "
    "RobertsonCptInterpretation(interpretation_method=InterpretationMethod.LENGKEEK_2024) "
    "+ cpt_utils.NEN9997.NEN_classification(interpretation_method=2) for Dutch main/qualifier names."
)


def build_reference_lengkeek_profile(cpt_xml_path: Union[str, Path]) -> pd.DataFrame:
    """Interpret a BRO CPT XML with the Lengkeek (2024) method via `geolib_plus`.

    Returns:
        pandas.DataFrame with columns: depth_m, elevation_m_nap, lithology_index,
        main_group (Dutch), qualifier (Dutch), Ic (geolib_plus's own Ic, kept
        only as a cross-reference - NOT used anywhere in the synthetic profile).
    """
    cpt = BroXmlCpt()
    cpt.read(Path(cpt_xml_path))
    cpt.pre_process_data()

    interpretation = RobertsonCptInterpretation(interpretation_method=InterpretationMethod.LENGKEEK_2024)
    cpt.interpret_cpt(interpretation)

    classifier = NEN_classification(interpretation_method=2)
    main_groups, qualifiers = [], []
    for lit in cpt.lithology:
        entry = classifier.map_dict.get(int(lit))
        if entry is None:
            main_groups.append(None)
            qualifiers.append(None)
        else:
            main_groups.append(entry[0][0])
            qualifiers.append(entry[1][0][0])

    ic = getattr(cpt, "IC", None)
    return pd.DataFrame({
        "depth_m": np.asarray(cpt.depth),
        "elevation_m_nap": np.asarray(cpt.depth_to_reference),
        "lithology_index": [int(l) for l in cpt.lithology],
        "main_group": main_groups,
        "qualifier": qualifiers,
        "reference_ic_geolib_plus": np.asarray(ic) if ic is not None else np.nan,
    })
