"""Minimal BRO XML readers for borehole (BHR_G) and CPT documents.

There were no existing borehole/CPT-XML readers in this repository to reuse
(only `geolib_plus`'s CPT-focused `BroXmlCpt`, which does not expose
borehole layer descriptions at all) - so this module parses the BRO XML
directly with `xml.etree.ElementTree`, stripping namespaces for simplicity.
Only the handful of fields this workflow actually needs are extracted; nothing
is fabricated or inferred.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union


def _strip_namespaces(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _load_root(path: Union[str, Path]) -> ET.Element:
    tree = ET.parse(str(path))
    return _strip_namespaces(tree.getroot())


def _find_rd_coords(root: ET.Element, container_tag: str) -> Optional[Dict[str, float]]:
    """Extract RD (EPSG:28992) x/y from the first `<pos>` under `container_tag`.

    BRO documents also carry a `standardizedLocation` (WGS84 lat/lon) block;
    restricting the search to `container_tag` (e.g. "deliveredLocation")
    avoids picking that up by mistake.
    """
    container = root.find(f".//{container_tag}")
    if container is None:
        return None
    pos_el = container.find(".//pos")
    if pos_el is None or not pos_el.text:
        return None
    parts = pos_el.text.split()
    if len(parts) < 2:
        return None
    return {"x": float(parts[0]), "y": float(parts[1])}


def read_cpt_xml_metadata(path: Union[str, Path]) -> Dict[str, object]:
    """Read just the identifying metadata (BRO id, RD coordinates) of a CPT XML.

    Used only to confirm which CPT a borehole is being associated with; full
    CPT measurement parsing is left to `geolib_plus.BroXmlCpt`.
    """
    root = _load_root(path)
    bro_id = root.findtext(".//broId")
    coords = _find_rd_coords(root, "deliveredLocation")
    return {"path": str(path), "bro_id": bro_id, "x": coords["x"] if coords else None,
            "y": coords["y"] if coords else None}


def read_borehole_xml(path: Union[str, Path]) -> Dict[str, object]:
    """Read borehole (BHR_G) metadata and its full layer log from a BRO XML file.

    Args:
        path (str or Path): path to the `BHR_*.xml` document.

    Returns:
        dict: {
            "path", "bro_id", "nitg_code", "x", "y",
            "surface_elevation_m_nap", "vertical_datum", "vertical_reference_point",
            "final_depth_m", "layers": [ {...}, ... ]
        }
        Each layer dict preserves the ORIGINAL raw BRO field values verbatim
        (no translation/normalization) plus "upper_boundary_m"/"lower_boundary_m".
    """
    root = _load_root(path)

    bro_id = root.findtext(".//broId")
    nitg_code = root.findtext(".//NITGCode")
    coords = _find_rd_coords(root, "deliveredLocation")

    vertical_position = root.find(".//deliveredVerticalPosition")
    surface_elevation = None
    vertical_datum = None
    vertical_reference_point = None
    if vertical_position is not None:
        offset_text = vertical_position.findtext("offset")
        surface_elevation = float(offset_text) if offset_text else None
        vertical_datum = vertical_position.findtext("verticalDatum")
        vertical_reference_point = vertical_position.findtext("localVerticalReferencePoint")

    final_depth_text = root.findtext(".//finalDepthBoring")
    final_depth_m = float(final_depth_text) if final_depth_text else None

    layers: List[Dict[str, object]] = []
    for layer_el in root.findall(".//layer/Layer"):
        soil_el = layer_el.find("soil")
        upper_text = layer_el.findtext("upperBoundary")
        lower_text = layer_el.findtext("lowerBoundary")
        layers.append({
            "upper_boundary_m": float(upper_text) if upper_text is not None else None,
            "lower_boundary_m": float(lower_text) if lower_text is not None else None,
            "upper_boundary_determination": layer_el.findtext("upperBoundaryDetermination"),
            "lower_boundary_determination": layer_el.findtext("lowerBoundaryDetermination"),
            "soil_name_nen5104": soil_el.findtext("soilNameNEN5104") if soil_el is not None else None,
            "gravel_content_class": soil_el.findtext("gravelContentClass") if soil_el is not None else None,
            "organic_matter_content_class": (
                soil_el.findtext("organicMatterContentClassNEN5104") if soil_el is not None else None
            ),
            "carbonate_content_class": soil_el.findtext("carbonateContentClass") if soil_el is not None else None,
            "colour": soil_el.findtext("colour") if soil_el is not None else None,
            "mottled": soil_el.findtext("mottled") if soil_el is not None else None,
            "sand_median_class": soil_el.findtext(".//sandMedianClass") if soil_el is not None else None,
            "peat_constituent": soil_el.findtext(".//peatConstituent") if soil_el is not None else None,
        })

    return {
        "path": str(path), "bro_id": bro_id, "nitg_code": nitg_code,
        "x": coords["x"] if coords else None, "y": coords["y"] if coords else None,
        "surface_elevation_m_nap": surface_elevation, "vertical_datum": vertical_datum,
        "vertical_reference_point": vertical_reference_point,
        "final_depth_m": final_depth_m, "layers": layers,
    }
