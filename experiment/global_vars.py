from pathlib import Path

# ======================================================================================================================
# HELIOS++
# ======================================================================================================================

helios_dirpath = Path.home() / "Apps" / "helios-plusplus-win"
# helios_assets_dirpath: Add backslash or slash at the end, otherwise HELIOS throws
# "ERROR: folder ...\helios-plusplus-win\assetsspectra not found"
helios_assets_dirpath = str((helios_dirpath / "assets").as_posix()) + "/"
helios_output_dirpath = str(helios_dirpath / "output")
helios_platforms_filepath = str(helios_dirpath / "data" / "platforms.xml")
helios_scanners_filepath = str(helios_dirpath / "data" / "scanners_als.xml")
helios_survey_template_filepath = str(Path(__file__).parent / "helios" / "survey_template.xml")

# ======================================================================================================================
# bayes_opt BayesianOptimization
# ======================================================================================================================

bo_target_value_equality_rel_tolerance = 1e-16

# ======================================================================================================================
# 3D RECONSTRUCTION AND GEOFLOW
# ======================================================================================================================

recon_vertices_outlier_threshold = 1e2

geoflow_cmd = "geof"
geoflow_reconstruct_template_filepath = str(Path(__file__).parent / "geoflow" / "reconstruct_template.json")
geoflow_reconstruct_nested_filepath = str(Path(__file__).parent / "geoflow" / "reconstruct_nested.json")
geoflow_output_cityjson_identifier_name = "OGRLoader.identificatie"
geoflow_recon_optim_timeout_default = 300
# Names of the 3D layers for each LOD in the input GPKG (for ComplexityDifferenceEvaluator)
geoflow_input_gpkg_lod_layer_names = {
    "1.2": "lod12_3d",
    "1.3": "lod13_3d",
    "2.2": "lod22_3d"
}
# Names of the 3D layers for each LOD in the output (reconstructed) GPKG (for ComplexityEvaluator)
geoflow_output_gpkg_lod_layer_names = {
    "1.2": "LoD12_3D",
    "1.3": "LoD13_3D",
    "2.2": "LoD22_3D"
}
# Names of height attributes in output (reconstructed) building models as found in CityJSON and GPKG
geoflow_output_height_attr_names = {
    "ground": "h_ground",
    "min": "h_min",
    "50p": "h_50p",
    "70p": "h_70p",
    "max": "h_max"
}
# Names of height attributes in input building models as found in CityJSON and GPKG
geoflow_input_height_attr_names = {
    "ground": "b3_h_maaiveld",
    "min": "b3_h_dak_min",
    "50p": "b3_h_dak_50p",
    "70p": "b3_h_dak_70p",
    "max": "b3_h_dak_max"
}

# ----------------------------------------------------------------------------------------------------------------------
# Geoflow reconstruction parameters
# ----------------------------------------------------------------------------------------------------------------------

# Geoflow parameter names as exposed in the flowcharts as global variables
# - optimize certainly
gf_param_thres_alpha = "thres_alpha"
gf_param_line_epsilon = "r_line_epsilon"
gf_param_plane_epsilon = "r_plane_epsilon"
gf_param_optimization_data_term = "r_optimisation_data_term"
gf_param_thres_reg_line_dist = "thres_reg_line_dist"
gf_param_thres_reg_line_ext = "thres_reg_line_ext"
# - optimize maybe
gf_param_plane_k = "r_plane_k"
gf_param_plane_min_points = "r_plane_min_points"
# - optimize conditionally
gf_param_thres_tri_snap = "thres_tri_snap"
# - likely do not optimize
gf_param_plane_normal_angle = "r_plane_normal_angle"
gf_param_normal_estimate_k = "r_normal_k"
# - other
gf_param_skip_lod12 = "skip_lod12"
gf_param_skip_lod13 = "skip_lod13"
gf_param_skip_lod22 = "skip_lod22"

# Parameter data types
gf_integer_params = [gf_param_plane_k, gf_param_plane_min_points, gf_param_normal_estimate_k]
geoflow_parameter_types = {
    gf_param_plane_epsilon: float,
    gf_param_plane_k: int,
    gf_param_plane_min_points: int,
    gf_param_plane_normal_angle: float,  # no optim
    gf_param_thres_alpha: float,
    gf_param_line_epsilon: float,
    gf_param_thres_reg_line_dist: float,
    gf_param_thres_reg_line_ext: float,
    gf_param_optimization_data_term: float,
    gf_param_normal_estimate_k: int,  # no optim
    gf_param_thres_tri_snap: float,
    gf_param_skip_lod12: bool,
    gf_param_skip_lod13: bool,
    gf_param_skip_lod22: bool
}

# Geoflow parameter default values, as originally found in example-config.toml from geoflow/gfc-brecon/stream, according
# to which they are optimized for the AHN3 dataset at 8 ppsm.
# - Params to definitely optimize
gf_param_thres_alpha_default = 0.25
gf_param_line_epsilon_default = 0.4
gf_param_plane_epsilon_default = 0.2
gf_param_optimization_data_term_default = 7
gf_param_thres_reg_line_dist_default = 0.5
gf_param_thres_reg_line_ext_default = 1.5
gf_param_plane_k_default = 15
gf_param_plane_min_points_default = 15
# - Params to optimize if turns out to be sensible / have an impact
gf_param_thres_tri_snap_default = 0.02
# - Params likely not to optimize
gf_param_plane_normal_angle_default = 0.75
gf_param_normal_estimate_k_default = 5

# Default values for each Geoflow reconstruction parameter
geoflow_parameters_default = {
    gf_param_plane_epsilon: gf_param_plane_epsilon_default,
    gf_param_plane_k: gf_param_plane_k_default,
    gf_param_plane_min_points: gf_param_plane_min_points_default,
    gf_param_plane_normal_angle: gf_param_plane_normal_angle_default,  # no optim
    gf_param_thres_alpha: gf_param_thres_alpha_default,
    gf_param_line_epsilon: gf_param_line_epsilon_default,
    gf_param_thres_reg_line_dist: gf_param_thres_reg_line_dist_default,
    gf_param_thres_reg_line_ext: gf_param_thres_reg_line_ext_default,
    gf_param_optimization_data_term: gf_param_optimization_data_term_default,
    gf_param_normal_estimate_k: gf_param_normal_estimate_k_default  # no optim
}

# Parameter space options for the optimization of each Geoflow reconstruction parameter that should be optimized
# Need to add prefix "range_" because all config keys must be unique across all (sub-) dictionaries (at the moment)

# - Narrow ranges
geoflow_optim_parameter_space_narrow = {
    "range_" + gf_param_plane_epsilon: (0.1, 1),
    # "range_" + gf_param_plane_k: (10, 30),  # Using the value of plane_min_points for this, too
    "range_" + gf_param_plane_min_points: (10, 30),
    "range_" + gf_param_thres_alpha: (0.1, 0.5),
    "range_" + gf_param_line_epsilon: (0.1, 1),
    "range_" + gf_param_thres_reg_line_dist: (0.1, 1),
    "range_" + gf_param_thres_reg_line_ext: (1, 3),
    "range_" + gf_param_optimization_data_term: (5, 10),
}

# - Wide ranges
geoflow_optim_parameter_space_wide = {
    "range_" + gf_param_plane_epsilon: (0.01, 1.5),
    # "range_" + gf_param_plane_k: (10, 100),  # Using the value of plane_min_points for this, too
    "range_" + gf_param_plane_min_points: (10, 100),
    "range_" + gf_param_thres_alpha: (0.01, 1),
    "range_" + gf_param_line_epsilon: (0.01, 2),
    "range_" + gf_param_thres_reg_line_dist: (0.01, 2),
    "range_" + gf_param_thres_reg_line_ext: (0.1, 5),
    "range_" + gf_param_optimization_data_term: (1, 100),
}

# - Narrower ranges, adapted for scenario_014, target metric: RMS min distance, after using Hausdorff and narrow ranges
geoflow_optim_parameter_space_narrow_2 = {
    "range_" + gf_param_plane_epsilon: (0.1, 2),  # upper from 1 to 2
    # "range_" + gf_param_plane_k: (10, 30),  # Using the value of plane_min_points for this, too
    "range_" + gf_param_plane_min_points: (10, 30),
    "range_" + gf_param_thres_alpha: (0.1, 1),  # upper from .5 to 1
    "range_" + gf_param_line_epsilon: (0.1, 1),
    "range_" + gf_param_thres_reg_line_dist: (0.1, 1),
    "range_" + gf_param_thres_reg_line_ext: (1, 3),
    "range_" + gf_param_optimization_data_term: (5, 10),
}

# ======================================================================================================================
# FME
# ======================================================================================================================

fme_cmd = r"C:\Program Files\FME\fme.exe"
fme_workspace_area_volume_filepath = str(Path(__file__).parent / "fme" / "computation_area_volume.fmw")
fme_workspace_iou3d_filepath = str(Path(__file__).parent / "fme" / "computation_iou_3d.fmw")
fme_area_field_name = "fme_area"
fme_volume_field_name = "fme_volume"
fme_iou3d_volume_input_field_name = "volume_input"
fme_iou3d_volume_output_field_name = "volume_output"
fme_iou3d_volume_intersection_field_name = "volume_intersection"
fme_iou3d_iou_field_name = "IOU"

target_identifier_name = "identificatie"
