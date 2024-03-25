from dataclasses import dataclass


@dataclass
class Simulations:
    grayscale_filenames: str
    folder_id: str


@dataclass
class Paths:
    workdir: str
    origaimdir: str
    aimdir: str
    feadir: str
    sumdir: str
    commondir: str
    folder_bc_psl_loadcases: str
    boundary_conditions: str
    odb_OF_python_script: str
    odb_python_script: str


@dataclass
class Filenames:
    filename_postfix_cort_mask: str
    filename_postfix_trab_mask: str
    filename_postfix_mask: str
    filename_postfix_bmd: str
    filename_postfix_seg: str
    filename_postfix_common: str
    filename_postfix_transform: str


@dataclass
class Version:
    verification_files: int
    current_version: str


@dataclass
class Socket:
    site: str
    abaqus: str


@dataclass
class ImgSettings:
    img_basepath: str
    meshpath: str
    outputpath: str


@dataclass
class MeshingSettings:
    aspect: int
    _slice: int
    undersampling: int
    slicing_coefficient: int
    inside_val: int
    outside_val: int
    lower_thresh: float
    upper_thresh: float
    s: int
    k: int
    interp_points: int
    thickness_tol: float
    phases: int
    center_square_length_factor: float
    mesh_order: int
    n_elms_longitudinal: int
    n_elms_transverse_trab: int
    n_elms_transverse_cort: int
    n_elms_radial: int
    ellipsoid_fitting: bool
    show_plots: bool
    show_gmsh: bool
    write_mesh: bool
    trab_refinement: bool
    mesh_analysis: bool


@dataclass
class Mesh:
    img_settings: ImgSettings
    meshing_settings: MeshingSettings


@dataclass
class Mesher:
    meshing: str
    element_size: float
    air_elements: bool


@dataclass
class ImageProcessing:
    origaim_separate: bool
    mask_separate: bool
    imtype: str
    fabric_type: str
    bvtv_scaling: int
    bvtv_slope: float
    bvtv_intercept: float
    BVTVd_as_BVTV: bool
    BVTVd_comparison: bool
    SEG_correction: bool
    BMC_conservation: bool


@dataclass
class Homogenization:
    roi_bvtv_size: int
    STL_tolerance: float
    ROI_kernel_size_cort: int
    ROI_kernel_size_trab: int
    ROI_BVTV_size_cort: float
    ROI_BVTV_size_trab: float
    isotropic_cortex: bool
    site_bone: str


@dataclass
class Loadcase:
    full_nonlinear_loadcases: bool
    BC_mode: int
    control: str
    start_step_size: float
    time_for_displacement: int
    min_step_size: float
    max_step_size: float
    load_displacement: float


@dataclass
class Abaqus:
    nlgeom: str
    abaqus_nprocs: int
    abaqus_memory: int
    delete_odb: bool
    max_increments: int
    umat: str


@dataclass
class Registration:
    registration: bool


@dataclass
class OldCfg:
    nphases: int
    ftype: str
    verification_file: int
    all_mask: bool
    adjust_element_size: bool


@dataclass
class HFEConfig:
    simulations: Simulations
    paths: Paths
    filenames: Filenames
    version: Version
    socket: Socket
    mesh: Mesh
    mesher: Mesher
    image_processing: ImageProcessing
    homogenization: Homogenization
    loadcase: Loadcase
    abaqus: Abaqus
    registration: Registration
    old_cfg: OldCfg
