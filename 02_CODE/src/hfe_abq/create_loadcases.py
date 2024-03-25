def create_loadcase_fz_max(cfg, sample: str, loadcase: str) -> None:
    """
    create BC file for force MAX loadcase RADIUS Hosseini
    - 2 in-plane DOF fixed
    Force MAX loadcases boundary conditions are displacements of respective linear load cases, scaled by a factor lambda
    to ensure failure. Lambda is computed so that the max displacement in the repective direction is equal to
    config['fz_max_factor'].
    @rtype: None
    """

    # read BC file to BC optim
    bc_file = open(
        cfg.paths.folder_bc_psl_loadcases + "boundary_conditions_disp_x.inp", "r"
    )
    bc_fmax_file_pwd = (
        cfg.paths.aimdir
        + cfg.simulations.folder_id[sample]
        + "/"
        + "boundary_conditions_"
        + loadcase
        + ".inp"
    )
    bc_fmax_file = open(bc_fmax_file_pwd, "w")

    # BC_mode NUMBERS:  0: all DOF fixed / 2: two in plane fixed / 5: all DOF free
    if cfg.loadcase.BC_mode == 0:
        for line in bc_file:
            if "REF_NODE, 1, 1," in line:
                bc_fmax_file.write("REF_NODE, 1, 1, 0.0" + "\n")
            elif "REF_NODE, 2, 2," in line:
                bc_fmax_file.write("REF_NODE, 2, 2, 0.0" + "\n")
            elif "REF_NODE, 3, 3," in line:
                bc_fmax_file.write(
                    "REF_NODE, 3, 3, " + str(cfg.optimization.fz_max_factor) + "\n"
                )
            elif "REF_NODE, 4, 4," in line:
                bc_fmax_file.write("REF_NODE, 4, 4, 0.0" + "\n")
            elif "REF_NODE, 5, 5," in line:
                bc_fmax_file.write("REF_NODE, 5, 5, 0.0" + "\n")
            elif "REF_NODE, 6, 6," in line:
                bc_fmax_file.write("REF_NODE, 6, 6, 0.0" + "\n")
            else:
                bc_fmax_file.write(line)
    elif cfg.loadcase.BC_mode == 2:
        for line in bc_file:
            if "REF_NODE, 1, 1," in line:
                bc_fmax_file.write("REF_NODE, 1, 1, 0.0" + "\n")
            elif "REF_NODE, 2, 2," in line:
                bc_fmax_file.write("REF_NODE, 2, 2, 0.0" + "\n")
            elif "REF_NODE, 3, 3," in line:
                bc_fmax_file.write(
                    "REF_NODE, 3, 3, " + str(cfg.optimization.fz_max_factor) + "\n"
                )
            elif "REF_NODE, 4, 4," in line:
                bc_fmax_file.write("")
            elif "REF_NODE, 5, 5," in line:
                bc_fmax_file.write("")
            elif "REF_NODE, 6, 6," in line:
                bc_fmax_file.write("")
            else:
                bc_fmax_file.write(line)

    elif cfg.loadcase.BC_mode == 5:
        for line in bc_file:
            if "REF_NODE, 1, 1," in line:
                bc_fmax_file.write("")
            elif "REF_NODE, 2, 2," in line:
                bc_fmax_file.write("")
            elif "REF_NODE, 3, 3," in line:
                bc_fmax_file.write(
                    "REF_NODE, 3, 3, " + str(cfg.optimization.fz_max_factor) + "\n"
                )
            elif "REF_NODE, 4, 4," in line:
                bc_fmax_file.write("")
            elif "REF_NODE, 5, 5," in line:
                bc_fmax_file.write("")
            elif "REF_NODE, 6, 6," in line:
                bc_fmax_file.write("")
            else:
                bc_fmax_file.write(line)

    else:
        raise ValueError(
            "BC_mode was not properly defined. Was "
            + str(cfg.loadcase.BC_mode)
            + ", but should be [0, 2, 5]"
        )

    bc_file.close()
    bc_fmax_file.close()

    # create optim input file
    inp_file_fx_pwd = (
        cfg.paths.feadir
        + cfg.simulations.folder_id[sample]
        + "/"
        + sample
        + "_V_"
        + cfg.version.current_version
        + "_FX.inp"
    )
    inp_file_fx = open(inp_file_fx_pwd, "r")
    inp_fzmax_file_pwd = (
        cfg.paths.feadir
        + cfg.simulations.folder_id[sample]
        + "/"
        + sample
        + "_V_"
        + cfg.version.current_version
        + "_"
        + loadcase
        + ".inp"
    )
    inp_fzmax_file = open(inp_fzmax_file_pwd, "w")

    # Include the OPT_MAX boundary condition file into inputfile and change NLGEOM flag to YES for nonlinear geometry
    for line in inp_file_fx:
        if "*INCLUDE, input=" in line:
            inp_fzmax_file.write("*INCLUDE, input=" + bc_fmax_file_pwd + "\n")
        elif "NLGEOM=" in line:
            inp_fzmax_file.write(
                "*STEP,AMPLITUDE=RAMP,UNSYMM=YES,INC="
                + str(cfg.abaqus.max_increments)
                + ",NLGEOM=YES\n"
            )
        else:
            inp_fzmax_file.write(line)

    inp_fzmax_file.close()
    inp_file_fx.close()
