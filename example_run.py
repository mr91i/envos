#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import envos

def main():
    # Config class is used to set parameters for the calculation.
    # The simplest way to set the parameters is to pass a Config class
    # to ModelGenerator or ObsSimulator.
    # The following is an example of setting parameters for the calculation.
    config = envos.Config(
        run_dir="./run", # directory to save the results
        n_thread=1, # number of threads used in OpenMP parallelization
        rau_in=10, # inner radius of the computational domain 
        rau_out=1000, # outer radius of the computational domain
        dr_to_r=0.1, # dr/r
        aspect_ratio=1, # rdtheta/dr
        nphi=100, # number of grid points in phi direction 
        CR_au=100, # centrifugal radius 
        Ms_Msun=0.3, # mass of the central star
        T=10, # temperature of the cloud core
        cavangle_deg=45, # opening angle of the cavity
        f_dg=0.01, # dust-to-gas mass ratio
        opac="MRN20", # dust opacity model 
        Lstar_Lsun=1.0, # luminosity of the central star
        molname="c3h2", # name of the molecule, which is used in the radmc-3d data file
        molabun=1e-20, # abundance of the molecule
        iline=69, # line index of the molecule in the radmc-3d data file
        size_au=2000, # size of the image
        pixsize_au=25, # pixel size of the image
        vfw_kms=6, # velocity range 
        dv_kms=0.01,    # pixel size of the velocity axis 
        beam_maj_au=50, # major axis of the elliptical beam
        beam_min_au=50, # minor axis of the elliptical beam
        vreso_kms=0.2,  # velocity resolution of convolution 
        beam_pa_deg=0,  # position angle of the major axis of the elliptical beam
        convmode="scipy", 
        incl=90, # angle between the disk rotation axis and the direction to the observer
        posang=0, # position angle of the disk rotation axis
        dpc=100, # distance to the source in pc
    )

    # Printing a Config class shows all arguments set in the instance,
    # including implicitly set arguments.
    print(config)

    # ModelGenerator generates a physical model with calculating
    # density, velocity, and temperature structure. The simplest way
    # to pass the argumets to ModelGenerator is just to pass a config class.
    mg = envos.ModelGenerator(config)

    # Calculate density and velocity structure.
    mg.calc_kinematic_structure()

    # Calculate temperature structure.
    mg.calc_thermal_structure()

    # Generate the physical model as a `Circumstellar Model` object.
    model = mg.get_model()

    # Once you save a data file as:
    #   model.save_pickle("model.pkl")
    # you can read the model as
    #   model = envos.read_model("run/model.pkl")

    # "Circumstellar Model" object is printed as a readable format.
    # For n-dimension array, to be readable, only shape, min, and max of the array are printed.
    # Learn what variables are available, and check if the calculation is done collectly.
    print(model)

    # To visualize the physical model, one can use "envos.plot_tools"
    # which provide functions to plot results. 
    envos.plot_tools.plot_density_map(model, streams=True)
    envos.plot_tools.plot_temperature_map(model, streams=True)

    # ObsSimulator executes the calculation of synthetic observation.
    # ObsSimulator can also get recieve a `Config` object.
    osim = envos.ObsSimulator(config)

    # ObsSimulator set radmc-3d input files again when it recieves a physical model,
    # unless input files used in calculaitng the thermal structure is used.
    # For example, if you want to understand which region corresponds to which intensity feature,
    # limiting emitting regions after calculating the temperture structure may help you.
    osim.set_model(model)

    # observe_line() calculates the radiative transfer with ray tracing method,
    # and returns a ObsData3D object, which contains intensity distribution in
    # 3-dimension (longitude, latitude, and velocity; i.e., 3D data cube).
    cube = osim.observe_line()

    # Once you save a data file as:
    #   cube.save_instance(filename="lineobs.pkl")
    # you can read the data as:
    #   cube = envos.read_obsdata("run/lineobs.pkl")

    # To get a moment 0 map, one can use get_mom0_map() method.
    # Image class contains 2D intensity array and the axes 
    img = cube.get_mom0_map()

    # Let's see what is contained in the ObsData3D object.
    print(img)

    # To visualize the moment 0 map, one can use "envos.plot_tools".
    # But all infomation is contained in ObsData3D object, i.e., `img`,
    # so you can access the data and use any plotting tools you like.
    envos.plot_tools.plot_mom0_map(
        img,
        arrow_angle_deg=90,
        norm="max"
    )

    # To get a PV diagram, one can use get_pv_map() method.
    # The following example shows how to get a PV diagram along the major axis.
    pv = cube.get_pv_map(pangle_deg=90)

    print(pv)

    # To visualize the PV diagram, one can use "envos.plot_tools"
    # which provide functions to plot results.
    envos.plot_tools.plot_pvdiagram(pv)


if __name__ == "__main__":
    main()
