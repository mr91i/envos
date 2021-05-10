#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import envos

def main():
    config = envos.Config(
        run_dir="./run",
        n_thread=1,
        rau_in=10,
        rau_out=1000,
        dr_to_r=0.1,
        aspect_ratio=1,
        CR_au=100,
        Ms_Msun=0.1,
        T=10,
        cavangle_deg=45,
        f_dg=0.01,
        opac="MRN20",
        Lstar_Lsun=1.0,
        molname="c18o",
        molabun=1e-20,
        iline=3,
        size_au=2000,
        pixsize_au=25,
        vfw_kms=6,
        dv_kms=0.01,
        beam_maj_au=50,
        beam_min_au=50,
        vreso_kms=0.2,
        beam_pa_deg=0,
        convmode="scipy",
        incl=90,
        posang=0,
        dpc=100,
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
    odat = osim.observe_line()
    
    # Once you save a data file as:
    #   odat.save_instance(filename="lineobs.pkl")
    # you can read the data as:
    #   odat = envos.read_obsdata("run/lineobs.pkl")

    
    PV = odat.get_PV_map(pangle_deg=0)
    print(PV)
    
    # To visualize the PV diagram, one can use "envos.plot_tools"
    # which provide functions to plot results.
    envos.plot_tools.plot_pvdiagram(PV)

if __name__=="__main__":
    main()
