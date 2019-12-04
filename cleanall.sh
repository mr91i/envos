rm -f log
echo Removed ./log

rm -rf calc/*.pyc calc/__pycache__
echo "Cleaned ./calc"

rm -rf calc/radmc/{\
*.{o,mod,pyc,dat,out,info,used,uout,udat},\
amr_grid.inp,gas_temperature.inp,wavelength_micron.inp,\
dustopac.inp,dust_density.inp,stars.inp,radmc3d.inp,*wavelength_micron.inp\
proc*,microturbulence.inp,gas_velocity.inp,numberdens_*.inp,lines.inp\
Makefile~,*.pro~,README*~,*.f90~,*.inp~,*.py~,*.pkl,\
proc*,microturbulence.inp,gas_velocity.inp,numberdens_*.inp,lines.inp}
#echo radmc Directory cleaned to basic
echo "Cleaned ./calc/radmc"

rm -rf calc/sobs/*.fits
echo "Cleaned ./calc/sobs"

rm -rf fig/*
echo "Cleaned ./fig"


ps 
echo "May I kill \"radmc3d\" ? : OK[Enter]"
read 
killall radmc3d

ps 
echo "May I kill \"python\" ? : OK[Enter]"
read 
killall python python2 python3 

