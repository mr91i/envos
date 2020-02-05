#rm -f log
#echo Removed ./log
dpath_home=$(cd $(dirname $0) && pwd)
echo "Home directory is "$dpath_home

rm -rf $dpath_home"/pmodes/"{*.pyc,__pycache__}
echo "Cleaned cache in "$dpath_home"/pmodes"

# rm -rf  $dpath_home"/radmc/"{\
# *.{o,mod,pyc,dat,out,info,used,uout,udat},\
# amr_grid.inp,gas_temperature.inp,wavelength_micron.inp,\
# dustopac.inp,dust_density.inp,stars.inp,radmc3d.inp,*wavelength_micron.inp\
# proc*,microturbulence.inp,gas_velocity.inp,numberdens_*.inp,lines.inp\
# Makefile~,*.pro~,README*~,*.f90~,*.inp~,*.py~,\
# proc*,microturbulence.inp,gas_velocity.inp,numberdens_*.inp,lines.inp}
# #echo radmc Directory cleaned to basic
# echo "Cleaned something in "$dpath_home"/radmc"
shopt -s extglob
rm -rfv $dpath_home"/radmc/"!(dustkappa_*.inp|molecule_*.inp)
#echo $?
#echo "Cleaned all files in "$dpath_home"/radmc except dustkappa_*.inp and molecule_*.inp"
shopt -u extglob

#rm -rf  $dpath_home"/radmc/"*.pkl
#echo "Cleaned pkl in "$dpath_home"/radmc"

#rm -rf  $dpath_home"/radmc/"*.pkl
#echo "Cleaned pkl in "$dpath_home"/radmc"

#rm -rf  $dpath_home"/radmc/"*.fits
#echo "Cleaned fits in "$dpath_home"/radmc"

#rm -rf $dpath_home"/fig/"*
#echo "Cleaned all in "$dpath_home"/fig"
 
echo -e "\nMay I kill \"radmc3d\" ? : OK[Enter]"
ps
read 
killall radmc3d

 
echo -e "\nMay I kill \"python\" ? : OK[Enter]"
ps
read 
killall python python2 python3 

