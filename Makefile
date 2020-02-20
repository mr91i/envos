#BASE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
#echo "Base directory is "$(BASE_DIR)
#BASE_DIR2 := $(shell cd $(shell dirname $0) && shell pwd)
BASE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))#$(abspath $(lastword $(MAKEFILE_LIST)))
RADMC_DIR := $(BASE_DIR)"/radmc"
SRC_DIR := $(SRC_DIR)"/pmodes"
RM = rm -rfv


default:
	pmodes/mkmodel.py
	pmodes/radmcset.py
	pmodes/sobs.py
	pmodes/analyze2.py


## Execut exec.py without necessary of connecting the server
nohup:
	nohup make &

clean: kill_process
	#clean_cache: 
	$(RM) $(SRC_DIR){*.pyc,__pycache__}
	#clean_subproc:
	$(RM) -d $(RADMC_DIR)/proc*
	#clean_radmcsetting:
	find $(RADMC_DIR) -maxdepth 1 -type f -name "*.inp" -name "*.out" ! -path "*molecule_*.inp" ! -path "*dustkappa_*.inp" -exec $(RM) {} \;	
	#clean_fits:
	$(RM) $(RADMC_DIR)/sobs.fits
	#clean_pkl:
	$(RM) $(RADMC_DIR)/*.pkl

kill_process:
	echo -e "\nMay I kill \"radmc3d\" ? : OK[Enter]"
	ps
	read 
	-killall radmc3d
	echo -e "\nMay I kill \"python\" ? : OK[Enter]"
	ps
	read 
	-killall python python2 python3 

