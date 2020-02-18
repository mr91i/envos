
default:
	pmodes/mkmodel.py
	pmodes/radmcset.py
	pmodes/sobs.py
	pmodes/analyxe.py


## Execut exec.py without necessar ofy connecting the server
nohup:
	nohup make &
#	nohup pmodes/mkmodel.py  >> log & 
#	nohup pmodes/radmcset.py  >> log & 
#	nohup pmodes/sobs.py  >> log & 

clean:
	./cleanall.sh
