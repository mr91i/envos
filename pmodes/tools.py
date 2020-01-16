

def exe(cmd):
	try:
		if args.dryrun:
			print(cmd)
			return 0
		subprocess.check_call(r'echo `date "+%Y/%m/%d-%H:%M:%S"` "		" "{}" >> .executed_cmd.txt'.format(cmd), shell=True )
#		subprocess.check_call('echo `date \"+\%Y/\%m/\%d-\%H:\%M:\%S\" ` \'%s\' >> .executed_cmd.txt'%cmd, shell=True )
		print("Execute: {}".format(cmd))
		if not args.debug:
			retcode = subprocess.check_call( cmd, shell=True )
		return 0
		#retcode = subprocess.check_call( cmd.split() )
	except subprocess.CalledProcessError as e:
		print('Error generated:')
		print(' - return code is %s'%e.returncode)
		print(' - cmd is \'%s\''%e.cmd)
		print(' - output is %s'%e.output)
		#exit()
		raise Exception()
