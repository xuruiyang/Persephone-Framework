import numpy as np

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def fair(trace, num_p=3, verbose=True):
	s=[]
	e=[]
	for i in range(len(trace)):
		if i%2==0:
			assert isinstance(trace[i],int)
			e.append(trace[i])
		else:
			s.append(trace[i])

	loop_start = 0
	for i in range(len(s)):
		ss = s[i]
		s0 = s[-1]
		if [ss[0]]+(ss[1:])==[s0[0]]+(s0[1:]):
			loop_start = i
			break
	cycle_set = set(e[loop_start+1:])
	if verbose:
		print(len(e),"(",len(e[loop_start+1:]),")",len(cycle_set))
	isfair=len(cycle_set) == num_p
	if isfair:
		print("Counterexample Found:")
		print(trace)
		print(len(e),"(",len(e[loop_start+1:]),")",len(cycle_set))
		exit()
	return isfair

