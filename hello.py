import pickle
with open('avengers.pickl','rb') as stupickle:
	dic = pickle.load(stupickle)
for i,v in dic.items():
	print(v)
