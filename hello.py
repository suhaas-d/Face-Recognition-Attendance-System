import pickle
with open('avengers.pickle','rb') as stupickle:
	dic = pickle.load(stupickle)
for i,v in dic.items():
	print(i)
	print(v)
