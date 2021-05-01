import pickle
with open('student_name_encodings.pkl','rb') as stupickle:
	dic = pickle.load(stupickle)
print(dic)
