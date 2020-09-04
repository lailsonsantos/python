import pandas
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

#definindo os nomes de cada coluna   
names = ['num-pregnant', 'glucose', 'diastolic', 'triceps-skin', 'insulin', 'body-mass', 'diabetes-pedigree', 'age', 'class']

#Fazendo o carregamento dos dados diretamente do UCI Machine Learning          
dataset = pandas.read_csv("pima-indians-diabetes.csv", names=names)

print("Primeiros dados")
print(dataset.head(5))

#Convertendo dados categoricos para dados numericos
le = LabelEncoder()
for column_name in dataset.columns:
    if dataset[column_name].dtype == object:
        dataset[column_name] = pandas.Categorical(dataset[column_name]).codes()
        #dataset[column_name] = le.fit_transform(dataset[column_name])
    else:
        pass

#divisao de dados atributos e classe
X = dataset.values[:, 0:7]
Y = dataset.values[:,8]

#usando o metodo para fazer uma unica divisao dos dados
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)

#criando diferentes RNAs
mlp1 = MLPClassifier(hidden_layer_sizes=([10]))
mlp2 = MLPClassifier(hidden_layer_sizes=(3,3),random_state=1)
mlp3 = MLPClassifier(hidden_layer_sizes=([5,5,2]))

mlp1=mlp1.fit(X_train,y_train)
mlp2=mlp2.fit(X_train,y_train)
mlp3=mlp3.fit(X_train,y_train)

print("Acuracia de trainamento MLP1: %0.3f" %  mlp1.score(X_train, y_train))
print("Acuracia de teste MLP1: %0.3f" %  mlp1.score(X_test, y_test))

print("Acuracia de trainamento MLP2: %0.3f" %  mlp2.score(X_train, y_train))
print("Acuracia de teste MLP2: %0.3f" %  mlp2.score(X_test, y_test))

print("Acuracia de trainamento MLP3 clf: %0.3f" %  mlp3.score(X_train, y_train))
print("Acuracia de teste MLP3: %0.3f" %  mlp3.score(X_test, y_test))

