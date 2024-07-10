import numpy as np
import matplotlib.pyplot as plt

# Sigmoidalna funkcja aktywacji i jej pochodna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dane wejściowe (X) i oczekiwane wyjścia (y)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicjalizacja wag
np.random.seed(1)
weights_input_hidden = np.random.uniform(-1, 1, (2, 2))
weights_hidden_output = np.random.uniform(-1, 1, (2, 1))
bias_hidden = np.random.uniform(-1, 1, (1, 2))
bias_output = np.random.uniform(-1, 1, (1, 1))

# Współczynnik uczenia
learning_rate = 0.1

# Ilość epok
epochs = 10000

# Listy do przechowywania historii błędów, wag i dokładności
mse_history = []
classification_error_history = []
accuracy_history = []
average_weights_input_hidden_history = []
average_weights_hidden_output_history = []

#UPGRADE
mse_history_input_hidden = []



# Trening
for i in range(epochs):
    # Propagacja w przód
    hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    # Błąd
    error = y - predicted_output
    mse = np.mean(np.square(error))
    mse_history.append(mse)

    # Błąd klasyfikacji
    predictions = (predicted_output > 0.5).astype(int)
    classification_error = np.mean(np.abs(predictions - y))
    classification_error_history.append(classification_error)

    # Dokładność
    accuracy = np.mean(predictions == y)
    accuracy_history.append(accuracy)

    # Propagacja wsteczna
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    #UPGRADE - mse warstwa ukryta
    mse_hidden = np.mean(np.square(error_hidden_layer))
    mse_history_input_hidden.append(mse_hidden)

    # Aktualizacja wag i biasów
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Zapisywanie średnich wartości wag
    average_weights_input_hidden_history.append(np.mean(weights_input_hidden))
    average_weights_hidden_output_history.append(np.mean(weights_hidden_output))

# Testowanie modelu
print("Wynik po uczeniu:")
print(predicted_output)

# Wykres błędu MSE
plt.figure(figsize=(10, 5))
plt.plot(mse_history, label='MSE')
plt.title('Błąd MSE w czasie')
plt.xlabel('Epoki')
plt.ylabel('MSE')
plt.legend()
plt.show()

# UPGRADE Wykres błędu MSE/ukryty
plt.figure(figsize=(10, 5))
plt.plot(mse_history_input_hidden, label='MSE hidden')
plt.title('Błąd MSE w czasie')
plt.xlabel('Epoki')
plt.ylabel('MSE')
plt.legend()
plt.show()


# Wykres błędu klasyfikacji
plt.figure(figsize=(10, 5))
plt.plot(classification_error_history, label='Błąd klasyfikacji')
plt.title('Błąd klasyfikacji w czasie')
plt.xlabel('Epoki')
plt.ylabel('Błąd klasyfikacji')
plt.legend()
plt.show()

### Tego raczej już nie musimy robić
## Wykres dokładności
#plt.figure(figsize=(10, 5))
#plt.plot(accuracy_history, label='Dokładność')
#plt.title('Dokładność w czasie')
#plt.xlabel('Epoki')
#plt.ylabel('Dokładność')
#plt.legend()
#plt.show()

# Wykresy średnich wartości wag
plt.figure(figsize=(10, 5))
plt.plot(average_weights_input_hidden_history, label='Średnia wartość wag - warstwa wejściowa do ukrytej')
plt.plot(average_weights_hidden_output_history, label='Średnia wartość wag - warstwa ukryta do wyjściowej')
plt.title('Zmiany średniej wartości wag w czasie')
plt.xlabel('Epoki')
plt.ylabel('Średnia wartość wag')
plt.legend()
plt.show()

#Upgrade

new_data = np.random.rand(100, 2)

# Propagacja w przód dla nowych danych
new_hidden_layer_activation = np.dot(new_data, weights_input_hidden) + bias_hidden
new_hidden_layer_output = sigmoid(new_hidden_layer_activation)

new_output_layer_activation = np.dot(new_hidden_layer_output, weights_hidden_output) + bias_output
new_predicted_output = sigmoid(new_output_layer_activation)

# Wykres wyników dla nowych danych
plt.scatter(new_data[:,0], new_data[:,1], c=new_predicted_output.flatten(), cmap='coolwarm')
plt.title('Wyniki dla nowych danych')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.colorbar(label='Przewidywane wyjście')
plt.show()

input("Nacisnij Enter aby zakonczyc: ")
