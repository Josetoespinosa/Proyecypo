import matplotlib.pyplot as plt
import numpy as np

def amdal(p, n):
    return 1 / ((1 - p) + (p / n))

#p =  # Conv2D, MaxPool2D, FullyConnected
p = 0.957549

cores = np.arange(1, 11)

speedup = [amdal(p, n) for n in cores]

speedup_diff = np.diff(speedup)

# Encontrar el índice de la mayor diferencia
max_diff_index = np.argmax(speedup_diff)

# El número de cores correspondiente a la mayor diferencia es el índice + 1
# (porque np.diff() devuelve una lista que es más corta por un elemento)
max_diff_cores = cores[max_diff_index + 1]
print(f'El mayor salto de aceleración se obtiene con {max_diff_cores} cores.')

# plt.figure(figsize=(10, 6))
plt.plot(cores, speedup, marker='o', linestyle='-', color='b', label='Amdahl')
plt.scatter(max_diff_cores, speedup[max_diff_index + 1], color='r', s=100, label='Mayor salto')
plt.xlabel('Cores')
plt.ylabel('Speedup')
plt.title('Cores vs Speedup')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()