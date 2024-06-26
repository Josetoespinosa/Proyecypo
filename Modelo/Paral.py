# Datos proporcionados
T_secuencial = 7490655  # microsegundos
T_paralelo = 3705801.025  # microsegundos (convertido de milisegundos)
N = 2  # Asumimos 2 núcleos disponibles en el ESP32

# Calculamos la aceleración S(N)
S_N = T_secuencial / T_paralelo

# Calculamos la fracción paralelizable P usando la fórmula reordenada de la Ley de Amdahl
P = (S_N - 1) / ((S_N / N) - 1)

print(f"Aceleración S(N): {S_N:.2f}")
print(f"Fracción paralelizable (P): {P * 100:.2f}%")
