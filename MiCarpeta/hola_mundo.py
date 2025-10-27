print("¡Hola Mundo!")

def pedir_numero(mensaje):
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("Entrada inválida. Introduce un número válido.")

# Captura dos números y calcula la suma
a = pedir_numero("Introduce el primer número: ")
b = pedir_numero("Introduce el segundo número: ")
suma = a + b
print(f"{a} + {b} = {suma}")
