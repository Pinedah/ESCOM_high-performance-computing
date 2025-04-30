import threading

import socket
# configuración del servidor
HOST = '127.0.0.1'  # localhost
PORT = 6666        # puerto de escucha

"""
# creación del socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()  # poner el socket en modo escucha
    print('Servidor escuchando en', (HOST, PORT))
    conn, addr = s.accept()  # esperar una conexión
    with conn:
        print('Conectado por', addr)
        while True:
            data = conn.recv(1024)  # recibir datos del cliente
            if not data:
                break
            # conn.sendall(data)  # enviar los mismos datos de vuelta al cliente
            print(f"Mensaje recibido del cliente: {data.decode('utf-8')}")
"""

# función que maneja las conexiones de los distintos clientes
def manejo_cliente(conx, dir):
    print('Conectado mediante', dir)
    while True:
        data = conx.recv(1024)  # recibir datos del cliente
        if not data:
            break
        print(f"Mensaje recibido del cliente {dir}: {data.decode('utf-8')}")
        conx.sendall(data)  # enviar los mismos datos de vuelta al cliente
    conx.close()

# creación del socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()  # poner el socket en modo escucha
    print('Servidor escuchando en', (HOST, PORT))
    while True:
        conn, addr = s.accept()  # esperar una conexión
        thread = threading.Thread(target=manejo_cliente, args=(conn, addr))
        thread.start()