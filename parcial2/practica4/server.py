import socket
# configuración del servidor
HOST = '127.0.0.1'  # localhost
PORT = 6666        # puerto de escucha

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