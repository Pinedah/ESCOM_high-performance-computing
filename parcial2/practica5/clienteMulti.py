import socket

# configuración del cliente
HOST = '224.0.0.1'  # dirección multicast
PORT = 5000        # puerto de envío

# creación del socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# permitir la reutilización de direcciones multicast
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)

# enviar mensajes a los servidores multicast
while True:
    mensaje = input("Introduzca un mensaje para enviar a los servidores: ")
    sock.sendto(mensaje.encode('utf-8'), (HOST, PORT))