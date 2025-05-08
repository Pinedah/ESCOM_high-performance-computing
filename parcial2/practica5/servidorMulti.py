import socket

# configuración del servidor
HOST = '224.0.0.1'  # dirección multicast
PORT = 5000        # puerto de escucha

# creación del socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

# permitir la reutilización de direcciones multicast
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

# unirse al grupo multicast
sock.bind((HOST, PORT))
mreq = socket.inet_aton(HOST) + socket.inet_aton('0.0.0.0')
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

# esperar mensajes de los clientes
while True:
    data, addr = sock.recvfrom(1024)
    print(f"Mensaje recibido del cliente: {data.decode('utf-8')}")
