#!/usr/bin/env python3
import socket
import threading
import time
import os

# Configuración del coordinador
COORDINATOR_HOST = '0.0.0.0'  # Escuchar en todas las interfaces
COORDINATOR_PORT = 6000

# Configuración del servidor
SERVER_HOST = ''  # Cambia a la IP del servidor
SERVER_PORT = 6001

# Para el seguimiento de conexiones
client_connections = []
server_connection = None
server_lock = threading.Lock()

def clear_screen():
    os.system('clear')

def handle_client(client_socket, client_address):
    """Manejar conexiones de clientes"""
    global server_connection
    
    print(f"[+] Nueva conexión de cliente: {client_address[0]}:{client_address[1]}")
    
    try:
        while True:
            # Recibir mensaje del cliente
            data = client_socket.recv(1024)
            if not data:
                break
                
            mensaje = data.decode('utf-8')
            print(f"[>] Mensaje recibido de {client_address[0]}: {mensaje}")
            
            # Confirmar recepción al cliente
            client_socket.sendall("Mensaje recibido por el coordinador".encode('utf-8'))
            
            # Reenviar mensaje al servidor si está disponible
            with server_lock:
                if server_connection:
                    try:
                        server_connection.sendall(data)
                        print(f"[>] Mensaje reenviado al servidor")
                    except:
                        print(f"[-] Error al enviar mensaje al servidor")
                        server_connection = None
                else:
                    print(f"[-] No hay servidor disponible para reenviar el mensaje")
    
    except Exception as e:
        print(f"[!] Error con cliente {client_address}: {e}")
    
    finally:
        print(f"[-] Cliente desconectado: {client_address[0]}:{client_address[1]}")
        if client_socket in client_connections:
            client_connections.remove(client_socket)
        client_socket.close()

def listen_for_clients():
    """Escuchar conexiones de clientes"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        client_socket.bind((COORDINATOR_HOST, COORDINATOR_PORT))
        client_socket.listen(5)
        print(f"[+] Coordinador escuchando clientes en {COORDINATOR_HOST}:{COORDINATOR_PORT}")
        
        while True:
            conn, addr = client_socket.accept()
            client_connections.append(conn)
            
            # Iniciar un hilo para manejar este cliente
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\n[!] Deteniendo escucha de clientes...")
    except Exception as e:
        print(f"[!] Error en escucha de clientes: {e}")
    finally:
        client_socket.close()

def connect_to_server():
    """Conectar con el servidor"""
    global server_connection
    
    while True:
        if server_connection is None:
            try:
                # Intentar conectar con el servidor
                new_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_connection.connect((SERVER_HOST, SERVER_PORT))
                
                with server_lock:
                    server_connection = new_connection
                    
                print(f"[+] Conectado al servidor en {SERVER_HOST}:{SERVER_PORT}")
                
                # Esperar a que se pierda la conexión
                while True:
                    try:
                        # Enviar ping para verificar conexión
                        server_connection.sendall("PING".encode('utf-8'))
                        response = server_connection.recv(1024)
                        if not response:
                            raise Exception("No response from server")
                        time.sleep(5)
                    except:
                        print("[-] Conexión perdida con el servidor")
                        with server_lock:
                            if server_connection:
                                server_connection.close()
                            server_connection = None
                        break
                        
            except ConnectionRefusedError:
                print(f"[-] El servidor en {SERVER_HOST}:{SERVER_PORT} no está disponible. Reintentando...")
                time.sleep(5)
            except Exception as e:
                print(f"[!] Error al conectar con el servidor: {e}")
                time.sleep(5)
        else:
            time.sleep(1)

def main():
    # Thread para escuchar conexiones de clientes
    client_thread = threading.Thread(target=listen_for_clients)
    client_thread.daemon = True
    client_thread.start()
    
    # Thread para mantener conexión con el servidor
    server_thread = threading.Thread(target=connect_to_server)
    server_thread.daemon = True
    server_thread.start()
    
    try:
        # Mantener el programa principal en ejecución
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Cerrando coordinador...")
        
        # Cerrar todas las conexiones
        with server_lock:
            if server_connection:
                server_connection.close()
                
        for conn in client_connections:
            conn.close()

if __name__ == "__main__":
    clear_screen()
    print("=== COORDINADOR DE MENSAJERÍA ===")
    main()