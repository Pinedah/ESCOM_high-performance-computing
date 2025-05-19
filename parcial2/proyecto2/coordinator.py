#!/usr/bin/env python3
import socket
import threading
import time
import os

# Configuración del coordinador
COORDINATOR_HOST = '0.0.0.0'  # Escuchar en todas las interfaces
COORDINATOR_PORT = 6000

# Configuración de servidores
SERVER_PORTS = [6001, 6002, 6003]  # Puertos para múltiples servidores
SERVER_HOSTS = []  # Se configurarán dinámicamente

# Para el seguimiento de conexiones
client_connections = []
server_connections = []
server_lock = threading.Lock()

def clear_screen():
    os.system('clear')

def handle_client(client_socket, client_address):
    """Manejar conexiones de clientes"""
    global server_connections
    
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
            
            # Reenviar mensaje a TODOS los servidores disponibles (broadcast)
            with server_lock:
                if server_connections:
                    active_servers = []
                    failed_servers = 0
                    
                    # Intentar enviar a cada servidor conectado
                    for idx, server_conn in enumerate(server_connections):
                        try:
                            server_conn.sendall(data)
                            print(f"[>] Mensaje reenviado al servidor #{idx}")
                            active_servers.append(idx)
                        except:
                            print(f"[-] Error al enviar al servidor #{idx}")
                            failed_servers += 1
                    
                    if active_servers:
                        print(f"[>] Mensaje distribuido a {len(active_servers)} servidores: {active_servers}")
                    
                    if failed_servers > 0:
                        print(f"[-] No se pudo enviar a {failed_servers} servidores")
                else:
                    print(f"[-] No hay servidores disponibles para reenviar el mensaje")
    
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

def listen_for_servers():
    """Escuchar conexiones de servidores"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Puerto especial para escuchar servidores
    SERVER_REGISTER_PORT = 6099
    
    try:
        server_socket.bind((COORDINATOR_HOST, SERVER_REGISTER_PORT))
        server_socket.listen(10)
        print(f"[+] Coordinador escuchando servidores en {COORDINATOR_HOST}:{SERVER_REGISTER_PORT}")
        
        while True:
            conn, addr = server_socket.accept()
            
            # Recibir información de identificación del servidor
            try:
                data = conn.recv(1024).decode('utf-8')
                if data.startswith("SERVER:"):
                    server_id = data.split(":", 1)[1]
                    print(f"[+] Nuevo servidor registrado ID:{server_id} desde {addr[0]}:{addr[1]}")
                    
                    with server_lock:
                        server_connections.append(conn)
                    
                    # Confirmar registro al servidor
                    conn.sendall(f"REGISTERED:{len(server_connections)}".encode('utf-8'))
                    
                    # Iniciar hilo para monitorear estado del servidor
                    server_monitor = threading.Thread(target=monitor_server, args=(conn, addr, server_id))
                    server_monitor.daemon = True
                    server_monitor.start()
                else:
                    print(f"[!] Conexión rechazada de {addr[0]}:{addr[1]} - No es un servidor")
                    conn.close()
            except Exception as e:
                print(f"[!] Error al registrar servidor desde {addr[0]}:{addr[1]}: {e}")
                conn.close()
            
    except KeyboardInterrupt:
        print("\n[!] Deteniendo escucha de servidores...")
    except Exception as e:
        print(f"[!] Error en escucha de servidores: {e}")
    finally:
        server_socket.close()

def monitor_server(conn, addr, server_id):
    """Monitorear la conexión con un servidor"""
    try:
        while True:
            # Verificar estado del servidor
            try:
                conn.sendall("PING".encode('utf-8'))
                response = conn.recv(1024)
                if not response:
                    raise Exception("No response")
                
                # Si recibimos algún mensaje que no es PONG, manejarlo
                if response.decode('utf-8') != "PONG":
                    print(f"[>] Mensaje del servidor ID:{server_id}: {response.decode('utf-8')}")
                
                time.sleep(5)
                
            except Exception as e:
                print(f"[-] Servidor ID:{server_id} desconectado: {e}")
                with server_lock:
                    if conn in server_connections:
                        server_connections.remove(conn)
                conn.close()
                break
                
    except Exception as e:
        print(f"[!] Error al monitorear servidor ID:{server_id}: {e}")
        with server_lock:
            if conn in server_connections:
                server_connections.remove(conn)
        conn.close()

def main():
    # Thread para escuchar conexiones de clientes
    client_thread = threading.Thread(target=listen_for_clients)
    client_thread.daemon = True
    client_thread.start()
    
    # Thread para escuchar conexiones de servidores
    server_listen_thread = threading.Thread(target=listen_for_servers)
    server_listen_thread.daemon = True
    server_listen_thread.start()
    
    try:
        # Información sobre el estado del sistema
        while True:
            time.sleep(20)
            with server_lock:
                print(f"\n[INFO] Estado del sistema:")
                print(f"  - Clientes conectados: {len(client_connections)}")
                print(f"  - Servidores conectados: {len(server_connections)}")
                
    except KeyboardInterrupt:
        print("\n[!] Cerrando coordinador...")
        
        # Cerrar todas las conexiones
        with server_lock:
            for conn in server_connections:
                try:
                    conn.close()
                except:
                    pass
                
        for conn in client_connections:
            try:
                conn.close()
            except:
                pass

if __name__ == "__main__":
    clear_screen()
    print("=== COORDINADOR DE MENSAJERÍA BROADCAST ===")
    main()