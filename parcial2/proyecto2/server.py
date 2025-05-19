#!/usr/bin/env python3
import socket
import threading
import os
import time
import datetime
import uuid

# Configuración del servidor
SERVER_HOST = '0.0.0.0'  # Escuchar en todas las interfaces
SERVER_PORT = 6001  # Puerto para recibir mensajes

# Configuración para conectar al coordinador
COORDINATOR_HOST = '192.168.1.100'  # Cambia a la IP del coordinador
COORDINATOR_REGISTER_PORT = 6099  # Puerto para registro de servidores

# Para guardar los mensajes
log_file = "mensajes.log"
server_id = str(uuid.uuid4())[:8]  # ID único para este servidor

def clear_screen():
    os.system('clear')

def save_message(message):
    """Guardar mensaje en archivo de log"""
    try:
        with open(log_file, "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"[-] Error al guardar mensaje: {e}")

def handle_coordinator(conn, addr):
    """Manejar conexión con el coordinador"""
    print(f"[+] Coordinador conectado desde: {addr[0]}:{addr[1]}")
    
    try:
        while True:
            # Recibir datos del coordinador
            data = conn.recv(1024)
            if not data:
                break
                
            mensaje = data.decode('utf-8')
            
            # Si es un ping del coordinador, simplemente responder
            if mensaje == "PING":
                conn.sendall("PONG".encode('utf-8'))
                continue
                
            # Procesar mensaje normal (formato: nombre_usuario|mensaje)
            try:
                nombre, contenido = mensaje.split('|', 1)
                print(f"\n[NUEVO] Mensaje de {nombre}: {contenido}")
                
                # Guardar el mensaje
                save_message(f"{nombre}: {contenido}")
                
            except ValueError:
                print(f"[!] Mensaje con formato incorrecto: {mensaje}")
    
    except Exception as e:
        print(f"[!] Error en conexión con coordinador: {e}")
    finally:
        print(f"[-] Coordinador desconectado: {addr[0]}:{addr[1]}")
        conn.close()
        
def register_with_coordinator():
    """Registrarse con el coordinador"""
    while True:
        try:
            # Conectar al coordinador
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((COORDINATOR_HOST, COORDINATOR_REGISTER_PORT))
            
            # Enviar mensaje de registro
            sock.sendall(f"SERVER:{server_id}".encode('utf-8'))
            
            # Recibir confirmación
            response = sock.recv(1024).decode('utf-8')
            if response.startswith("REGISTERED:"):
                print(f"[+] Registrado con el coordinador: {response}")
                return sock
                
        except Exception as e:
            print(f"[!] No se pudo registrar con el coordinador: {e}")
            print("[!] Reintentando en 5 segundos...")
            time.sleep(5)

def coordinator_communication(sock):
    """Mantener comunicación con el coordinador"""
    try:
        while True:
            # Esperar mensajes del coordinador
            data = sock.recv(1024)
            if not data:
                raise Exception("Conexión cerrada por el coordinador")
                
            mensaje = data.decode('utf-8')
            
            # Responder a pings
            if mensaje == "PING":
                sock.sendall("PONG".encode('utf-8'))
                continue
                
            # Procesar mensajes regulares
            try:
                nombre, contenido = mensaje.split('|', 1)
                print(f"\n[NUEVO] Mensaje de {nombre}: {contenido}")
                
                # Guardar el mensaje
                save_message(f"{nombre}: {contenido}")
                
            except ValueError:
                print(f"[!] Mensaje con formato incorrecto: {mensaje}")
                
    except Exception as e:
        print(f"[!] Error en comunicación con coordinador: {e}")
        print("[!] Intentando reconectar...")
        return False

def main():
    print(f"[+] Iniciando servidor de mensajes ID:{server_id}")
    print(f"[+] Los mensajes se guardarán en {log_file}")
    
    # Registrarse con el coordinador
    coordinator_socket = register_with_coordinator()
    
    # Iniciar hilo para comunicación con el coordinador
    comm_thread = threading.Thread(target=coordinator_communication, args=(coordinator_socket,))
    comm_thread.daemon = True
    comm_thread.start()
    
    try:
        while True:
            if not comm_thread.is_alive():
                print("[!] Reconectando con el coordinador...")
                coordinator_socket = register_with_coordinator()
                
                comm_thread = threading.Thread(target=coordinator_communication, args=(coordinator_socket,))
                comm_thread.daemon = True
                comm_thread.start()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[!] Deteniendo servidor...")
        if coordinator_socket:
            coordinator_socket.close()
        
if __name__ == "__main__":
    clear_screen()
    print("=== SERVIDOR DE MENSAJES ===")
    main()