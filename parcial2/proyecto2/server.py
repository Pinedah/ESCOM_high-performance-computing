#!/usr/bin/env python3
import socket
import threading
import os
import time
import datetime

# Configuración del servidor
SERVER_HOST = '0.0.0.0'  # Escuchar en todas las interfaces
SERVER_PORT = 6001

# Para guardar los mensajes
log_file = "mensajes.log"

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

def main():
    # Crear socket servidor
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen(1)  # Solo esperamos conexión del coordinador
        print(f"[+] Servidor iniciado en {SERVER_HOST}:{SERVER_PORT}")
        print(f"[+] Los mensajes se guardarán en {log_file}")
        
        while True:
            conn, addr = server_socket.accept()
            
            # Iniciar hilo para manejar la conexión
            coordinator_thread = threading.Thread(target=handle_coordinator, args=(conn, addr))
            coordinator_thread.daemon = True
            coordinator_thread.start()
            
    except KeyboardInterrupt:
        print("\n[!] Deteniendo servidor...")
    finally:
        server_socket.close()
        
if __name__ == "__main__":
    clear_screen()
    print("=== SERVIDOR DE MENSAJES ===")
    main()