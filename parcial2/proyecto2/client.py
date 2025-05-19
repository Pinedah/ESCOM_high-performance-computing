#!/usr/bin/env python3
import socket
import os

# Configuración del cliente
COORDINATOR_HOST = '192.168.100.52'  # Cambiar a la IP del coordinador
COORDINATOR_PORT = 6000

def clear_screen():
    os.system('clear')

def main():
    # Crear socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Conectar al coordinador
    try:
        client_socket.connect((COORDINATOR_HOST, COORDINATOR_PORT))
        print(f"[+] Conectado al coordinador en {COORDINATOR_HOST}:{COORDINATOR_PORT}")
    except ConnectionRefusedError:
        print(f"[-] No se pudo conectar al coordinador en {COORDINATOR_HOST}:{COORDINATOR_PORT}")
        return
    
    # Obtener nombre de cliente
    client_name = input("Ingresa tu nombre de usuario: ")
    
    try:
        while True:
            mensaje = input(f"\n[{client_name}] Escribe un mensaje (o 'salir' para terminar): ")
            
            if mensaje.lower() == 'salir':
                break
                
            # Formato: nombre_usuario|mensaje
            mensaje_formateado = f"{client_name}|{mensaje}"
            client_socket.sendall(mensaje_formateado.encode('utf-8'))
            print("[+] Mensaje enviado correctamente")
            
            # Esperar confirmación
            try:
                client_socket.settimeout(2.0)
                response = client_socket.recv(1024).decode('utf-8')
                print(f"[+] Respuesta del coordinador: {response}")
            except socket.timeout:
                print("[-] No se recibió confirmación del coordinador")
            
    except KeyboardInterrupt:
        print("\n[!] Cerrando cliente...")
    finally:
        client_socket.close()
        
if __name__ == "__main__":
    clear_screen()
    print("=== CLIENTE DE MENSAJERÍA ===")
    main()