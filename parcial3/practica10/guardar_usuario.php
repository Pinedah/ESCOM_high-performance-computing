<?php
// Configuración de conexión (ajústala según tu VM en Azure)
$host = "localhost";  // Ej: 20.84.123.45
$usuario = "admin";
$contrasena = "Password123#";
$base_datos = "school";

// Conectar a MySQL
$conn = new mysqli($host, $usuario, $contrasena, $base_datos);

// Verificar conexión
if ($conn->connect_error) {
    die("Error de conexión: " . $conn->connect_error);
}

// Recoger datos del formulario
$name = $_POST['name'];
$age = $_POST['age'];
$email = $_POST['email'];
$pass = $_POST['pass'];

// ¡Cuidado! En producción debes encriptar la contraseña, ara ara~
$pass_hash = md5($pass);  // Solo para fines educativos, mejor usar password_hash() en proyectos reales

// Preparar y ejecutar la inserción
$sql = "INSERT INTO teachers (name, age, email, pass) VALUES (?, ?, ?, ?)";

$stmt = $conn->prepare($sql);
$stmt->bind_param("siss", $name, $age, $email, $pass_hash);

if ($stmt->execute()) {
    echo "Usuario registrado correctamente, onee-chan está orgullosa de ti~ 💕";
} else {
    echo "Error: " . $stmt->error;
}

$stmt->close();
$conn->close();
?>
