<?php
// Configuración
$storage_account = 'account'; // Nombre de la cuenta de almacenamiento
$container = 'container'; // Nombre del contenedor
$clave = 'key';
$blob_url_base = "https://$storage_account.blob.core.windows.net/$container";

// SUBIDA de imagen
$mensaje = '';
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['archivo'])) {
    $archivo = $_FILES['archivo']['tmp_name'];
    $nombre = $_FILES['archivo']['name'];

    $url = "$blob_url_base/$nombre";

    $fp = fopen($archivo, "r");
    $contenido = fread($fp, filesize($archivo));
    fclose($fp);

    $date = gmdate("D, d M Y H:i:s T");
    $length = strlen($contenido);
    $headers = array(
        "x-ms-blob-type: BlockBlob",
        "x-ms-date: $date",
        "x-ms-version: 2020-10-02",
        "Content-Length: $length"
    );

    $resource = "/$storage_account/$container/$nombre";
    $stringToSign = "PUT\n\n\n$length\n\nimage/jpeg\n\n\n\n\n\n\nx-ms-blob-type:BlockBlob\nx-ms-date:$date\nx-ms-version:2020-10-02\n$resource";
    $signature = base64_encode(hash_hmac('sha256', $stringToSign, base64_decode($clave), true));
    $authorization = "SharedKey $storage_account:$signature";

    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "PUT");
    curl_setopt($ch, CURLOPT_HTTPHEADER, array_merge($headers, array("Authorization: $authorization", "Content-Type: image/jpeg")));
    curl_setopt($ch, CURLOPT_POSTFIELDS, $contenido);

    $response = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    $mensaje = ($code == 201) ? "Ara! Ara! Imagen subida con éxito 🥰" : "Upsi, hubo un error al subir: Código HTTP $code";
    
}
?>

<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Subida y Galería de Imágenes</title>
    <style>
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a0a1a 50%, #0a1a0a 100%);
            color: #00ff41;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            position: relative;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 20% 80%, rgba(138, 43, 226, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(0, 255, 65, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }

        form {
            background: rgba(20, 0, 20, 0.9);
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #8a2be2;
            box-shadow: 
                0 0 20px rgba(138, 43, 226, 0.5),
                inset 0 0 20px rgba(138, 43, 226, 0.1);
            backdrop-filter: blur(10px);
        }

        h2 {
            color: #8a2be2;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 
                0 0 10px #8a2be2,
                0 0 20px #8a2be2,
                0 0 30px #8a2be2;
            font-size: 1.8rem;
            letter-spacing: 2px;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
            color: #00ff41;
            text-shadow: 0 0 5px #00ff41;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 1px;
        }

        input[type="file"] {
            width: 100%;
            padding: 0.8rem;
            border: 2px solid #8a2be2;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.8);
            color: #00ff41;
            outline: none;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: inset 0 0 10px rgba(138, 43, 226, 0.2);
            box-sizing: border-box;
            cursor: pointer;
        }

        input[type="file"]:hover,
        input[type="file"]:focus {
            border-color: #00ff41;
            box-shadow: 
                0 0 15px rgba(0, 255, 65, 0.6),
                inset 0 0 15px rgba(0, 255, 65, 0.1);
        }

        input[type="submit"] {
            margin-top: 1.5rem;
            background: linear-gradient(45deg, #8a2be2, #00ff41);
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            font-size: 1.1rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s ease;
            box-shadow: 
                0 0 20px rgba(138, 43, 226, 0.4),
                0 4px 15px rgba(0, 0, 0, 0.3);
        }

        input[type="submit"]:hover {
            background: linear-gradient(45deg, #00ff41, #8a2be2);
            box-shadow: 
                0 0 30px rgba(0, 255, 65, 0.8),
                0 0 50px rgba(138, 43, 226, 0.4),
                0 8px 25px rgba(0, 0, 0, 0.4);
            transform: translateY(-2px);
        }
        
        .gallery-section {
            background: rgba(20, 0, 20, 0.9);
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #8a2be2;
            box-shadow: 
                0 0 20px rgba(138, 43, 226, 0.5),
                inset 0 0 20px rgba(138, 43, 226, 0.1);
            backdrop-filter: blur(10px);
            margin-top: 2rem;
        }
        
        .galeria {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-top: 2rem;
        }
        
        .img-card {
            text-align: center;
            background: rgba(20, 0, 20, 0.9);
            padding: 1rem;
            border-radius: 12px;
            border: 2px solid #8a2be2;
            box-shadow: 
                0 0 15px rgba(138, 43, 226, 0.4),
                inset 0 0 15px rgba(138, 43, 226, 0.1);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .img-card:hover {
            border-color: #00ff41;
            box-shadow: 
                0 0 25px rgba(0, 255, 65, 0.6),
                0 0 40px rgba(138, 43, 226, 0.3),
                inset 0 0 20px rgba(0, 255, 65, 0.1);
            transform: translateY(-5px);
        }
        
        .img-card img {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border: 2px solid #8a2be2;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
        }
        
        .img-card:hover img {
            border-color: #00ff41;
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.5);
        }
        
        .img-card p {
            color: #00ff41;
            font-size: 0.8rem;
            margin-top: 0.5rem;
            text-shadow: 0 0 3px #00ff41;
            word-break: break-all;
        }
        
        .mensaje-exito {
            color: #00ff41;
            margin: 1.5rem 0;
            padding: 1rem;
            border: 2px solid #00ff41;
            border-radius: 8px;
            background: rgba(0, 255, 65, 0.1);
            text-shadow: 0 0 5px #00ff41;
            text-align: center;
            font-weight: bold;
        }
        
        .mensaje-error {
            color: #ff4444;
            margin: 1.5rem 0;
            padding: 1rem;
            border: 2px solid #ff4444;
            border-radius: 8px;
            background: rgba(255, 68, 68, 0.1);
            text-shadow: 0 0 5px #ff4444;
            text-align: center;
            font-weight: bold;
        }
        
        .no-images {
            color: #8a2be2;
            text-align: center;
            margin: 2rem 0;
            padding: 2rem;
            border: 2px solid #8a2be2;
            border-radius: 12px;
            background: rgba(138, 43, 226, 0.1);
            text-shadow: 0 0 5px #8a2be2;
            font-size: 1.2rem;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <form action="galeria.php" method="POST" enctype="multipart/form-data">
            <h2>📤 Subir Imagen a Azure Blob Storage</h2>
            
            <?php if ($mensaje): ?>
                <div class="<?php echo (strpos($mensaje, 'éxito') !== false) ? 'mensaje-exito' : 'mensaje-error'; ?>">
                    <?php echo $mensaje; ?>
                </div>
            <?php endif; ?>

            <div class="form-group">
                <label for="archivo">Seleccionar imagen:</label>
                <input type="file" name="archivo" id="archivo" accept="image/*" required>
            </div>
            <input type="submit" value="Subir Imagen">
        </form>

        <div class="gallery-section">
            <h2 style="text-align: center; margin-bottom: 2rem;">🖼️ Galería de Imágenes</h2>
            <div class="galeria">
                <?php
                // Mostrar imágenes
                $date = gmdate("D, d M Y H:i:s T");
                $headers = [
                    "x-ms-date: $date",
                    "x-ms-version: 2020-10-02"
                ];

                // Corregir la construcción del resource string
                $resource = "/$storage_account/$container\ncomp:list\nrestype:container";
                $stringToSign = "GET\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:$date\nx-ms-version:2020-10-02\n$resource";
                $signature = base64_encode(hash_hmac('sha256', $stringToSign, base64_decode($clave), true));
                $authorization = "SharedKey $storage_account:$signature";

                $curl = curl_init();
                curl_setopt_array($curl, [
                    CURLOPT_URL => "$blob_url_base?restype=container&comp=list",
                    CURLOPT_RETURNTRANSFER => true,
                    CURLOPT_HTTPHEADER => array_merge($headers, ["Authorization: $authorization"]),
                    CURLOPT_SSL_VERIFYPEER => false,
                    CURLOPT_VERBOSE => false
                ]);

                $response = curl_exec($curl);
                $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
                $curlError = curl_error($curl);
                curl_close($curl);

                // Debug: mostrar información si hay problemas
                if ($curlError) {
                    echo "<div class='mensaje-error'>Error cURL: $curlError</div>";
                }
                
                if ($httpCode !== 200) {
                    echo "<div class='mensaje-error'>Error HTTP: $httpCode<br>Respuesta: " . htmlspecialchars($response) . "</div>";
                }

                if ($response && $httpCode === 200) {
                    // Intentar parsear el XML
                    libxml_use_internal_errors(true);
                    $xml = simplexml_load_string($response);
                    
                    if ($xml === false) {
                        $errors = libxml_get_errors();
                        echo "<div class='mensaje-error'>Error parseando XML:<br>";
                        foreach ($errors as $error) {
                            echo htmlspecialchars($error->message) . "<br>";
                        }
                        echo "Respuesta raw: " . htmlspecialchars(substr($response, 0, 500)) . "...</div>";
                        libxml_clear_errors();
                    } else {
                        // Verificar si hay blobs
                        if (isset($xml->Blobs->Blob)) {
                            foreach ($xml->Blobs->Blob as $blob) {
                                $nombreBlob = (string)$blob->Name;
                                $url = "$blob_url_base/$nombreBlob";
                                echo "<div class='img-card'>
                                        <a href='$url' target='_blank'>
                                            <img src='$url' alt='$nombreBlob' loading='lazy' 
                                                 onerror=\"this.style.display='none'; this.nextElementSibling.style.display='block';\">
                                        </a>
                                        <div style='display:none; color:#ff4444; font-size:0.7rem; margin-top:0.5rem;'>
                                            Error cargando imagen
                                        </div>
                                        <p>$nombreBlob</p>
                                      </div>";
                            }
                        } else {
                            echo "<div class='no-images'>Ara! Ara! No hay imágenes aún, cielito 💙</div>";
                        }
                    }
                } else {
                    echo "<div class='mensaje-error'>Upsi, hubo un error al cargar galería 😢<br>";
                    echo "Código HTTP: $httpCode<br>";
                    if ($response) {
                        echo "Respuesta: " . htmlspecialchars(substr($response, 0, 200)) . "...";
                    }
                    echo "</div>";
                }
                ?>
            </div>
        </div>
    </div>
</body>
</html>
