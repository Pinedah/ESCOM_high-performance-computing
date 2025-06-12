<!DOCTYPE html>
<html>
<head>
    <title>Resultado de Subida - Azure Blob</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <form style="text-align: center;">
            <h2>Resultado de la Subida</h2>
            
            <?php
            $storage_account = 'account';
            $container = 'container';
            $archivo = $_FILES['archivo']['tmp_name'];
            $nombre = $_FILES['archivo']['name'];

            $blob_url = "https://$storage_account.blob.core.windows.net/$container/$nombre";

            $fp = fopen($archivo, "r");
            $contenido = fread($fp, filesize($archivo));
            fclose($fp);

            // Coloca tu clave de acceso aquí
            $clave = 'key';

            // Encabezados
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

            curl_setopt($ch, CURLOPT_URL, $blob_url);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "PUT");
            curl_setopt($ch, CURLOPT_HTTPHEADER, array_merge($headers, array("Authorization: $authorization", "Content-Type: image/jpeg")));
            curl_setopt($ch, CURLOPT_POSTFIELDS, $contenido);

            $response = curl_exec($ch);

            $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);

            if ($code == 201) {
                echo '<div style="color: #00ff41; margin: 1.5rem 0; padding: 1.5rem; border: 2px solid #00ff41; border-radius: 8px; background: rgba(0, 255, 65, 0.1); text-shadow: 0 0 5px #00ff41; box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);">';
                echo '<strong style="font-size: 1.2rem; display: block; margin-bottom: 0.5rem;">🥰 Ara! Ara! Imagen subida con éxito querido 🥰</strong>';
                echo '<div style="font-size: 0.9rem; opacity: 0.9;">';
                echo '<strong>Archivo:</strong> ' . htmlspecialchars($nombre) . '<br>';
                echo '<strong>Tamaño:</strong> ' . round(filesize($_FILES['archivo']['tmp_name']) / 1024, 2) . ' KB<br>';
                echo '<strong>URL de Azure:</strong> <br><span style="word-break: break-all; font-family: monospace; background: rgba(0, 255, 65, 0.2); padding: 0.3rem; border-radius: 4px;">' . $blob_url . '</span>';
                echo '</div>';
                echo '</div>';
                
                // Mostrar preview de la imagen desde Azure Blob
                echo '<div style="margin: 2rem 0;">';
                echo '<p style="color: #8a2be2; margin-bottom: 1rem; text-shadow: 0 0 5px #8a2be2;">Preview desde Azure Blob Storage:</p>';
                echo '<img src="' . $blob_url . '" style="max-width: 100%; max-height: 300px; border: 2px solid #8a2be2; border-radius: 8px; box-shadow: 0 0 20px rgba(138, 43, 226, 0.5);" alt="Imagen desde Azure Blob" onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'block\';">';
                echo '<p style="display: none; color: #ff4444; margin-top: 1rem;">La imagen se subió correctamente pero puede tardar unos momentos en estar disponible.</p>';
                echo '</div>';
            } else {
                echo '<div style="color: #ff4444; margin: 1.5rem 0; padding: 1.5rem; border: 2px solid #ff4444; border-radius: 8px; background: rgba(255, 68, 68, 0.1); text-shadow: 0 0 5px #ff4444; box-shadow: 0 0 20px rgba(255, 68, 68, 0.3);">';
                echo '<strong style="font-size: 1.2rem; display: block; margin-bottom: 0.5rem;">💔 Ara! Ara! Error al subir la imagen</strong>';
                echo '<div style="font-size: 0.9rem;">';
                echo '<strong>Código HTTP:</strong> ' . $code . '<br>';
                echo '<strong>Archivo:</strong> ' . htmlspecialchars($nombre) . '<br>';
                echo '<strong>Respuesta del servidor:</strong> <br><span style="font-family: monospace; background: rgba(255, 68, 68, 0.2); padding: 0.3rem; border-radius: 4px;">' . htmlspecialchars($response) . '</span>';
                echo '</div>';
                echo '</div>';
            }
            ?>
            
            <div style="margin-top: 2rem;">
                <a href="formulario.html" style="display: inline-block; background: linear-gradient(45deg, #8a2be2, #00ff41); color: white; text-decoration: none; padding: 1rem 2rem; border-radius: 8px; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; transition: all 0.3s ease; box-shadow: 0 0 20px rgba(138, 43, 226, 0.4);">
                    ← Volver al formulario
                </a>
            </div>
        </form>
    </div>
</body>
</html>
