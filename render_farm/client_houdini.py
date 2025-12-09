"""Client Houdini pour la ferme de rendu distribuée."""

import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import socketio

# Paramètres généraux
CLIENT_ID_FILE = Path(os.environ.get("CLIENT_ID_FILE", "client_id.txt"))
SERVER_URL = os.environ.get("RENDER_SERVER_URL", "http://127.0.0.1:5000")
HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))
HOU_SCENE_PATH = os.environ.get("HOU_SCENE_PATH", "/path/to/scene.hip")

sio = socketio.Client()
client_id = None


def charger_client_id():
    # Charge ou génère un identifiant persistant pour la machine cliente
    if CLIENT_ID_FILE.exists():
        return CLIENT_ID_FILE.read_text().strip()

    new_id = str(uuid.uuid4())
    CLIENT_ID_FILE.write_text(new_id)
    return new_id


def envoyer_heartbeat():
    # Envoie périodiquement un heartbeat au serveur
    while True:
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)
        try:
            sio.emit("heartbeat", {"client_id": client_id})
        except Exception:
            # On ne casse pas le thread en cas de déconnexion temporaire
            pass


def construire_video(sequence_dir, output_path, fps=30):
    # Construit une vidéo à partir d'une séquence d'images
    images = sorted(
        [
            f
            for f in os.listdir(sequence_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".exr"))
        ]
    )
    if not images:
        raise RuntimeError(f"Aucune image trouvée dans {sequence_dir}")

    first_image_path = os.path.join(sequence_dir, images[0])
    first_frame = cv2.imread(first_image_path)
    if first_frame is None:
        raise RuntimeError(f"Impossible de lire la première image: {first_image_path}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for filename in images:
            frame_path = os.path.join(sequence_dir, filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            writer.write(frame)
    finally:
        writer.release()


def executer_rendu(path):
    # Configure la scène Houdini et lance le rendu pour un dossier de sortie
    import hou

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    hou.hipFile.load(HOU_SCENE_PATH)

    # Ces noms de nœuds sont à adapter à la scène de production
    karma_node = hou.node("/out/karma1")
    if karma_node is None:
        raise RuntimeError("Nœud Karma introuvable dans la scène (/out/karma1)")

    sequence_pattern = str(output_dir / "frame.$F4.exr")
    karma_node.parm("vm_picture").set(sequence_pattern)

    karma_node.render()

    video_path = str(output_dir / "render.mp4")
    construire_video(output_dir, video_path)
    return video_path


@sio.event
def connect():
    # Déclenche la demande de première tâche lors de la connexion
    print(f"[CONNECT] Connecté au serveur en tant que {client_id}")
    sio.emit("waiting")


@sio.event
def disconnect():
    # Log de déconnexion
    print("[DISCONNECT] Déconnecté du serveur")


@sio.on("render")
def on_render(data):
    # Reçoit une tâche de rendu, l'exécute puis informe le serveur
    if not isinstance(data, dict):
        return

    path = data.get("path")
    if not path:
        return

    print(f"[RENDER] Démarrage du rendu pour: {path}")
    try:
        video_path = executer_rendu(path)
        payload = {
            "path": path,
            "video_path": video_path,
            "client_id": client_id,
        }
        sio.emit("done", payload)
        print(f"[RENDER] Rendu terminé: {path}")
    except Exception as exc:
        print(f"[ERROR] Erreur pendant le rendu {path}: {exc}")
    finally:
        # Se remet en attente d'une nouvelle tâche
        sio.emit("waiting")


@sio.on("idle")
def on_idle(data=None):
    # Indique qu'il n'y a plus de tâches disponibles pour le moment
    print("[IDLE] Aucune tâche disponible, attente avant nouvelle demande")
    time.sleep(5)
    sio.emit("waiting")


def main():
    # Point d'entrée du client
    global client_id

    client_id = charger_client_id()
    headers = {"X-Client-ID": client_id}
    print(f"[CLIENT] Démarrage avec client_id={client_id}, serveur={SERVER_URL}")

    sio.connect(SERVER_URL, headers=headers)
    threading.Thread(target=envoyer_heartbeat, daemon=True).start()
    sio.wait()


if __name__ == "__main__":
    main()
