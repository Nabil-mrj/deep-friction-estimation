"""Serveur de répartition pour la ferme de rendu Houdini."""

import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import socketio

# Paramètres généraux
REPERTOIRE_RENDUS = Path(os.environ.get("RENDER_ROOT", "/media/modalive/dataset-tissu"))
HEARTBEAT_TIMEOUT_SECONDS = int(os.environ.get("HEARTBEAT_TIMEOUT", "60"))
ASSIGN_RETRY_DELAY_SECONDS = int(os.environ.get("ASSIGN_RETRY_DELAY", "2"))

# Socket.IO
sio = socketio.Server()
app = socketio.WSGIApp(sio)

# État global protégé par un verrou
rendus = []  # liste de dicts: {path, completed, assigned_to, started_at, completed_at}
clients = {}  # sid -> client_id
dernier_heartbeat = {}  # client_id -> datetime
lock = threading.Lock()


def charger_rendus():
    # Construit la liste des tâches à partir de la structure disque
    with lock:
        rendus.clear()
        if not REPERTOIRE_RENDUS.exists():
            REPERTOIRE_RENDUS.mkdir(parents=True, exist_ok=True)

        for root, dirs, files in os.walk(REPERTOIRE_RENDUS):
            # On considère seulement les dossiers feuilles comme une unité de rendu
            if dirs:
                continue

            dossier = Path(root)
            has_video = any(
                f.lower().endswith((".mp4", ".mov", ".mkv"))
                for f in files
            )

            rendus.append(
                {
                    "path": str(dossier),
                    "completed": has_video,
                    "assigned_to": None,
                    "started_at": None,
                    "completed_at": datetime.utcnow() if has_video else None,
                }
            )

        total = len(rendus)
        termines = sum(1 for r in rendus if r["completed"])
        print(f"[INIT] Tâches détectées: {termines}/{total}")


def obtenir_tache_suivante(client_id):
    # Sélectionne la prochaine tâche non complétée et non assignée
    with lock:
        for rendu in rendus:
            if not rendu["completed"] and rendu["assigned_to"] is None:
                rendu["assigned_to"] = client_id
                rendu["started_at"] = datetime.utcnow()
                return rendu
    return None


def marquer_tache_complete(path, client_id):
    # Marque la tâche comme terminée si elle existe
    path_str = str(Path(path))
    with lock:
        for rendu in rendus:
            if Path(rendu["path"]) == Path(path_str):
                rendu["completed"] = True
                rendu["assigned_to"] = client_id
                rendu["completed_at"] = datetime.utcnow()
                break


def reaffecter_taches_client(client_id):
    # Libère les tâches en cours d'un client inactif
    with lock:
        for rendu in rendus:
            if not rendu["completed"] and rendu["assigned_to"] == client_id:
                rendu["assigned_to"] = None
                rendu["started_at"] = None


def verifier_clients_inactifs():
    # Surveille les heartbeats et réaffecte les tâches si nécessaire
    while True:
        time.sleep(HEARTBEAT_TIMEOUT_SECONDS)
        maintenant = datetime.utcnow()
        inactifs = []

        with lock:
            for client_id, last in list(dernier_heartbeat.items()):
                if maintenant - last > timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS):
                    inactifs.append(client_id)

        for client_id in inactifs:
            print(f"[HEARTBEAT] Client inactif détecté: {client_id}")
            reaffecter_taches_client(client_id)
            with lock:
                dernier_heartbeat.pop(client_id, None)


def afficher_progression():
    # Affiche périodiquement la progression globale
    while True:
        time.sleep(10)
        with lock:
            total = len(rendus)
            termines = sum(1 for r in rendus if r["completed"])
        if total > 0:
            pourcentage = 100.0 * termines / total
            print(f"[PROGRESSION] {termines}/{total} rendus terminés ({pourcentage:.2f}%)")
        else:
            print("[PROGRESSION] Aucune tâche détectée")


@sio.event
def connect(sid, environ):
    # Enregistre la connexion d'un nouveau client
    client_id = environ.get("HTTP_X_CLIENT_ID")
    if not client_id:
        client_id = f"anonymous-{sid}"

    with lock:
        clients[sid] = client_id
        dernier_heartbeat[client_id] = datetime.utcnow()

    print(f"[CONNECT] Client connecté: sid={sid}, client_id={client_id}")


@sio.event
def disconnect(sid):
    # Supprime la connexion d'un client
    with lock:
        client_id = clients.pop(sid, None)
    if client_id:
        print(f"[DISCONNECT] Client déconnecté: {client_id}")


@sio.on("heartbeat")
def heartbeat(sid, data=None):
    # Met à jour le heartbeat du client
    with lock:
        client_id = clients.get(sid)
        if not client_id:
            return
        dernier_heartbeat[client_id] = datetime.utcnow()


@sio.on("waiting")
def waiting(sid, data=None):
    # Attribue une tâche au client quand il se déclare disponible
    with lock:
        client_id = clients.get(sid)
    if not client_id:
        return

    rendu = obtenir_tache_suivante(client_id)
    if not rendu:
        sio.emit("idle", {}, room=sid)
        return

    payload = {
        "path": rendu["path"],
    }
    print(f"[ASSIGN] {client_id} -> {rendu['path']}")
    sio.emit("render", payload, room=sid)


@sio.on("done")
def done(sid, data):
    # Reçoit la confirmation de fin de rendu
    if not isinstance(data, dict):
        return

    path = data.get("path")
    if not path:
        return

    with lock:
        client_id = clients.get(sid)
    if not client_id:
        return

    marquer_tache_complete(path, client_id)
    print(f"[DONE] Rendu terminé par {client_id}: {path}")


# Initialisation et lancement
charger_rendus()
threading.Thread(target=verifier_clients_inactifs, daemon=True).start()
threading.Thread(target=afficher_progression, daemon=True).start()

if __name__ == "__main__":
    import eventlet
    import eventlet.wsgi

    host = os.environ.get("RENDER_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("RENDER_SERVER_PORT", "5000"))
    print(f"[SERVER] Démarrage sur {host}:{port}, racine: {REPERTOIRE_RENDUS}")
    eventlet.wsgi.server(eventlet.listen((host, port)), app, log_output=False)
