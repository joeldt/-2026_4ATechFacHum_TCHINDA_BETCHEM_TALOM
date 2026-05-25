# -2026_4ATechFacHum_TCHINDA_BETCHEM_TALOM
# FeatherMind

Application de biofeedback temps reel inspiree de la sieste hypnagogique de Salvador Dali.

FeatherMind utilise une carte PLUX/BITalino et des signaux physiologiques pour visualiser l'etat de relaxation d'un utilisateur. Une plume descend progressivement lorsque la relaxation augmente. Si la plume atteint le seuil d'endormissement, une alarme sonore se declenche pour reveiller l'utilisateur.

## Fonctionnalites

- Acquisition temps reel avec l'API PLUX.
- Mode simulation sans capteurs.
- Affichage des courbes PZT, PPG et accelerometre.
- Score de relaxation explicable, sans machine learning.
- Alarme de reveil lorsque la plume atteint le seuil.
- Alarme de mouvement brusque via accelerometre.

## Materiel

Capteurs utilises :

| Signal | Capteur | Port |
| --- | --- | --- |
| Respiration | PZT / ceinture respiratoire | A5 |
| Pouls | PPG | A4 |
| Mouvement | Accelerometre X/Y/Z | A1/A2/A3 |

Frequence d'acquisition : **100 Hz**.

Le signal EMG a ete exclu car il n'etait pas assez pertinent pour ce contexte de relaxation.

## Installation

Depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

L'acquisition reelle utilise l'API PLUX locale. Le dossier `PLUX-API-Python3` doit etre disponible dans le projet, notamment sous :

```text
feathermind/PLUX-API-Python3
```

## Configuration

La configuration principale est dans `config.local.json`.

Exemple utilise pendant le projet :

```json
{
  "mac_address": "BTH98:D3:51:FE:87:0B",
  "sampling_rate_hz": 100,
  "channels": {
    "acc_x": 1,
    "acc_y": 2,
    "acc_z": 3,
    "ppg": 4,
    "pzt": 5
  }
}
```

En mode PLUX, les ports sont numerotes a partir de 1 :

- A1 = `1`
- A2 = `2`
- A3 = `3`
- A4 = `4`
- A5 = `5`
- A6 = `6`

## Lancer l'application

### Mode reel avec capteurs

```powershell
python -m feathermind.app --mode plux --config config.local.json
```

### Simulation simple

```powershell
python -m feathermind.app --mode simulate --config config.local.json
```

### Demo automatique

```powershell
python -m feathermind.app --mode demo --config config.local.json
```

Le scenario demo enchaine : utilisateur actif, relaxation progressive, plume basse, mouvement brusque, retour au calme.

## Tests et validation

Verifier la connexion PLUX :

```powershell
python -m feathermind.check_plux --config config.local.json
```

Enregistrer 30 secondes de donnees reelles :

```powershell
python -m feathermind.record_raw --mode plux --config config.local.json --seconds 30
```

Enregistrer une simulation :

```powershell
python -m feathermind.record_raw --mode demo --config config.local.json --seconds 20 --output data\demo_test.csv
```

## Score de relaxation

Le score est base sur trois contributions :

- **50 % respiration** : frequence, amplitude et regularite du signal PZT.
- **35 % immobilite** : mouvement RMS calcule avec l'accelerometre.
- **15 % PPG** : qualite du signal PPG et rythme cardiaque plausible.

Plus le score augmente, plus la plume descend.

## Alarmes

Deux alarmes sont disponibles :

- **Alarme de reveil** : declenchee quand le score de relaxation depasse le seuil configure.
- **Alarme mouvement** : declenchee quand l'accelerometre detecte une secousse brusque.

Les seuils et les cooldowns sont configurables dans `config.local.json`.

## Structure du projet

```text
feathermind/
  app.py          Interface Tkinter et affichage temps reel
  acquisition.py  Acquisition PLUX, simulation et demo
  processing.py   Traitement PZT, PPG et accelerometre
  biofeedback.py  Calcul du score et etat de la plume
  config.py       Lecture de la configuration JSON
  check_plux.py   Test de connexion a la carte
  record_raw.py   Enregistrement CSV des signaux

```

## Membres du groupe

- Warren Shamir Betchem Ngon
- Nana Talom Franck
- Joel Tchinda Tapa

Projet realise dans le cadre du cours **Technologie et Facteurs Humains - EPITA 4A**.

