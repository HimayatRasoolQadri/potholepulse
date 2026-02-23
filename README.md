# 🕳️ PotholePulse

Urban road damage intelligence platform with real-time mapping, REST API, and SmartDriveAR iOS integration.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Then open:
- **Map UI:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/potholes` | List all potholes (filter: `?severity=`, `?source=`, `?status=`) |
| `POST` | `/api/potholes` | Report pothole from web |
| `POST` | `/api/potholes/ios` | Report pothole from SmartDriveAR iOS app |
| `GET` | `/api/potholes/stats` | Aggregate statistics |
| `GET` | `/api/potholes/geojson` | GeoJSON feed for map integrations |
| `GET` | `/api/potholes/{id}` | Get single pothole |
| `PATCH` | `/api/potholes/{id}` | Update pothole (severity, status, description) |
| `DELETE` | `/api/potholes/{id}` | Remove pothole |

## iOS Integration (SmartDriveAR)

The `/api/potholes/ios` endpoint accepts additional fields from the iOS app:

```swift
let body: [String: Any] = [
    "latitude": location.coordinate.latitude,
    "longitude": location.coordinate.longitude,
    "severity": classifySeverity(confidence),
    "description": "Auto-detected pothole",
    "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "",
    "confidence": confidence,       // ML model confidence 0.0-1.0
    "model_version": "RDD2022-v3", // CoreML model version
    "image_url": uploadedImageURL   // Optional
]
```

Features:
- Auto-severity upgrade when confidence > 0.9
- Device tracking for deduplication
- Confidence scores displayed on map
- Source filtering (web vs iOS)

## Architecture

```
potholepulse/
├── main.py              # FastAPI backend + API routes
├── potholes.db          # SQLite database (auto-created)
├── requirements.txt
├── static/
│   └── index.html       # Leaflet map frontend (self-contained)
└── README.md
```

## Map Features

- **Leaflet** with OpenStreetMap tiles (dark theme)
- Click map to set coordinates
- Color-coded severity markers with popups
- Filter by severity or source (iOS/Web)
- Auto-refresh every 10 seconds
- Resolve or delete from map popups

## Database

SQLite with WAL mode. Schema auto-creates on first run. Data persists in `potholes.db`.
