"""
PotholePulse — FastAPI Backend
Real REST API with SQLite storage, iOS device integration, and Leaflet frontend.
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import sqlite3
import json
import os
import uuid

# ─── App Setup ───────────────────────────────────────────────

app = FastAPI(
    title="PotholePulse API",
    description="Urban road damage intelligence platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "potholes.db")

# ─── Database ────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS potholes (
            id TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            severity TEXT NOT NULL DEFAULT 'medium',
            description TEXT DEFAULT '',
            reported_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            source TEXT DEFAULT 'web',
            device_id TEXT DEFAULT NULL,
            confidence REAL DEFAULT NULL,
            image_url TEXT DEFAULT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_severity ON potholes(severity);
        CREATE INDEX IF NOT EXISTS idx_status ON potholes(status);
        CREATE INDEX IF NOT EXISTS idx_location ON potholes(latitude, longitude);
        CREATE INDEX IF NOT EXISTS idx_source ON potholes(source);
    """)
    conn.close()

init_db()

# ─── Models ──────────────────────────────────────────────────

class PotholeCreate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    severity: str = Field("medium", pattern="^(low|medium|high)$")
    description: Optional[str] = ""

class PotholeIOSReport(BaseModel):
    """Schema for SmartDriveAR iOS auto-reports"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    severity: str = Field("medium", pattern="^(low|medium|high)$")
    description: Optional[str] = ""
    device_id: str = Field(..., description="Unique iOS device identifier")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="ML model confidence score")
    image_url: Optional[str] = Field(None, description="URL to captured pothole image")
    model_version: Optional[str] = Field(None, description="CoreML model version string")

class PotholeUpdate(BaseModel):
    severity: Optional[str] = Field(None, pattern="^(low|medium|high)$")
    description: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(active|resolved|disputed)$")

def generate_id():
    """Generate a short readable ID"""
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) as c FROM potholes").fetchone()
    conn.close()
    return f"PH-{row['c'] + 1:04d}"

def row_to_dict(row):
    return dict(row)

# ─── API Routes ──────────────────────────────────────────────

# --- POST /api/potholes — Report from web ---
@app.post("/api/potholes", status_code=201)
def create_pothole(data: PotholeCreate):
    pothole_id = generate_id()
    now = datetime.utcnow().isoformat() + "Z"
    conn = get_db()
    conn.execute(
        """INSERT INTO potholes (id, latitude, longitude, severity, description, reported_at, source)
           VALUES (?, ?, ?, ?, ?, ?, 'web')""",
        (pothole_id, data.latitude, data.longitude, data.severity, data.description, now)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,)).fetchone()
    conn.close()
    return row_to_dict(row)


# --- POST /api/potholes/ios — Report from SmartDriveAR ---
@app.post("/api/potholes/ios", status_code=201)
def create_pothole_ios(data: PotholeIOSReport):
    """
    Endpoint for SmartDriveAR iOS app.
    Accepts ML confidence scores, device IDs, and optional image URLs.
    
    Usage from Swift:
        let body: [String: Any] = [
            "latitude": location.coordinate.latitude,
            "longitude": location.coordinate.longitude,
            "severity": classifySeverity(confidence),
            "description": "Auto-detected pothole",
            "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "",
            "confidence": confidence,
            "model_version": "RDD2022-v3"
        ]
    """
    pothole_id = generate_id()
    now = datetime.utcnow().isoformat() + "Z"

    # Auto-upgrade severity based on confidence
    severity = data.severity
    if data.confidence and data.confidence > 0.9 and severity != "high":
        severity = "high"

    conn = get_db()
    conn.execute(
        """INSERT INTO potholes 
           (id, latitude, longitude, severity, description, reported_at, source, device_id, confidence, image_url)
           VALUES (?, ?, ?, ?, ?, ?, 'ios', ?, ?, ?)""",
        (pothole_id, data.latitude, data.longitude, severity,
         data.description or "Auto-detected by SmartDriveAR",
         now, data.device_id, data.confidence, data.image_url)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,)).fetchone()
    conn.close()
    return row_to_dict(row)


# --- GET /api/potholes — List all ---
@app.get("/api/potholes")
def list_potholes(
    severity: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    status: Optional[str] = Query("active", pattern="^(active|resolved|disputed|all)$"),
    source: Optional[str] = Query(None, pattern="^(web|ios)$"),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    conn = get_db()
    clauses = []
    params = []

    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if status and status != "all":
        clauses.append("status = ?")
        params.append(status)
    if source:
        clauses.append("source = ?")
        params.append(source)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    
    count = conn.execute(f"SELECT COUNT(*) as c FROM potholes {where}", params).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM potholes {where} ORDER BY reported_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset]
    ).fetchall()
    conn.close()

    return {
        "count": count,
        "limit": limit,
        "offset": offset,
        "potholes": [row_to_dict(r) for r in rows]
    }


# --- GET /api/potholes/stats ---
@app.get("/api/potholes/stats")
def pothole_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM potholes WHERE status = 'active'").fetchone()["c"]
    
    severity_rows = conn.execute(
        "SELECT severity, COUNT(*) as c FROM potholes WHERE status = 'active' GROUP BY severity"
    ).fetchall()
    by_severity = {r["severity"]: r["c"] for r in severity_rows}

    source_rows = conn.execute(
        "SELECT source, COUNT(*) as c FROM potholes GROUP BY source"
    ).fetchall()
    by_source = {r["source"]: r["c"] for r in source_rows}

    # Recent activity (last 24h)
    recent = conn.execute(
        "SELECT COUNT(*) as c FROM potholes WHERE reported_at > datetime('now', '-1 day')"
    ).fetchone()["c"]

    conn.close()
    return {
        "total_active": total,
        "by_severity": {"low": 0, "medium": 0, "high": 0, **by_severity},
        "by_source": {"web": 0, "ios": 0, **by_source},
        "last_24h": recent,
    }


# --- GET /api/potholes/geojson — For map consumption ---
@app.get("/api/potholes/geojson")
def potholes_geojson(
    severity: Optional[str] = Query(None),
):
    conn = get_db()
    if severity:
        rows = conn.execute(
            "SELECT * FROM potholes WHERE status = 'active' AND severity = ?", (severity,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM potholes WHERE status = 'active'").fetchall()
    conn.close()

    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r["longitude"], r["latitude"]]
            },
            "properties": {
                "id": r["id"],
                "severity": r["severity"],
                "description": r["description"],
                "reported_at": r["reported_at"],
                "source": r["source"],
                "confidence": r["confidence"],
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }


# --- GET /api/potholes/:id ---
@app.get("/api/potholes/{pothole_id}")
def get_pothole(pothole_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Pothole not found")
    return row_to_dict(row)


# --- PATCH /api/potholes/:id ---
@app.patch("/api/potholes/{pothole_id}")
def update_pothole(pothole_id: str, data: PotholeUpdate):
    conn = get_db()
    row = conn.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Pothole not found")
    
    updates = {}
    if data.severity is not None:
        updates["severity"] = data.severity
    if data.description is not None:
        updates["description"] = data.description
    if data.status is not None:
        updates["status"] = data.status

    if updates:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE potholes SET {set_clause} WHERE id = ?", list(updates.values()) + [pothole_id])
        conn.commit()

    row = conn.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,)).fetchone()
    conn.close()
    return row_to_dict(row)


# --- DELETE /api/potholes/:id ---
@app.delete("/api/potholes/{pothole_id}")
def delete_pothole(pothole_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Pothole not found")
    conn.execute("DELETE FROM potholes WHERE id = ?", (pothole_id,))
    conn.commit()
    conn.close()
    return {"deleted": pothole_id}


# ─── Frontend ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path) as f:
        return f.read()


# ─── Startup ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n🕳️  PotholePulse API running at http://localhost:8000")
    print("📡 API Docs:  http://localhost:8000/docs")
    print("🗺️  Map:       http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
