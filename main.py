"""
PotholePulse — FastAPI Backend
Real REST API with SQLite storage, iOS device integration, and Leaflet frontend.
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import sqlite3
import json
import os
import uuid
import math
import httpx

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


class RoutePoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class RouteCompareRequest(BaseModel):
    """Compare multiple routes for pothole density"""
    routes: List[dict] = Field(..., description="List of routes, each with 'name' and 'waypoints' (list of {lat, lng})")
    corridor_width: float = Field(100, ge=10, le=500, description="Corridor width in meters from route line")


# ─── Geo Helpers ─────────────────────────────────────────

def haversine(lat1, lng1, lat2, lng2):
    """Distance in meters between two GPS points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Minimum distance (meters) from point P to line segment A-B, all in lat/lng."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx*abx + aby*aby
    if ab2 == 0:
        return haversine(px, py, ax, ay)
    t = max(0, min(1, (apx*abx + apy*aby) / ab2))
    closest_lat = ax + t * abx
    closest_lng = ay + t * aby
    return haversine(px, py, closest_lat, closest_lng)


def distance_along_route(waypoints, pothole_lat, pothole_lng):
    """Approximate distance in km from route start to the nearest point on route to the pothole."""
    cumulative = 0
    min_dist = float('inf')
    best_km = 0
    for i in range(len(waypoints) - 1):
        seg_dist = point_to_segment_distance(
            pothole_lat, pothole_lng,
            waypoints[i][0], waypoints[i][1],
            waypoints[i+1][0], waypoints[i+1][1]
        )
        if seg_dist < min_dist:
            min_dist = seg_dist
            # Approximate position along segment
            d_to_start = haversine(pothole_lat, pothole_lng, waypoints[i][0], waypoints[i][1])
            seg_len = haversine(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1])
            frac = min(1, d_to_start / seg_len) if seg_len > 0 else 0
            best_km = (cumulative + frac * seg_len) / 1000
        cumulative += haversine(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1])
    return round(best_km, 2)


def find_potholes_along_route(waypoints, corridor_width_m, potholes_rows):
    """Find all potholes within corridor_width of a polyline defined by waypoints."""
    results = []
    for r in potholes_rows:
        plat, plng = r["latitude"], r["longitude"]
        min_dist = float('inf')
        for i in range(len(waypoints) - 1):
            d = point_to_segment_distance(
                plat, plng,
                waypoints[i][0], waypoints[i][1],
                waypoints[i+1][0], waypoints[i+1][1]
            )
            min_dist = min(min_dist, d)
            if min_dist <= corridor_width_m:
                break
        if min_dist <= corridor_width_m:
            results.append({
                "id": r["id"],
                "latitude": plat,
                "longitude": plng,
                "severity": r["severity"],
                "confidence": r["confidence"],
                "description": r["description"],
                "reported_at": r["reported_at"],
                "source": r["source"],
                "distance_from_route_m": round(min_dist, 1),
                "distance_from_start_km": distance_along_route(waypoints, plat, plng),
            })
    # Sort by distance along route
    results.sort(key=lambda x: x["distance_from_start_km"])
    return results


def compute_road_quality_score(route_potholes):
    """Compute 0-100 road quality score. 100 = perfect, 0 = disaster."""
    if not route_potholes:
        return 100
    severity_penalty = {"low": 1, "medium": 3, "high": 7}
    total_penalty = sum(severity_penalty.get(p["severity"], 2) for p in route_potholes)
    # Scale: 10 high-severity potholes = score of 30
    score = max(0, 100 - total_penalty * 1.5)
    return round(score)


def decode_polyline(encoded, precision=5):
    """Decode a Google/OSRM encoded polyline into list of (lat, lng) tuples."""
    inv = 1.0 / (10 ** precision)
    decoded = []
    previous = [0, 0]
    i = 0
    while i < len(encoded):
        for dim in range(2):
            shift = 0
            result = 0
            while True:
                b = ord(encoded[i]) - 63
                i += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            if result & 1:
                result = ~result
            result >>= 1
            previous[dim] += result
        decoded.append((previous[0] * inv, previous[1] * inv))
    return decoded


async def get_osrm_route(start_lat, start_lng, end_lat, end_lng, via_points=None, alternatives=False):
    """
    Fetch actual road geometry from OSRM (free, no API key).
    Returns list of routes, each with waypoints and metadata.
    """
    coords = f"{start_lng},{start_lat};"
    if via_points:
        for vp in via_points:
            coords += f"{vp[1]},{vp[0]};"
    coords += f"{end_lng},{end_lat}"

    url = f"https://router.project-osrm.org/route/v1/driving/{coords}"
    params = {
        "overview": "full",
        "geometries": "polyline",
        "alternatives": "true" if alternatives else "false",
        "steps": "false",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("code") != "Ok":
            return None

    routes = []
    for route in data.get("routes", []):
        waypoints = decode_polyline(route["geometry"])
        routes.append({
            "waypoints": waypoints,
            "distance_km": round(route["distance"] / 1000, 1),
            "duration_min": round(route["duration"] / 60, 1),
        })
    return routes

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


# ─── Route Intelligence API ──────────────────────────────
# IMPORTANT: These must be defined BEFORE /api/potholes/{pothole_id}
# or FastAPI will match "route", "nearby" etc as a pothole_id.

def _build_route_response(route_potholes, route_info=None):
    """Build a standard route response with summary."""
    by_severity = {"low": 0, "medium": 0, "high": 0}
    for p in route_potholes:
        by_severity[p["severity"]] = by_severity.get(p["severity"], 0) + 1

    resp = {
        "route_potholes": route_potholes,
        "summary": {
            "total": len(route_potholes),
            "critical": by_severity.get("high", 0),
            "moderate": by_severity.get("medium", 0),
            "minor": by_severity.get("low", 0),
            "road_quality_score": compute_road_quality_score(route_potholes),
        }
    }
    if route_info:
        resp["route"] = route_info
    return resp


# --- GET /api/potholes/route — OSRM-powered road-based route query ---
@app.get("/api/potholes/route")
async def potholes_along_route(
    start_lat: float = Query(..., ge=-90, le=90),
    start_lng: float = Query(..., ge=-180, le=180),
    end_lat: float = Query(..., ge=-90, le=90),
    end_lng: float = Query(..., ge=-180, le=180),
    corridor_width: float = Query(50, ge=10, le=500, description="Meters from road centerline"),
    use_road: bool = Query(True, description="Use OSRM road geometry (True) or straight line (False)"),
):
    """
    Find all active potholes along the actual road between two points.
    Uses OSRM (free) for real road geometry — follows actual streets, not straight lines.
    """
    conn = get_db()
    rows = conn.execute("SELECT * FROM potholes WHERE status = 'active'").fetchall()
    conn.close()

    if use_road:
        osrm_routes = await get_osrm_route(start_lat, start_lng, end_lat, end_lng)
        if not osrm_routes:
            raise HTTPException(status_code=502, detail="Could not fetch road geometry from OSRM")

        road = osrm_routes[0]
        waypoints = road["waypoints"]
        route_potholes = find_potholes_along_route(waypoints, corridor_width, rows)

        return _build_route_response(route_potholes, route_info={
            "distance_km": road["distance_km"],
            "duration_min": road["duration_min"],
            "waypoint_count": len(waypoints),
            "polyline": [[w[0], w[1]] for w in waypoints],
        })
    else:
        waypoints = [(start_lat, start_lng), (end_lat, end_lng)]
        route_potholes = find_potholes_along_route(waypoints, corridor_width, rows)
        return _build_route_response(route_potholes)


# --- POST /api/potholes/route-polyline — Query with custom polyline ---
@app.post("/api/potholes/route-polyline")
def potholes_along_polyline(body: dict):
    """
    Find potholes along a polyline you provide (e.g. from Apple Maps directions).
    """
    waypoints = body.get("waypoints", [])
    corridor_width = body.get("corridor_width", 50)

    if len(waypoints) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 waypoints")

    waypoints = [(w[0], w[1]) for w in waypoints]

    conn = get_db()
    rows = conn.execute("SELECT * FROM potholes WHERE status = 'active'").fetchall()
    conn.close()

    route_potholes = find_potholes_along_route(waypoints, corridor_width, rows)
    return _build_route_response(route_potholes)


# --- POST /api/potholes/compare-routes — Compare with real road geometry ---
@app.post("/api/potholes/compare-routes")
async def compare_routes(data: RouteCompareRequest):
    """
    Compare pothole density across multiple routes.
    """
    conn = get_db()
    rows = conn.execute("SELECT * FROM potholes WHERE status = 'active'").fetchall()
    conn.close()

    results = []
    for route in data.routes:
        name = route.get("name", "Unnamed")
        waypoints_raw = route.get("waypoints", [])

        if len(waypoints_raw) < 2 and "start" in route and "end" in route:
            s, e = route["start"], route["end"]
            osrm = await get_osrm_route(s[0], s[1], e[0], e[1])
            if osrm:
                waypoints = osrm[0]["waypoints"]
                dist_km = osrm[0]["distance_km"]
                dur_min = osrm[0]["duration_min"]
            else:
                results.append({"name": name, "error": "OSRM route fetch failed"})
                continue
        else:
            if len(waypoints_raw) < 2:
                results.append({"name": name, "error": "Need at least 2 waypoints"})
                continue
            waypoints = [(w[0], w[1]) for w in waypoints_raw]
            dist_km = sum(
                haversine(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1])
                for i in range(len(waypoints) - 1)
            ) / 1000
            dur_min = None

        route_potholes = find_potholes_along_route(waypoints, data.corridor_width, rows)

        by_severity = {"low": 0, "medium": 0, "high": 0}
        for p in route_potholes:
            by_severity[p["severity"]] = by_severity.get(p["severity"], 0) + 1

        results.append({
            "name": name,
            "total_potholes": len(route_potholes),
            "critical": by_severity.get("high", 0),
            "moderate": by_severity.get("medium", 0),
            "minor": by_severity.get("low", 0),
            "road_quality_score": compute_road_quality_score(route_potholes),
            "potholes_per_km": round(len(route_potholes) / max(dist_km, 0.1), 2),
            "route_distance_km": round(dist_km, 1),
            "duration_min": dur_min,
            "potholes": route_potholes,
        })

    results.sort(key=lambda x: x.get("road_quality_score", 0), reverse=True)

    return {
        "recommended": results[0]["name"] if results else None,
        "routes": results,
    }


# --- GET /api/potholes/nearby — Quick proximity check for iOS ---
@app.get("/api/potholes/nearby")
def nearby_potholes(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    radius: float = Query(500, ge=10, le=5000, description="Radius in meters"),
):
    """
    Find potholes near a GPS point. Designed for real-time iOS alerts.
    """
    conn = get_db()
    deg_offset = radius / 111000
    rows = conn.execute(
        """SELECT * FROM potholes WHERE status = 'active'
           AND latitude BETWEEN ? AND ?
           AND longitude BETWEEN ? AND ?""",
        (lat - deg_offset, lat + deg_offset, lng - deg_offset, lng + deg_offset)
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        dist = haversine(lat, lng, r["latitude"], r["longitude"])
        if dist <= radius:
            results.append({
                "id": r["id"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "severity": r["severity"],
                "confidence": r["confidence"],
                "description": r["description"],
                "distance_m": round(dist, 1),
                "bearing": round(math.degrees(math.atan2(
                    r["longitude"] - lng, r["latitude"] - lat
                )) % 360),
            })

    results.sort(key=lambda x: x["distance_m"])

    return {
        "count": len(results),
        "radius_m": radius,
        "center": {"lat": lat, "lng": lng},
        "potholes": results,
    }


# --- GET /api/potholes/:id --- (MUST be after /route, /nearby, /stats etc)
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
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🕳️  PotholePulse API running on port {port}")
    print(f"📡 API Docs:  http://localhost:{port}/docs")
    print(f"🗺️  Map:       http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
