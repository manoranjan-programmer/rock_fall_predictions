import React, { useState, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  LayersControl,
  LayerGroup,
  Marker,
  Popup,
  Circle,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./index.css";

const rockfallPoints = [
  { lat: 20.595, lng: 78.96, risk: "HIGH", rockType: "Boulder Cluster" },
  { lat: 20.6, lng: 78.97, risk: "MEDIUM", rockType: "Loose Rocks" },
  { lat: 20.585, lng: 78.965, risk: "LOW", rockType: "Small Stones" },
];

const riskColor = (risk) => {
  if (risk === "HIGH") return "red";
  if (risk === "MEDIUM") return "orange";
  return "green";
};

// Component to switch base layer based on zoom
function DynamicBaseLayer() {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());

  useEffect(() => {
    const onZoom = () => {
      setZoom(map.getZoom());
    };
    map.on("zoomend", onZoom);
    return () => {
      map.off("zoomend", onZoom);
    };
  }, [map]);

  // Define zoom threshold to switch tile layers
  const zoomThreshold = 13;

  return zoom >= zoomThreshold ? (
    <TileLayer
      url="https://tile.opentopomap.org/{z}/{x}/{y}.png"
      attribution='Map data: &copy; <a href="https://opentopomap.org">OpenTopoMap</a>'
    />
  ) : (
    <TileLayer
      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      attribution='&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors'
    />
  );
}

export default function MapView() {
  return (
    <MapContainer
      center={[20.5937, 78.9629]}
      zoom={14}
      minZoom={5}
      maxZoom={18}
      className="map-container leaflet-map"
      style={{ height: "100vh", width: "100%" }}
    >
      <LayersControl position="topright">
        {/* Instead of static base layers, use dynamic base layer */}
        <LayersControl.BaseLayer checked name="Dynamic Base Layer">
          <DynamicBaseLayer />
        </LayersControl.BaseLayer>

        {/* You can keep other base layers if you want */}
        <LayersControl.BaseLayer name="OpenStreetMap">
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer name="Terrain">
          <TileLayer url="https://tile.opentopomap.org/{z}/{x}/{y}.png" />
        </LayersControl.BaseLayer>

        {/* Rockfall points overlay */}
        <LayersControl.Overlay checked name="Rockfall Zones">
          <LayerGroup>
            {rockfallPoints.map((point, idx) => (
              <Circle
                key={`circle-${idx}`}
                center={[point.lat, point.lng]}
                radius={50}
                pathOptions={{
                  color: riskColor(point.risk),
                  fillColor: riskColor(point.risk),
                  fillOpacity: 0.5,
                }}
              />
            ))}
            {rockfallPoints.map((point, idx) => (
              <Marker key={`marker-${idx}`} position={[point.lat, point.lng]}>
                <Popup>
                  <strong>Risk:</strong> {point.risk} <br />
                  <strong>Rock Type:</strong> {point.rockType}
                </Popup>
              </Marker>
            ))}
          </LayerGroup>
        </LayersControl.Overlay>
      </LayersControl>
    </MapContainer>
  );
}