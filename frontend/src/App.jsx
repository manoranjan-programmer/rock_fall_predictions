import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./Login";
import MapView from "./MapView";

export default function App() {
  const [user, setUser] = React.useState(null);

  return (
    <Routes>
      {/* Default route: if logged in, go to map, else login */}
      <Route
        path="/"
        element={user ? <Navigate to="/map" /> : <Login onLogin={setUser} />}
      />

      {/* Map route */}
      <Route
        path="/map"
        element={
          user ? (
            <div style={{ display: "flex", height: "100vh" }}>
              <div style={{ flex: 1 }}>
                <MapView />
              </div>
              <div style={{ width: 300, padding: 20, background: "#f5f5f5" }}>
                <h3>Welcome, {user}</h3>
                <p>Rockfall Risk Dashboard</p>
              </div>
            </div>
          ) : (
            <Navigate to="/" />
          )
        }
      />
    </Routes>
  );
}
