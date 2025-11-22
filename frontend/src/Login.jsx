import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    // Dummy login check
    if (username === "admin" && password === "password123") {
      onLogin(username);
      navigate("/map"); // Navigate to MapView after login
    } else {
      setError("Invalid username or password");
    }
  };

  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh", background: "#f0f0f0" }}>
      <div style={{ background: "#fff", padding: 40, borderRadius: 12, boxShadow: "0 4px 10px rgba(0,0,0,0.1)", width: 320 }}>
        <h2 style={{ textAlign: "center", marginBottom: 20 }}>Login</h2>
        {error && <div style={{ color: "red", marginBottom: 10 }}>{error}</div>}
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            style={{ width: "100%", padding: 10, marginBottom: 10, borderRadius: 6, border: "1px solid #ccc" }}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ width: "100%", padding: 10, marginBottom: 10, borderRadius: 6, border: "1px solid #ccc" }}
          />
          <button type="submit" style={{ width: "100%", padding: 10, borderRadius: 6, background: "#007bff", color: "#fff", fontWeight: "bold" }}>
            Login
          </button>
        </form>
      </div>
    </div>
  );
}
