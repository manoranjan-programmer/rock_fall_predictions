import sqlite3

def log_alert(risk_level, details):
    # Connect or create DB
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_level TEXT,
            details TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert new alert
    cursor.execute("INSERT INTO alerts (risk_level, details) VALUES (?, ?)",
                   (risk_level, details))
    conn.commit()
    conn.close()

# ---- Example usage ----
if __name__ == "__main__":
    log_alert("high", "Medical waste found in Zone 3 bin")
    log_alert("medium", "Plastic bottle in organic bin")
    log_alert("low", "Paper in mixed waste bin")
    print("✅ Alerts logged successfully!")
