# Rockfall Prediction System

A comprehensive AI-powered system for predicting and monitoring rockfall risks in real-time. This project combines machine learning models with interactive dashboards and web interfaces to provide early warnings and visualizations for rockfall-prone areas.

## Features

- **Real-Time Risk Monitoring**: Live dashboard with simulated sensor data (vibration, strain, rainfall) updating every few seconds
- **AI Predictions**: LSTM-based machine learning model for rockfall risk assessment
- **Interactive Map View**: Web-based map interface showing risk zones with color-coded markers
- **Alert Logging**: Database-backed system for recording and tracking risk alerts
- **User Authentication**: Secure login system for accessing the dashboard
- **Responsive Design**: Modern, colorful UI with customizable visualizations

## Architecture

The system consists of two main components:

### Backend (Python/Streamlit)
- `app.py`: Main Streamlit application with real-time dashboard
- `detection.py`: Alert logging system using SQLite database
- `rockfall_lstm.h5`: Pre-trained LSTM model for predictions

### Frontend (React/Vite)
- Login interface for user authentication
- Interactive map view using Leaflet
- Risk visualization with color-coded markers
- Responsive design for desktop and mobile

## Technologies Used

### Backend
- Python 3.x
- Streamlit
- TensorFlow/Keras (for LSTM model)
- Pandas, NumPy
- Plotly (for visualizations)
- SQLite (for alert storage)

### Frontend
- React 19
- Vite (build tool)
- React Router DOM
- Leaflet & React-Leaflet (mapping)
- Axios (HTTP client)
- ESLint (code linting)

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rock_fall_predictions.git
cd rock_fall_predictions
```

2. Install Python dependencies:
```bash
pip install streamlit pandas numpy tensorflow plotly streamlit-autorefresh
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser to `http://localhost:5173`

## Usage

### Backend Dashboard
- Access the Streamlit dashboard at the provided URL
- View real-time sensor data visualizations
- Monitor risk predictions with interactive charts
- Customize refresh intervals and thresholds in the sidebar

### Frontend Map View
- Login with any username (no password required for demo)
- View rockfall risk zones on the interactive map
- Click on markers to see detailed risk information
- Map automatically switches between satellite and topographic views based on zoom level

## Model Training

The LSTM model (`rockfall_lstm.h5`) was trained on simulated sensor data including:
- Vibration measurements
- Strain gauge readings
- Rainfall data
- Pore pressure
- Displacement metrics
- NDVI (vegetation index)

## Alert System

The detection module logs alerts to a local SQLite database with the following risk levels:
- HIGH: Immediate danger, evacuation recommended
- MEDIUM: Monitor closely, prepare contingency plans
- LOW: Normal conditions, routine monitoring

## Configuration

### Backend Configuration
- `SEQ_LENGTH`: Sequence length for LSTM predictions (default: 10)
- `AUTOREFRESH_INTERVAL_MS`: Dashboard refresh interval (default: 2000ms)
- Color palette can be customized in the `COLORS` dictionary

### Frontend Configuration
- Map center coordinates and zoom levels can be adjusted in `MapView.jsx`
- Risk thresholds and colors are defined in the component

## Development

### Running Tests
```bash
# Backend tests (if available)
python -m pytest

# Frontend linting
cd frontend
npm run lint
```

### Building for Production
```bash
# Frontend build
cd frontend
npm run build
npm run preview
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built with Streamlit for rapid dashboard development
- Mapping powered by OpenStreetMap and OpenTopoMap
- Icons and styling inspired by modern data visualization practices
