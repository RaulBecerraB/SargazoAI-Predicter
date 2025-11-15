# SargazoAI-Predicter 🌊

Sargassum prediction system using machine learning models to predict coordinate trajectories and sargassum biomass in the Gulf of Mexico and Caribbean Sea.

## 📋 Description

This project provides a REST API built with FastAPI that offers two prediction services:

1. **Coordinate Prediction**: Uses an LSTM model to predict the next position (latitude and longitude) based on a sequence of historical positions.
2. **Biomass Prediction**: Uses an XGBoost model to estimate sargassum biomass based on environmental features.

## 🚀 Features

- REST API with FastAPI
- Two independent prediction models:
  - LSTM (TensorFlow/Keras) for trajectories
  - XGBoost for biomass
- Input validation with Pydantic
- Flexible configuration through JSON files
- Launch scripts for PowerShell and Bash
- Uvicorn server with auto-reload support

## 📁 Project Structure

```
SargazoAI-Predicter/
├── models/
│   ├── biomasa/
│   │   ├── sargassum_xgb_model.pkl       # Trained XGBoost model
│   │   └── sargassum_model_config.json   # Biomass model configuration
│   └── coordinates/
│       ├── sargazo_lstm_model.h5         # Trained LSTM model
│       ├── sargazo_scaler.pkl            # MinMaxScaler for normalization
│       └── sargazo_config.json           # Coordinate model configuration
├── sargazo_predictor_service/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # Main FastAPI application
│   │   ├── predictor.py                  # Coordinate predictor (LSTM)
│   │   └── biomasa_predictor.py          # Biomass predictor (XGBoost)
│   ├── requirements.txt                  # Python dependencies
│   ├── run_uvicorn.ps1                   # PowerShell launch script
│   └── run_uvicorn.sh                    # Bash launch script
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/RaulBecerraB/SargazoAI-Predicter.git
   cd SargazoAI-Predicter
   ```

2. **Create virtual environment (recommended)**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r sargazo_predictor_service/requirements.txt
   ```

## 🎮 Usage

### Start the Server

**PowerShell (Windows):**

```powershell
.\sargazo_predictor_service\run_uvicorn.ps1
```

With auto-reload for development:

```powershell
.\sargazo_predictor_service\run_uvicorn.ps1 -Reload
```

**Bash (Linux/Mac):**

```bash
chmod +x sargazo_predictor_service/run_uvicorn.sh
./sargazo_predictor_service/run_uvicorn.sh
```

The server will be available at: `http://localhost:8000`

### Interactive API Documentation

Once the server is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. Health Check

**GET** `/health`

Checks the service status and predictors.

**Response:**

```json
{
  "status": "ok",
  "coordinate_predictor": "ready",
  "biomasa_predictor": "ready",
  "n_steps": 5
}
```

### 2. Coordinate Prediction

**POST** `/predict-coordinate`

Predicts the next position (lat/lon) based on a sequence of 5 historical positions.

**Request Body:**

```json
{
  "sequence": [
    [18.5, -88.3],
    [18.52, -88.28],
    [18.54, -88.26],
    [18.56, -88.24],
    [18.58, -88.22]
  ]
}
```

**Response:**

```json
{
  "lat_next": 18.60234,
  "lon_next": -88.19876
}
```

**PowerShell Example:**

```powershell
$body = @{
    sequence = @(
        @(18.5, -88.3),
        @(18.52, -88.28),
        @(18.54, -88.26),
        @(18.56, -88.24),
        @(18.58, -88.22)
    )
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://localhost:8000/predict-coordinate" -Method POST -Body $body -ContentType "application/json"
```

**curl Example:**

```bash
curl -X POST "http://localhost:8000/predict-coordinate" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      [18.5, -88.3],
      [18.52, -88.28],
      [18.54, -88.26],
      [18.56, -88.24],
      [18.58, -88.22]
    ]
  }'
```

### 3. Biomass Prediction

**POST** `/predict-biomass`

Predicts sargassum biomass based on environmental features.

**Request Body:**

```json
{
  "lat": 18.5,
  "lon": -88.3,
  "avg_sea_surface_temperature": 28.5,
  "avg_ocean_current_velocity": 0.35,
  "avg_ocean_current_direction": 1.57
}
```

**Response:**

```json
{
  "sargassum_biomass": 245.678
}
```

**PowerShell Example:**

```powershell
$body = @{
    lat = 18.5
    lon = -88.3
    avg_sea_surface_temperature = 28.5
    avg_ocean_current_velocity = 0.35
    avg_ocean_current_direction = 1.57
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict-biomass" -Method POST -Body $body -ContentType "application/json"
```

**curl Example:**

```bash
curl -X POST "http://localhost:8000/predict-biomass" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 18.5,
    "lon": -88.3,
    "avg_sea_surface_temperature": 28.5,
    "avg_ocean_current_velocity": 0.35,
    "avg_ocean_current_direction": 1.57
  }'
```

## 📊 Parameters and Ranges

### Coordinate Prediction

- **sequence**: List of 5 pairs [latitude, longitude]
- **Latitude range**: 17.0 to 22.0 (Caribbean/Gulf of Mexico)
- **Longitude range**: -92.0 to -85.0 (Caribbean/Gulf of Mexico)

### Biomass Prediction

- **lat**: Latitude (17.0 to 22.0)
- **lon**: Longitude (-92.0 to -85.0)
- **avg_sea_surface_temperature**: Sea surface temperature in °C (25.0 to 30.0)
- **avg_ocean_current_velocity**: Ocean current velocity in m/s (0.1 to 0.6)
- **avg_ocean_current_direction**: Current direction in radians (0 to 6.28)

## 🔧 Configuration

### Environment Variables

You can customize model paths using environment variables:

```bash
# Coordinate model directory
export SARGAZO_MODEL_DIR="path/to/models/coordinates"

# Biomass model directory
export SARGAZO_BIOMASA_MODEL_DIR="path/to/models/biomasa"
```

### Configuration Files

**models/coordinates/sargazo_config.json**

```json
{
  "N_STEPS": 5,
  "FEATURES": ["lat", "lon"],
  "TARGETS": ["lat_next", "lon_next"],
  "ALL_COLS": ["lat", "lon"]
}
```

**models/biomasa/sargassum_model_config.json**

```json
{
  "features": [
    "lat",
    "lon",
    "avg_sea_surface_temperature",
    "avg_ocean_current_velocity",
    "avg_ocean_current_direction"
  ],
  "target": "sargassum_biomass",
  "model_type": "XGBRegressor"
}
```

## 📦 Dependencies

- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI server
- **tensorflow**: For LSTM coordinate model
- **xgboost**: For biomass model
- **scikit-learn**: For data preprocessing
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **pydantic**: Data validation
- **joblib**: Model serialization

See `sargazo_predictor_service/requirements.txt` for specific versions.

## 🧪 Testing

To quickly test the service:

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Test coordinate prediction
$coordBody = @{
    sequence = @(@(18.5, -88.3), @(18.52, -88.28), @(18.54, -88.26), @(18.56, -88.24), @(18.58, -88.22))
} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Uri "http://localhost:8000/predict-coordinate" -Method POST -Body $coordBody -ContentType "application/json"

# Test biomass prediction
$biomasaBody = @{
    lat = 18.5
    lon = -88.3
    avg_sea_surface_temperature = 28.5
    avg_ocean_current_velocity = 0.35
    avg_ocean_current_direction = 1.57
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict-biomass" -Method POST -Body $biomasaBody -ContentType "application/json"
```

## 🐛 Troubleshooting

### Error: "Could not deserialize 'keras.metrics.mse'"

The service includes a fallback that loads the LSTM model with `compile=False` if there are deserialization issues with older Keras versions.

### Error: "Predictor not loaded"

Verify that:

1. Model files exist at the correct paths
2. JSON configuration files are valid
3. Paths in environment variables are correct

### Error: "Missing required features"

Ensure that the input JSON includes all required features with exact names.

## 👥 Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Raul Becerra - [@RaulBecerraB](https://github.com/RaulBecerraB)

Project Link: [https://github.com/RaulBecerraB/SargazoAI-Predicter](https://github.com/RaulBecerraB/SargazoAI-Predicter)

## 🙏 Acknowledgments

- LSTM model for trajectory prediction
- XGBoost model for biomass prediction
- FastAPI for the excellent framework
- Open source community

---

**Developed for sargassum prediction and monitoring in the Caribbean and Gulf of Mexico 🌊🌴**
