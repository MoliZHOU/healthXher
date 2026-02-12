# healthXher Frontend (React + Vite)

The "Face" of the healthXher application. A modern, clinical dashboard for RA risk assessment.

## Features
- **Clinical Dashboard:** Real-time clinical data entry for RA risk.
- **User Management:** Secure Login and Registration flows.
- **Risk Visualization:** Semi-circle gauge (Risk Probability) with clinical risk level labeling.
- **Future Roadmap:** Grayscale placeholders for MRI Computer Vision and EMR Sync.

## Project Structure
- `src/api/`: API client and configuration.
- `src/components/`: Reusable UI components (Forms, Gauges).
- `src/pages/`: Main application views (Login, Dashboard).
- `src/App.jsx`: Routing and Auth Guards.

## Development
```bash
npm install
npm run dev
```

## Production Build
```bash
npm run build
```
The output will be in the `dist/` folder, ready to be served by any static file host or integrated with the FastAPI backend.
