#!/bin/bash
cd "$(dirname "$0")"
echo "======================================================="
echo "🩺 Launching WoundCare Web Application..."
echo "👉 Web URL: http://localhost:5000"
echo "======================================================="
/opt/anaconda3/bin/python3 server.py
