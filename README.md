# MediSign - USL Healthcare Assistant

A clinical screening system with 3D avatar-based Ugandan Sign Language (USL) translation for healthcare communication.

## Features

- Real-time USL processing with Graph Attention Networks
- 3D avatar synthesis for medical text-to-USL translation
- Clinical screening with FHIR-structured results
- Multiple avatar types with anatomical accuracy
- Offline-first privacy-focused design

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run streamlit_app.py
```

## Avatar Types

- **Perfect Human (263 Specs)**: Most advanced with complete anatomical structure
- **Anatomical (130+ Joints)**: Advanced with detailed finger anatomy
- **Connected Joints**: Avatar with proper joint connections
- **Realistic Human**: Intermediate avatar with better proportions
- **Basic Textured**: Simple textured avatar with basic gestures

## Usage

1. Launch the Streamlit application
2. Configure patient information in the sidebar
3. Use the avatar system for text-to-USL translation
4. Process USL input for clinical screening
5. Generate FHIR-structured clinical results

## System Requirements

- Python 3.8+
- Modern web browser with WebGL support
- Camera access for real-time USL processing