# ARIAKE OCTA Stack Registration Tool

A professional-grade desktop application for registering and averaging retinal OCTA image stacks. This tool is a direct Python translation of the ARIAKE ImageJ macro, optimized for standalone performance and ease of use.

## Features
- **Strict Parity**: Matches ImageJ's preprocessing, 4x enlargement, and CLAHE optimization exactly.
- **Reference-Based Alignment**: Uses the "Stack of Averages" (Image 5) method for high-precision Affine registration.
- **Data Validation**: Automatically verifies file order and consistency across all patient folders.
- **Premium UI**: Built with Flet (Flutter) for a modern, responsive, and cross-platform experience.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stack_reg
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the app in development mode:
```bash
python -m app.main
```

## Packaging for Desktop (.exe / .dmg)

This application uses `flet build` to create native standalone executables.

### Windows (.exe)
```bash
flet build windows
```

### macOS (.dmg)
```bash
flet build macos
```

The output will be located in the `build/` directory.

---
Developed by Team Yanagi (2025/2026)
