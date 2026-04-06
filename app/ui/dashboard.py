import flet as ft
import os
import threading
import sys
import contextlib
import time
from app.core.pipeline import run_registration_pipeline

def create_dashboard(page: ft.Page):
    """
    Creates the main dashboard UI for the ARIAKE OCTA Registration Tool.
    
    Flet v0.84.0 API Notes:
    - FilePicker.get_directory_path() is now synchronous and returns str|None directly.
    - No on_result / on_select callbacks exist anymore.
    - The picker must be added to page.overlay before use.
    """
    
    # --- State Variables ---
    input_path = ft.Text("No directory selected", color=ft.Colors.GREY_400)
    output_path = ft.Text("No directory selected", color=ft.Colors.GREY_400)
    progress_bar = ft.ProgressBar(width=None, value=0, color=ft.Colors.CYAN_400, bgcolor=ft.Colors.GREY_800)
    progress_text = ft.Text("Ready", size=14, color=ft.Colors.CYAN_200)
    log_messages = ft.ListView(expand=True, spacing=5, padding=10, auto_scroll=True)
    
    # --- File Pickers (v0.84: Service, not visual control) ---
    input_picker = ft.FilePicker()
    output_picker = ft.FilePicker()
    page.services.append(input_picker)
    page.services.append(output_picker)

    # --- Helper Functions ---
    def add_log(message: str, color=ft.Colors.WHITE):
        if len(log_messages.controls) > 150:
            log_messages.controls.pop(0)
        log_messages.controls.append(ft.Text(message, color=color, size=13, font_family="Consolas"))
        log_messages.update()
        page.update()

    class JournalRedirector:
        def __init__(self, color=ft.Colors.CYAN_200):
            self.color = color
        def write(self, data):
            if data and data.strip():
                # Add terminal lines to Journal
                add_log(f"  > {data.strip()}", color=self.color)
        def flush(self):
            pass

    async def select_input_dir(e):
        """Opens the native directory picker for the input folder."""
        result = await input_picker.get_directory_path(dialog_title="Select Input Directory")
        if result:
            input_path.value = result
            input_path.update()
            add_log(f"Input directory set to: {result}")

    async def select_output_dir(e):
        """Opens the native directory picker for the output folder."""
        result = await output_picker.get_directory_path(dialog_title="Select Output Directory")
        if result:
            output_path.value = result
            output_path.update()
            add_log(f"Output directory set to: {result}")

    # --- UI Components ---
    
    # Header Card
    header = ft.Container(
        content=ft.Column([
            ft.Text("ARIAKE OCTA Registration", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.CYAN_400),
            ft.Text("Professional Stack Registration & Averaging Utility", size=16, color=ft.Colors.GREY_400),
        ]),
        padding=20,
    )

    # Directory Selection Cards
    path_card = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.FOLDER_OPEN, color=ft.Colors.CYAN_400),
                    title=ft.Text("Input Directory"),
                    subtitle=input_path,
                    trailing=ft.ElevatedButton("Select", on_click=select_input_dir),
                ),
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.DRIVE_FILE_MOVE, color=ft.Colors.AMBER_400),
                    title=ft.Text("Output Directory"),
                    subtitle=output_path,
                    trailing=ft.ElevatedButton("Select", on_click=select_output_dir),
                ),
            ], spacing=10),
            padding=15,
        ),
        margin=ft.Margin.all(10),
    )

    # Settings Section
    clahe_layer1_switch = ft.Switch(label="Apply CLAHE to Layer 1 (Standard Reference)", value=False)
    auto_tuning_switch = ft.Switch(label="Automate Parameter Tuning", value=True)
    
    settings_card = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Text("Processing Settings", size=18, weight=ft.FontWeight.BOLD),
                clahe_layer1_switch,
                auto_tuning_switch,
            ], spacing=10),
            padding=15,
        ),
        margin=ft.Margin.all(10),
    )

    def start_processing(e):
        if input_path.value == "No directory selected" or output_path.value == "No directory selected":
            page.snack_bar = ft.SnackBar(ft.Text("Please select both input and output directories!"))
            page.snack_bar.open = True
            page.update()
            return
        
        # Capture current switch states
        apply_clahe_val = clahe_layer1_switch.value
        auto_tuning_val = auto_tuning_switch.value
        
        # Start the pipeline in a background thread
        def run_thread():
            def progress_cb(val, status):
                progress_bar.value = val
                progress_text.value = status
                progress_bar.update()
                progress_text.update()
                page.update()

            def log_cb(msg):
                add_log(msg)
                page.update()

            # Capture stdout (print statements from core) into the journal
            with contextlib.redirect_stdout(JournalRedirector()):
                success = run_registration_pipeline(
                    input_path.value, 
                    output_dir=output_path.value, 
                    apply_clahe_to_ref=apply_clahe_val,
                    automate_tuning=auto_tuning_val,
                    progress_callback=progress_cb, 
                    log_callback=log_cb
                )
            
            if success:
                add_log("--- All Tasks Completed! ---", color=ft.Colors.GREEN_400)
                page.snack_bar = ft.SnackBar(ft.Text("Registration successful!"))
                page.snack_bar.open = True
            else:
                add_log("--- Error Occurred! ---", color=ft.Colors.RED_400)
            
            page.update()

        thread = threading.Thread(target=run_thread, daemon=True)
        thread.start()

    # Progress & Logs Section
    log_card = ft.Container(
        content=ft.Column([
            ft.Text("Processing Journal", size=18, weight=ft.FontWeight.BOLD),
            ft.Container(
                content=log_messages,
                bgcolor=ft.Colors.BLACK,
                border_radius=10,
                border=ft.Border.all(1, ft.Colors.GREY_800),
                height=250,
            ),
            ft.Row([progress_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            progress_bar,
            ft.ElevatedButton(
                "Run Registration", 
                icon=ft.Icons.PLAY_ARROW, 
                style=ft.ButtonStyle(bgcolor=ft.Colors.CYAN_700, color=ft.Colors.WHITE),
                on_click=start_processing,
                height=50,
            )
        ], spacing=15),
        padding=20,
        expand=True,
    )

    # Assemble main layout
    layout = ft.Column([
        header,
        ft.Row([
            ft.Column([path_card, settings_card], expand=4),
            ft.Column([log_card], expand=6),
        ], expand=True),
    ], expand=True, spacing=0)

    return layout
