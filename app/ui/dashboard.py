import flet as ft
import contextlib
import os
import time
from app.core.pipeline import (
    run_registration_pipeline,
    discover_visits,
    prepare_visit,
    finalize_visit,
)
from app.core.registration import DEFAULT_CONFIDENCE_THRESHOLD
from app.core.manual_align import save_session
from app.ui.manual_align_view import create_manual_align_view
from app.ui.results_view import create_results_view

def create_dashboard(page: ft.Page, mount_view=None):
    """
    Creates the main dashboard UI for the ARIAKE OCTA Registration Tool.

    ``mount_view`` (optional) swaps the app's root content to another control,
    enabling the manual corresponding-point correction screen. When provided, a
    "Review & Correct" action is exposed alongside the automatic run.

    Flet v0.84.0 API Notes:
    - FilePicker.get_directory_path() is now synchronous and returns str|None directly.
    - No on_result / on_select callbacks exist anymore.
    - The picker must be added to page.overlay before use.
    """
    layout_ref = {}  # holds the dashboard layout so we can navigate back to it
    
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

    # --- Helper Functions (UI updates must run on the page event loop — Flet 0.84+) ---
    def append_log_line(message: str, color=ft.Colors.WHITE):
        if len(log_messages.controls) > 150:
            log_messages.controls.pop(0)
        log_messages.controls.append(
            ft.Text(message, color=color, size=13, font_family="Consolas")
        )

    async def ui_log(message: str, color=ft.Colors.WHITE):
        append_log_line(message, color)
        log_messages.update()

    async def ui_progress(val: float, status: str):
        progress_bar.value = val
        progress_text.value = status
        progress_bar.update()
        progress_text.update()

    async def ui_complete(success: bool):
        if success:
            append_log_line("--- All Tasks Completed! ---", color=ft.Colors.GREEN_400)
            page.snack_bar = ft.SnackBar(ft.Text("Registration successful!"))
            page.snack_bar.open = True
        else:
            append_log_line("--- Error Occurred! ---", color=ft.Colors.RED_400)
        log_messages.update()
        progress_bar.update()
        progress_text.update()

    _last_progress_ts = [0.0]

    def schedule_log(message: str, color=ft.Colors.WHITE):
        page.run_task(ui_log, message, color=color)

    def schedule_progress(val: float, status: str):
        now = time.monotonic()
        if val < 1.0 and (now - _last_progress_ts[0]) < 0.15:
            return
        _last_progress_ts[0] = now
        page.run_task(ui_progress, val, status)

    def add_log(message: str, color=ft.Colors.WHITE):
        schedule_log(message, color=color)

    class JournalRedirector:
        def __init__(self, color=ft.Colors.CYAN_200):
            self.color = color
        def write(self, data):
            if data and data.strip():
                schedule_log(f"  > {data.strip()}", color=self.color)
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
        
        _last_progress_ts[0] = 0.0
        schedule_log("--- Starting registration ---")

        def run_pipeline():
            success = False
            try:
                with contextlib.redirect_stdout(JournalRedirector()):
                    success = run_registration_pipeline(
                        input_path.value,
                        output_dir=output_path.value,
                        apply_clahe_to_ref=apply_clahe_val,
                        automate_tuning=auto_tuning_val,
                        progress_callback=schedule_progress,
                        log_callback=schedule_log,
                    )
            except Exception as exc:
                schedule_log(f"ERROR: {exc}", color=ft.Colors.RED_400)
            page.run_task(ui_complete, success)

        page.run_thread(run_pipeline)

    # --- Manual corresponding-point review workflow ---
    review_state = {"plans": []}

    def go_dashboard():
        if mount_view and layout_ref.get("layout") is not None:
            mount_view(layout_ref["layout"])

    async def open_manual_view(plans):
        review_state["plans"] = plans
        total_low = sum(len(p.low_confidence_indices()) for p in plans)
        append_log_line(
            f"Analysis complete: {len(plans)} visit(s). "
            f"{total_low} capture(s) flagged for review.",
            color=ft.Colors.AMBER_200 if total_low else ft.Colors.GREEN_400,
        )
        log_messages.update()
        view = create_manual_align_view(
            page,
            plans,
            on_back=go_dashboard,
            on_finalize=handle_finalize,
            threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        if mount_view:
            mount_view(view)

    def start_review(e):
        if input_path.value == "No directory selected" or output_path.value == "No directory selected":
            page.snack_bar = ft.SnackBar(ft.Text("Please select both input and output directories!"))
            page.snack_bar.open = True
            page.update()
            return
        _last_progress_ts[0] = 0.0
        schedule_log("--- Analyzing for review (auto-alignment) ---")

        def work():
            try:
                visits = discover_visits(input_path.value)
                if not visits:
                    schedule_log("ERROR: No valid Visit folders found.", color=ft.Colors.RED_400)
                    page.run_task(ui_complete, False)
                    return
                plans = []
                total = len(visits)
                for v_idx, vd in enumerate(visits):
                    def prep_cb(val, status, _b=v_idx, _t=total):
                        schedule_progress((_b + val) / _t, status)
                    with contextlib.redirect_stdout(JournalRedirector()):
                        plan = prepare_visit(vd, progress_callback=prep_cb, log_callback=schedule_log)
                    plans.append(plan)
                page.run_task(open_manual_view, plans)
            except Exception as exc:
                schedule_log(f"ERROR: {exc}", color=ft.Colors.RED_400)
                page.run_task(ui_complete, False)

        page.run_thread(work)

    async def show_results(patient_output_dir, corrections_summary):
        view = create_results_view(
            page,
            patient_output_dir,
            on_back=go_dashboard,
            corrections_summary=corrections_summary,
        )
        if mount_view:
            mount_view(view)

    def handle_finalize(overrides_by_visit, points_by_visit):
        plans = review_state["plans"]
        if not plans:
            return
        apply_clahe_val = clahe_layer1_switch.value
        auto_tuning_val = auto_tuning_switch.value
        input_dir = input_path.value
        output_dir = output_path.value
        _last_progress_ts[0] = 0.0
        go_dashboard()
        schedule_log("--- Finalizing with manual corrections ---")

        def work():
            success = False
            try:
                patient_name = os.path.basename(input_dir.rstrip(os.sep))
                patient_output_dir = os.path.join(output_dir, patient_name)
                os.makedirs(patient_output_dir, exist_ok=True)

                session = {
                    "patient": patient_name,
                    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                    "visits": {},
                }
                total = len(plans)
                with contextlib.redirect_stdout(JournalRedirector()):
                    for v_idx, plan in enumerate(plans):
                        ov = overrides_by_visit.get(plan.visit_name)

                        def fin_cb(val, status, _b=v_idx, _t=total):
                            schedule_progress((_b + val) / _t, status)

                        ok = finalize_visit(
                            plan,
                            patient_name=patient_name,
                            patient_output_dir=patient_output_dir,
                            apply_clahe_to_ref=apply_clahe_val,
                            automate_tuning=auto_tuning_val,
                            matrix_overrides=ov,
                            progress_callback=fin_cb,
                            log_callback=schedule_log,
                        )
                        session["visits"][plan.visit_name] = {
                            "scores": list(plan.scores),
                            "auto_matrices": [m.tolist() for m in plan.matrices],
                            "overrides": {str(k): v.tolist() for k, v in (ov or {}).items()},
                            "points": {
                                str(k): pt for k, pt in points_by_visit.get(plan.visit_name, {}).items()
                            },
                        }
                        if not ok:
                            schedule_log(f"ERROR finalizing visit {plan.visit_name}.", color=ft.Colors.RED_400)
                            break
                    else:
                        success = True
                save_session(patient_output_dir, session)
                schedule_log(f"Saved alignment session to {patient_output_dir}.")
            except Exception as exc:
                schedule_log(f"ERROR: {exc}", color=ft.Colors.RED_400)
            page.run_task(ui_complete, success)
            if success:
                corrections = {
                    v: sorted(d.keys()) for v, d in (overrides_by_visit or {}).items() if d
                }
                page.run_task(show_results, patient_output_dir, corrections)

        page.run_thread(work)

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
            ft.Row([
                ft.ElevatedButton(
                    "Run Registration",
                    icon=ft.Icons.PLAY_ARROW,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.CYAN_700, color=ft.Colors.WHITE),
                    on_click=start_processing,
                    height=50,
                    expand=True,
                ),
                ft.OutlinedButton(
                    "Review & Correct",
                    icon=ft.Icons.TUNE,
                    tooltip="Analyze alignment, then manually fix low-confidence captures "
                            "with corresponding points before finalizing.",
                    on_click=start_review,
                    height=50,
                    expand=True,
                    visible=mount_view is not None,
                ),
            ], spacing=12),
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

    layout_ref["layout"] = layout
    return layout
