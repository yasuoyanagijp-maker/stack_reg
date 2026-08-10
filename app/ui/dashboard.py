import flet as ft
import asyncio
import contextlib
import os
import time
from app.core.pipeline import (
    run_registration_pipeline,
    finalize_visit,
)
from app.ui.manual_align_view import create_manual_align_view
from app.ui.results_view import create_results_view

def create_dashboard(page: ft.Page, mount_view=None):
    """
    Creates the main dashboard UI for the ARIAKE OCTA Registration Tool.

    ``mount_view`` (optional) swaps the app's root content to another control,
    enabling the post-run review screen and manual corresponding-point editor.

    Flet v0.84.0 API Notes:
    - FilePicker.get_directory_path() is now synchronous and returns str|None directly.
    - No on_result / on_select callbacks exist anymore.
    - The picker must be added to page.services before use.
    """
    layout_ref = {}  # holds the dashboard layout so we can navigate back to it

    # --- State Variables ---
    input_path = ft.Text("No directory selected", color=ft.Colors.GREY_400)
    output_path = ft.Text("No directory selected", color=ft.Colors.GREY_400)
    progress_bar = ft.ProgressBar(width=None, value=0, color=ft.Colors.CYAN_400, bgcolor=ft.Colors.GREY_800)
    progress_text = ft.Text("Ready", size=14, color=ft.Colors.CYAN_200)
    log_messages = ft.ListView(expand=True, spacing=5, padding=10, auto_scroll=True)

    # Session state shared across run → review → manual → finalize.
    # ``plans`` always holds the full VisitPlan list from the last Run Registration.
    # Manual opens may filter to one visit for editing — never overwrite ``plans``
    # with that filtered list (second Review & Correct needs the full session).
    review_state = {
        "plans": [],
        "patient_output_dir": None,
        "focus_layer": None,
        "focus_visit": None,
        "last_corrections": None,
    }
    # Bumps on each manual navigation so a slower preload cannot mount a stale view.
    _manual_nav_gen = [0]

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
        try:
            await log_messages.scroll_to(offset=-1, duration=0)
        except Exception:
            pass

    async def ui_progress(val: float, status: str):
        progress_bar.value = val
        progress_text.value = status
        progress_bar.update()
        progress_text.update()

    async def ui_complete(success: bool, show_review: bool = False):
        if success:
            append_log_line("--- All Tasks Completed! ---", color=ft.Colors.GREEN_400)
            page.snack_bar = ft.SnackBar(ft.Text("Registration successful!"))
            page.snack_bar.open = True
        else:
            append_log_line("--- Error Occurred! ---", color=ft.Colors.RED_400)
        log_messages.update()
        try:
            await log_messages.scroll_to(offset=-1, duration=0)
        except Exception:
            pass
        progress_bar.update()
        progress_text.update()
        if success and show_review and review_state.get("patient_output_dir"):
            await show_results(
                review_state["patient_output_dir"],
                review_state.get("last_corrections"),
                initial_image_num=(
                    (review_state["focus_layer"] + 1)
                    if review_state.get("focus_layer") is not None
                    else None
                ),
            )

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

    def go_dashboard():
        if mount_view and layout_ref.get("layout") is not None:
            mount_view(layout_ref["layout"])

    async def show_results(patient_output_dir, corrections_summary=None, initial_image_num=None):
        review_state["patient_output_dir"] = patient_output_dir
        review_state["last_corrections"] = corrections_summary

        def on_review_correct(image_num, visit_name):
            start_manual_for_image(image_num, visit_name)

        view = create_results_view(
            page,
            patient_output_dir,
            on_back=go_dashboard,
            on_review_correct=on_review_correct if mount_view else None,
            corrections_summary=corrections_summary,
            initial_image_num=initial_image_num,
        )
        if mount_view:
            mount_view(view)

    def start_processing(e):
        if input_path.value == "No directory selected" or output_path.value == "No directory selected":
            page.snack_bar = ft.SnackBar(ft.Text("Please select both input and output directories!"))
            page.snack_bar.open = True
            page.update()
            return

        apply_clahe_val = clahe_layer1_switch.value
        auto_tuning_val = auto_tuning_switch.value
        input_dir = input_path.value
        output_dir = output_path.value

        _last_progress_ts[0] = 0.0
        schedule_log("--- Starting registration ---")
        review_state["focus_layer"] = None
        review_state["last_corrections"] = None
        review_state["plans"] = []

        def run_pipeline():
            success = False
            try:
                with contextlib.redirect_stdout(JournalRedirector()):
                    plans = run_registration_pipeline(
                        input_dir,
                        output_dir=output_dir,
                        apply_clahe_to_ref=apply_clahe_val,
                        automate_tuning=auto_tuning_val,
                        progress_callback=schedule_progress,
                        log_callback=schedule_log,
                    )
                if plans:
                    success = True
                    review_state["plans"] = plans
                    patient_name = os.path.basename(input_dir.rstrip(os.sep))
                    review_state["patient_output_dir"] = os.path.join(output_dir, patient_name)
            except Exception as exc:
                schedule_log(f"ERROR: {exc}", color=ft.Colors.RED_400)
            page.run_task(ui_complete, success, True)

        page.run_thread(run_pipeline)

    def _loading_view(message: str):
        return ft.Column(
            [
                ft.Container(expand=True),
                ft.Row(
                    [
                        ft.ProgressRing(width=36, height=36, stroke_width=3),
                        ft.Text(message, size=16, color=ft.Colors.CYAN_200),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=16,
                ),
                ft.Container(expand=True),
            ],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    async def open_manual_view(edit_plans, focus_layer):
        """
        Navigate to the manual editor for ``edit_plans`` / ``focus_layer``.

        Important: the dashboard (and its journal ListView) is unmounted while
        the results screen is showing. Do **not** call log_messages.update()
        here — that used to abort navigation silently on Flet 0.84.

        Does **not** replace ``review_state["plans"]`` with the (possibly
        single-visit) ``edit_plans`` list — that caused the second Review &
        Correct to lose visits / reuse a stale focus context.
        """
        review_state["focus_layer"] = focus_layer
        _manual_nav_gen[0] += 1
        nav_token = _manual_nav_gen[0]

        if not mount_view:
            page.snack_bar = ft.SnackBar(
                ft.Text("View host is not available; restart the app.")
            )
            page.snack_bar.open = True
            page.update()
            return

        image_label = f"image{focus_layer + 1}" if focus_layer is not None else "reference"
        mount_view(_loading_view(f"Opening manual registration ({image_label})…"))

        def back_to_results():
            out = review_state.get("patient_output_dir")
            # Use the current focus from session state (not a stale closure).
            fl = review_state.get("focus_layer")
            if out and mount_view:
                page.run_task(
                    show_results,
                    out,
                    review_state.get("last_corrections"),
                    fl + 1 if fl is not None else None,
                )
            else:
                go_dashboard()

        def preload():
            stacks = {}
            for plan in edit_plans:
                if focus_layer is not None:
                    stacks[plan.visit_name] = plan.load_layer_stack(focus_layer)
                else:
                    stacks[plan.visit_name] = plan.ref_stack
            return stacks

        try:
            preloaded = await asyncio.to_thread(preload)
            if nav_token != _manual_nav_gen[0]:
                return  # a newer Review & Correct superseded this open
            view = create_manual_align_view(
                page,
                edit_plans,
                on_back=back_to_results,
                on_finalize=handle_finalize,
                focus_layer=focus_layer,
                preloaded_stacks=preloaded,
            )
            if nav_token != _manual_nav_gen[0]:
                return
            mount_view(view)
        except Exception as exc:
            if nav_token != _manual_nav_gen[0]:
                return
            page.snack_bar = ft.SnackBar(
                ft.Text(f"Failed to open manual registration: {exc}")
            )
            page.snack_bar.open = True
            out = review_state.get("patient_output_dir")
            fl = review_state.get("focus_layer")
            if out:
                await show_results(
                    out,
                    review_state.get("last_corrections"),
                    fl + 1 if fl is not None else None,
                )
            else:
                go_dashboard()

    def start_manual_for_image(image_num: int, visit_name):
        """
        Open manual corresponding-point correction for the selected result image.

        Does **not** re-run automatic slice alignment. Uses the VisitPlan(s)
        retained from the completed Run Registration.
        """
        try:
            image_num = int(image_num)
        except (TypeError, ValueError):
            page.snack_bar = ft.SnackBar(
                ft.Text("Invalid result image selection.")
            )
            page.snack_bar.open = True
            try:
                page.update()
            except Exception:
                pass
            return

        session_plans = list(review_state.get("plans") or [])
        if not session_plans:
            page.snack_bar = ft.SnackBar(
                ft.Text(
                    "Session plans are missing. Run Registration again in this "
                    "session, then press Review & Correct."
                )
            )
            page.snack_bar.open = True
            try:
                page.update()
            except Exception:
                pass
            return

        focus_layer = image_num - 1  # image1 → layer 0
        for plan in session_plans:
            n_layers = len(plan.folder_contents[plan.sorted_captures[0]])
            if focus_layer < 0 or focus_layer >= n_layers:
                page.snack_bar = ft.SnackBar(
                    ft.Text(
                        f"image{image_num} is out of range "
                        f"({n_layers} layer(s) in {plan.visit_name})."
                    )
                )
                page.snack_bar.open = True
                try:
                    page.update()
                except Exception:
                    pass
                return

        # Edit only the visit matching the reviewed result (do not touch other visits).
        edit_plans = session_plans
        if visit_name:
            preferred = [p for p in session_plans if p.visit_name == visit_name]
            if preferred:
                edit_plans = preferred

        review_state["focus_layer"] = focus_layer
        review_state["focus_visit"] = visit_name
        # Do not schedule_log here: journal is unmounted on the results screen.
        page.run_task(open_manual_view, edit_plans, focus_layer)

    def handle_finalize(overrides_by_visit, points_by_visit, excluded_by_visit=None):
        plans = review_state["plans"]
        if not plans:
            return
        apply_clahe_val = clahe_layer1_switch.value
        auto_tuning_val = auto_tuning_switch.value
        input_dir = input_path.value
        output_dir = output_path.value
        focus_layer = review_state.get("focus_layer")
        focus_visit = review_state.get("focus_visit")
        excluded_by_visit = excluded_by_visit or {}
        _last_progress_ts[0] = 0.0
        go_dashboard()
        focus_label = (
            f"image{focus_layer + 1}" if focus_layer is not None else "manual points"
        )
        # Same Visit: rewrite all result images (image1–N) with these capture matrices.
        # Other Visits are never re-finalized here.
        schedule_log(
            f"--- Finalizing: applying {focus_label} registration params "
            "to all result images of this Visit only (other Visits untouched) ---"
        )

        def work():
            success = False
            patient_output_dir = None
            try:
                patient_name = os.path.basename(input_dir.rstrip(os.sep))
                patient_output_dir = os.path.join(output_dir, patient_name)
                os.makedirs(patient_output_dir, exist_ok=True)
                review_state["patient_output_dir"] = patient_output_dir

                # Only the Visit being edited — never re-synthesize sibling Visits.
                edited_names = set(overrides_by_visit or {}) | set(excluded_by_visit or {})
                if focus_visit:
                    plans_to_run = [p for p in plans if p.visit_name == focus_visit]
                elif edited_names:
                    plans_to_run = [p for p in plans if p.visit_name in edited_names]
                else:
                    plans_to_run = list(plans)[:1]

                if not plans_to_run:
                    schedule_log("ERROR: No Visit selected to finalize.", color=ft.Colors.RED_400)
                    page.run_task(ui_complete, False, False)
                    return

                total = len(plans_to_run)
                with contextlib.redirect_stdout(JournalRedirector()):
                    for v_idx, plan in enumerate(plans_to_run):
                        ov = (overrides_by_visit or {}).get(plan.visit_name)
                        excl = (excluded_by_visit or {}).get(plan.visit_name)

                        def fin_cb(val, status, _b=v_idx, _t=total):
                            schedule_progress((_b + val) / _t, status)

                        ok = finalize_visit(
                            plan,
                            patient_name=patient_name,
                            patient_output_dir=patient_output_dir,
                            apply_clahe_to_ref=apply_clahe_val,
                            automate_tuning=auto_tuning_val,
                            matrix_overrides=ov,
                            source_layer=focus_layer,
                            target_layers=None,  # all layers of this Visit
                            excluded_captures=excl,
                            persist_overrides=True,
                            progress_callback=fin_cb,
                            log_callback=schedule_log,
                        )
                        if not ok:
                            schedule_log(f"ERROR finalizing visit {plan.visit_name}.", color=ft.Colors.RED_400)
                            break
                    else:
                        success = True
                        schedule_log(
                            f"Saved all layers for "
                            f"{', '.join(p.visit_name for p in plans_to_run)} "
                            f"→ {patient_output_dir}."
                        )
            except Exception as exc:
                schedule_log(f"ERROR: {exc}", color=ft.Colors.RED_400)
            corrections = {
                v: sorted(d.keys()) for v, d in (overrides_by_visit or {}).items() if d
            }
            for v, caps in (excluded_by_visit or {}).items():
                if caps:
                    corrections.setdefault(v, [])
                    # mark exclusions distinctly in summary via negative? keep separate later
            review_state["last_corrections"] = corrections if success else None
            page.run_task(ui_complete, success, True)

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
            ], spacing=12),
            ft.Text(
                "After registration finishes, a review screen lets you inspect "
                "image1–image4 and start Review & Correct for a selected image.",
                size=12,
                color=ft.Colors.GREY_500,
            ),
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
