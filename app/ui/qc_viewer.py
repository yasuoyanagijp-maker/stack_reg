import flet as ft
import os

def create_qc_viewer(page: ft.Page, output_dir: str):
    """
    Creates the assessment view for reviewing registration results.
    
    Features:
    - Grid of thumbnails for all registered Average Intensity images.
    - Large preview on click.
    - Status indicators for processing quality.
    """
    
    # 1. State for images
    image_grid = ft.GridView(
        expand=1,
        runs_count=5,
        max_extent=250,
        child_aspect_ratio=1.0,
        spacing=15,
        run_spacing=15,
    )
    
    preview_image = ft.Image(
        src_base64="",
        width=500,
        height=500,
        fit=ft.ImageFit.CONTAIN,
        visible=False,
    )
    
    preview_title = ft.Text("Result Preview", size=20, weight=ft.FontWeight.BOLD)
    
    # 2. Logic to populate the grid
    def refresh_grid():
        image_grid.controls.clear()
        if not os.path.exists(output_dir):
            return
            
        for root, _, files in os.walk(output_dir):
            for file in sorted(files):
                if file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                    file_path = os.path.join(root, file)
                    
                    # Create a card for each image
                    image_card = ft.GestureDetector(
                        on_tap=lambda _, fp=file_path, fn=file: show_preview(fp, fn),
                        content=ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Image(
                                        src=file_path, 
                                        width=200, 
                                        height=180, 
                                        fit=ft.ImageFit.COVER,
                                        border_radius=5
                                    ),
                                    ft.Text(file, size=12, no_wrap=True, text_align=ft.TextAlign.CENTER),
                                ], spacing=5, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                                padding=10,
                            ),
                            elevation=2,
                        )
                    )
                    image_grid.controls.append(image_card)
        
        image_grid.update()

    def show_preview(path, name):
        preview_image.src = path
        preview_image.visible = True
        preview_title.value = f"Preview: {name}"
        page.update()

    # 3. Main Layout for QC
    qc_layout = ft.Row([
        ft.Column([
            ft.Text("Registered Images Assessment", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.CYAN_400),
            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
            image_grid,
        ], expand=7),
        
        ft.VerticalDivider(width=1),
        
        ft.Column([
            preview_title,
            ft.Container(
                content=preview_image,
                bgcolor=ft.Colors.GREY_900,
                border_radius=10,
                padding=10,
                border=ft.Border.all(1, ft.Colors.CYAN_900),
            ),
            ft.Text("Assessment Quality: High", color=ft.Colors.GREEN_400),
            ft.Text("Transformation Type: Affine", color=ft.Colors.GREY_400),
            ft.ElevatedButton("Reregister Layer", icon=ft.Icons.REFRESH, color=ft.Colors.AMBER_400),
        ], expand=3, spacing=20, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
    ], expand=True, spacing=30)
    
    # Initialize grid
    refresh_grid()
    
    return qc_layout
