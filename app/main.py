import flet as ft
from app.ui.dashboard import create_dashboard

async def main(page: ft.Page):
    # 1. Native Window Configuration (Premium Feel)
    page.title = "ARIAKE OCTA Stack Registration"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1100
    page.window.height = 800
    page.window.min_width = 800
    page.window.min_height = 600
    
    # Custom Theme with modern color palette
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.Colors.CYAN_400,
            secondary=ft.Colors.AMBER_400,
            surface=ft.Colors.GREY_900,
            on_surface=ft.Colors.WHITE,
        ),
        visual_density=ft.VisualDensity.COMFORTABLE,
    )
    
    # 2. Root container acting as a simple view host so the dashboard can swap in
    #    the manual corresponding-point correction screen and back.
    root = ft.Container(expand=True)

    def mount_view(control):
        """Swap the visible root content. Safe to call from the page event loop."""
        root.content = control
        try:
            root.update()
        except Exception:
            # Fallback when the control tree is mid-rebuild.
            try:
                page.update()
            except Exception:
                pass

    dashboard = create_dashboard(page, mount_view=mount_view)
    root.content = dashboard

    # 3. Add to page
    page.add(root)
    page.update()

if __name__ == "__main__":
    # Start the Flet application
    ft.run(main)
