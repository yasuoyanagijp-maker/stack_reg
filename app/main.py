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
    
    # 2. Main Dashboard Layout
    dashboard = create_dashboard(page)
    
    # 3. Add to page
    page.add(dashboard)
    page.update()

if __name__ == "__main__":
    # Start the Flet application
    ft.run(main)
