from fastapi.openapi.docs import (
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse


def get_swagger_ui_html() -> HTMLResponse:
    """
    Generate the HTML response for a self-hosted Swagger UI.

    This function creates the HTML page that loads Swagger UI using local static
    files (CSS and JS) instead of relying on a CDN.

    Returns:
        HTMLResponse: The HTML page for the Swagger UI.
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <link type="text/css" rel="stylesheet" href="/static/swagger-ui.css">
        <link rel="shortcut icon" href="/static/favicon.png">
        <title>Swagger UI</title>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="/static/swagger-ui-bundle.js"></script>
        <!-- `SwaggerUIBundle` is now available on the page -->
        <script>
        const ui = SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            presets: [SwaggerUIBundle.presets.apis],
        })
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)