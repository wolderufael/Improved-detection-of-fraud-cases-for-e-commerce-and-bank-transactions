# from app_instance import app  # Import the app instance
# from layout import create_layout  # Import the layout
# # from form import form_layout
# import callbacks  # Import callbacks to register them


# # Assign the layout from layout.py
# app.layout = create_layout(app)
# # app.form=form_layout(app)
# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

from app_instance import app  # Import the app instance
from layout import create_layout  # Import the layout
import callbacks  # Import callbacks to register them
import os  # Import os to access environment variables

# Assign the layout from layout.py
app.layout = create_layout(app)

# Run the app on the specified port and host
if __name__ == '__main__':
    # Use Render's dynamically assigned port
    port = int(os.getenv("PORT", 8000))  # Fallback to 8000 if $PORT is not set
    app.run_server(host="0.0.0.0", port=port, debug=True)
