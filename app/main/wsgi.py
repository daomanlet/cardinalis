"""Web Server Gateway Interface"""

##################
# FOR PRODUCTION
####################
from app import create_app
application = create_app()