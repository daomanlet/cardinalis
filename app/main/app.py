"""Flask Application"""

# load libaries
import sys
import os 
from pathlib import Path
from flask import Flask, jsonify

# load modules
# init Flask app
#app = Flask(__name__)

# register blueprints. ensure that all paths are versioned!
#app.register_blueprint(bird_species, url_prefix="/api/v1/bird")

def create_app():
    app = Flask(__name__)
    path = os.getcwd()
    with app.app_context():
        app.config['model_path'] = os.path.join(path, 'deploy/mobilenet_v2.h5')
        from service.bird_species import bird_species
    app.register_blueprint(bird_species, url_prefix="/api/v1/bird")
    #     model = init_model()
    #     app.blueprints['bird_species']
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=51480)
