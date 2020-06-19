from flask import jsonify, request
from app import app
from app.models import db
from app.models import Photo, CraftPhoto

import numpy as np
import cv2

from craft import run_craft_image

@app.route('/craft', methods=['POST'])
def run_craft():
    id = 2
    binStrImg = request.data 
    arrImg = np.fromstring(binStrImg, np.uint8)
    img = cv2.imdecode(arrImg, cv2.IMREAD_COLOR)

    root_photo = Photo.query.get(id)

    List_path_ROI = run_craft_image(img)
    for path_ROI in List_path_ROI:
        p = CraftPhoto(link=path_ROI, root=root_photo)
        db.session.add(p)
        db.session.commit()

    return jsonify(id=id, List_path_ROI=List_path_ROI)