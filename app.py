from flask import (
    Flask,
    request,
    jsonify,
    send_file
)
from flask_cors import CORS

import os
import io
import zipfile
import random
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import cloudinary
import cloudinary.uploader
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

load_dotenv()
print("=" * 60)
print("Starting BAE Backend...")
print("=" * 60)

# =======================================
# FLASK APP
# =======================================
app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)
print("✓ Flask Initialized")

# =======================================
# CLOUDINARY CONFIG
# =======================================

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)
print("✓ Cloudinary Configured")

# =======================================
# MONGODB ATLAS
# =======================================
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)
db = client["baeDB"]
client.admin.command("ping")

print("✓ MongoDB Connected")

users_collection = db["users"]
wardrobe_collection = db["wardrobe"]
favourites_collection = db["favourites"]

# =======================================
# MODEL PATHS AND LABELS
# =======================================
MOOD_MODEL_PATH = "models/mood_model/mobilenetv2_mood_2class.tflite"
MOOD_LABELS = ['happy', 'neutral']

OUTFIT_MODEL_PATH = "models/outfit_model/mobilenetv2_top_bottom.tflite"

mood_interpreter = None
mood_input_details = None
mood_output_details = None

outfit_interpreter = None
outfit_input_details = None
outfit_output_details = None

# =======================================
# TENSORFLOW LITE UTILITIES
# =======================================

def load_interpreter(model_path):
    """
    Loads a TensorFlow Lite model once and returns:
    interpreter,
    input tensor details,
    output tensor details
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

# =======================================
# LOAD MOOD MODEL
# =======================================

def get_mood_interpreter():
    global mood_interpreter
    global mood_input_details
    global mood_output_details

    if mood_interpreter is None:
        print("Loading Mood TFLite Model...")
        (
            mood_interpreter,
            mood_input_details,
            mood_output_details
        ) = load_interpreter(MOOD_MODEL_PATH)

        print("✓ Mood Model Loaded")

    return (
        mood_interpreter,
        mood_input_details,
        mood_output_details
    )

# =======================================
# LOAD OUTFIT MODEL
# =======================================

def get_outfit_interpreter():
    global outfit_interpreter
    global outfit_input_details
    global outfit_output_details

    if outfit_interpreter is None:
        print("Loading Outfit TFLite Model...")
        (
            outfit_interpreter,
            outfit_input_details,
            outfit_output_details
        ) = load_interpreter(OUTFIT_MODEL_PATH)
        print("✓ Outfit Model Loaded")

    return (
        outfit_interpreter,
        outfit_input_details,
        outfit_output_details
    )

# =======================================
# IMAGE PREPROCESSING
# =======================================

def preprocess_for_mood(img):
    img = img.resize((224, 224))
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_for_outfit(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# =======================================
# GENERIC TFLITE INFERENCE
# =======================================

def run_tflite(
    interpreter,
    input_details,
    output_details,
    input_data
):
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_dtype = input_details[0]["dtype"]

    if input_data.dtype != input_dtype:
        input_data = input_data.astype(input_dtype)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)

    return prediction

# =======================================
# BASIC ROUTES
# =======================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "BAE Backend Running",
        "status": "success",
        "version": "1.0"
    })

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint used by Render.
    """
    try:
        client.admin.command("ping")

        return jsonify({
            "status": "healthy",
            "database": "connected",
            "server": "running"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }), 500

# =======================================
# USER SIGNUP
# =======================================

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No data received"
            }), 400

        username = data.get("username", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()

        if not username or not email or not password:
            return jsonify({
                "success": False,
                "message": "All fields are required."
            }), 400

        if not email.endswith("@thapar.edu"):
            return jsonify({
                "success": False,
                "message": "Only @thapar.edu email addresses are allowed."
            }), 400

        existing_user = users_collection.find_one({"email": email})

        if existing_user:
            return jsonify({
                "success": False,
                "message": "Email already registered."
            }), 400

        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password
        })

        return jsonify({
            "success": True,
            "full_name": username,
            "email": email
        }), 201

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# =======================================
# USER LOGIN
# =======================================

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "No data received"
            }), 400

        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({
                "success": False,
                "message": "Email and password are required."
            }), 400

        user = users_collection.find_one({
            "email": email,
            "password": password
        })

        if not user:
            return jsonify({
                "success": False,
                "message": "Invalid credentials."
            }), 401

        return jsonify({
            "success": True,
            "full_name": user["username"],
            "email": user["email"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# =======================================
# GET PROFILE
# =======================================

@app.route("/get_profile", methods=["GET"])
def get_profile():
    try:
        email = request.args.get("email", "").strip().lower()

        if not email:
            return jsonify({
                "success": False,
                "message": "Email is required."
            }), 400

        user = users_collection.find_one(
            {"email": email},
            {"_id": 0, "password": 0}
        )

        if user is None:
            return jsonify({
                "success": False,
                "message": "User not found."
            }), 404

        return jsonify({
            "success": True,
            "user": user
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# =======================================
# UPDATE PROFILE
# =======================================

@app.route("/update_profile", methods=["POST"])
def update_profile():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No data received."
            }), 400

        email = data.get("email", "").strip().lower()
        username = data.get("username", "").strip()

        if not email or not username:
            return jsonify({
                "success": False,
                "message": "Email and username are required."
            }), 400

        result = users_collection.update_one(
            {"email": email},
            {
                "$set": {
                    "username": username
                }
            }
        )

        if result.matched_count == 0:
            return jsonify({
                "success": False,
                "message": "User not found."
            }), 404

        return jsonify({
            "success": True,
            "username": username
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# =======================================
# MOOD DETECTION
# =======================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400

        file = request.files["image"]
        img = Image.open(file).convert("RGB")
        x = preprocess_for_mood(img)
        (
            interpreter,
            input_details,
            output_details
        ) = get_mood_interpreter()

        prediction = run_tflite(
            interpreter,
            input_details,
            output_details,
            x
        )

        prediction = prediction[0]
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        mood = MOOD_LABELS[predicted_index]

        return jsonify({
            "success": True,
            "mood": mood,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# CLOUDINARY UPLOAD WITH BACKGROUND REMOVAL
# =======================================

@app.route("/upload-image", methods=["POST"])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        ext = file.filename.rsplit(".", 1)[-1].lower()

        allowed_extensions = {
            "png",
            "jpg",
            "jpeg",
            "webp"
        }

        if ext not in allowed_extensions:
            return jsonify({
                "success": False,
                "error": f"Unsupported file type: {ext}"
            }), 400
        
        print("Uploading image...")

        upload = cloudinary.uploader.upload(
            file,
            folder="wardrobe_items",
            public_id=os.path.splitext(file.filename)[0],
            overwrite=True,
            resource_type="image"
        )

        image_url = upload["secure_url"]
        print("Upload Successful")

        return jsonify({
            "success": True,
            "message": "Image uploaded successfully.",
            "url": image_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# ADD WARDROBE ITEM
# =======================================

@app.route("/wardrobe/add", methods=["POST"])
def add_wardrobe():
    try:
        print("=" * 60)
        print("NEW WARDROBE REQUEST")
        print("=" * 60)
        # -----------------------------
        # Validate Request
        # -----------------------------
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400
        file = request.files["image"]
        user_id = request.form.get("userId")

        if not user_id:
            return jsonify({
                "success": False,
                "error": "Missing userId."
            }), 400

        ext = file.filename.rsplit(".", 1)[-1].lower()

        if ext not in {"png", "jpg", "jpeg", "webp"}:
            return jsonify({
                "success": False,
                "error": "Unsupported image type."
            }), 400
        print("Reading uploaded image...")

        img = Image.open(file).convert("RGBA")

        print("Preparing image buffer...")

        buffer = io.BytesIO()

        img.save(buffer, format="PNG")

        buffer.seek(0)

        print("Uploading image to Cloudinary...")

        upload = cloudinary.uploader.upload(
            buffer,
            folder="wardrobe_items",
            public_id=os.path.splitext(file.filename)[0],
            overwrite=True,
            resource_type="image"
        )
        image_url = upload["secure_url"]
        public_id = upload["public_id"]
        print("Cloudinary upload complete.")
        # --------------------------------------------
        # TensorFlow Lite Outfit Prediction
        # --------------------------------------------
        print("Running outfit prediction...")
        img_arr = np.array(img)
        img_arr = cv2.cvtColor(
            img_arr,
            cv2.COLOR_RGBA2BGR
        )
        x = preprocess_for_outfit(img_arr)
        (
            interpreter,
            input_details,
            output_details
        ) = get_outfit_interpreter()

        prediction = run_tflite(
            interpreter,
            input_details,
            output_details,
            x
        )

        score = float(prediction[0][0])

        THRESHOLD = 0.5
        if score >= THRESHOLD:
            predicted_class = "Bottomwear"
        else:
            predicted_class = "Topwear"

        print(f"Prediction Score: {score:.6f} -> {predicted_class}")

        wardrobe_collection.insert_one({
            "userId": user_id,
            "imageUrl": image_url,
            "publicId": public_id,
            "category": predicted_class,
            "deleted": False,
            "createdAt": datetime.utcnow()
        })

        print("Wardrobe item stored successfully.")

        return jsonify({
            "success": True,
            "message": "Wardrobe item added successfully.",
            "imageUrl": image_url,
            "predicted_category": predicted_class
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# GET WARDROBE
# =======================================

@app.route("/wardrobe/all", methods=["GET"])
def get_wardrobe():
    try:
        user_id = request.args.get("userId", "").strip()
        if not user_id:
            return jsonify({
                "success": False,
                "error": "Missing userId."
            }), 400

        items = list(
            wardrobe_collection.find(
                {
                    "userId": user_id,
                    "deleted": False
                }
            ).sort("createdAt", -1)
        )

        results = []

        for item in items:
            results.append({
                "id": str(item["_id"]),
                "userId": item.get("userId"),
                "imageUrl": item.get("imageUrl"),
                "category": item.get("category"),
                "deleted": item.get("deleted", False),
                "createdAt": item.get("createdAt")
            })

        return jsonify({
            "success": True,
            "items": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =======================================
# SOFT DELETE
# =======================================

@app.route("/wardrobe/delete", methods=["POST"])
def delete_wardrobe_item():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No data received."
            }), 400

        user_id = data.get("userId")
        item_id = data.get("itemId")

        if not user_id or not item_id:
            return jsonify({
                "success": False,
                "error": "Missing required fields."
            }), 400

        result = wardrobe_collection.update_one(
            {
                "_id": ObjectId(item_id),
                "userId": user_id
            },
            {
                "$set": {
                    "deleted": True
                }
            }
        )

        if result.matched_count == 0:
            return jsonify({
                "success": False,
                "error": "Item not found."
            }), 404

        return jsonify({
            "success": True,
            "message": "Item deleted successfully."
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# GET TRASH
# =======================================

@app.route("/wardrobe/trash", methods=["GET"])
def get_trash():
    try:
        user_id = request.args.get("userId", "").strip()

        if not user_id:
            return jsonify({
                "success": False,
                "error": "Missing userId."
            }), 400

        items = list(
            wardrobe_collection.find(
                {
                    "userId": user_id,
                    "deleted": True
                }
            ).sort("createdAt", -1)
        )

        results = []

        for item in items:
            results.append({
                "id": str(item["_id"]),
                "userId": item.get("userId"),
                "imageUrl": item.get("imageUrl"),
                "category": item.get("category"),
                "deleted": True,
                "createdAt": item.get("createdAt")
            })

        return jsonify({
            "success": True,
            "items": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# RESTORE ITEM
# =======================================

@app.route("/wardrobe/restore", methods=["POST"])
def restore_wardrobe_item():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No data received."
            }), 400

        user_id = data.get("userId")
        item_id = data.get("itemId")

        if not user_id or not item_id:
            return jsonify({
                "success": False,
                "error": "Missing required fields."
            }), 400

        result = wardrobe_collection.update_one(
            {
                "_id": ObjectId(item_id),
                "userId": user_id
            },
            {
                "$set": {
                    "deleted": False
                }
            }
        )

        if result.matched_count == 0:
            return jsonify({
                "success": False,
                "error": "Item not found."
            }), 404

        return jsonify({
            "success": True,
            "message": "Item restored successfully."
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# PERMANENT DELETE
# =======================================

@app.route("/wardrobe/deletePermanent", methods=["POST"])
def delete_wardrobe_item_permanent():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No data received."
            }), 400

        user_id = data.get("userId")
        item_id = data.get("itemId")

        if not user_id or not item_id:
            return jsonify({
                "success": False,
                "error": "Missing required fields."
            }), 400

        # Find the item first
        item = wardrobe_collection.find_one({
            "_id": ObjectId(item_id),
            "userId": user_id
        })

        if not item:
            return jsonify({
                "success": False,
                "error": "Item not found."
            }), 404

        # Delete from Cloudinary if publicId exists
        if item.get("publicId"):
            cloudinary.uploader.destroy(item["publicId"])

        # Delete from MongoDB
        wardrobe_collection.delete_one({
            "_id": ObjectId(item_id),
            "userId": user_id
        })

        return jsonify({
            "success": True,
            "message": "Item permanently deleted."
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# OUTFIT RECOMMENDATION / GENERATOR
# =======================================

def recommend_outfit(wardrobe_items):
    def normalize(category):
        return category.lower().replace(" ", "").strip()

    tops = []
    bottoms = []

    for item in wardrobe_items:
        category = normalize(item.get("category", ""))
        if category in {
            "topwear",
            "top",
            "shirt",
            "tshirt",
            "upper"
        }:
            tops.append(item)

        elif category in {
            "bottomwear",
            "bottom",
            "pant",
            "pants",
            "jeans",
            "trouser"
        }:
            bottoms.append(item)

    random.shuffle(tops)
    random.shuffle(bottoms)

    if not tops and not bottoms:
        return None

    if not tops:
        return {
            "topwear": None,
            "bottomwear": bottoms[0]
        }

    if not bottoms:
        return {
            "topwear": tops[0],
            "bottomwear": None
        }

    return {
        "topwear": tops[0],
        "bottomwear": bottoms[0]
    }

# =======================================
# GENERATE NEXT OUTFIT
# =======================================

@app.route("/generator/next", methods=["GET"])
def next_outfit():
    try:
        user_id = request.args.get("userId", "").strip()

        if not user_id:
            return jsonify({
                "success": False,
                "error": "User ID is required."
            }), 400

        wardrobe_items = list(
            wardrobe_collection.find({
                "userId": user_id,
                "deleted": False
            })
        )

        if len(wardrobe_items) == 0:
            return jsonify({
                "topwear": None,
                "bottomwear": None
            })

        outfit = recommend_outfit(wardrobe_items)

        if outfit is None:
            return jsonify({
                "topwear": None,
                "bottomwear": None
            })

        return jsonify({
            "topwear":
                outfit["topwear"]["imageUrl"]
                if outfit["topwear"] else None,

            "bottomwear":
                outfit["bottomwear"]["imageUrl"]
                if outfit["bottomwear"] else None
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# SAVE FAVOURITE
# =======================================

@app.route("/saveFavourite", methods=["POST"])
def save_favourite():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No data received."
            }), 400

        user_id = data.get("userId", "").strip()
        topwear = data.get("topwear")
        bottomwear = data.get("bottomwear")

        if not user_id or not topwear or not bottomwear:
            return jsonify({
                "success": False,
                "error": "Missing required fields."
            }), 400

        # Prevent duplicate favourites
        existing = favourites_collection.find_one({
            "userId": user_id,
            "topwear": topwear,
            "bottomwear": bottomwear
        })

        if existing:
            return jsonify({
                "success": True,
                "message": "Already saved."
            })

        favourites_collection.insert_one({
            "userId": user_id,
            "topwear": topwear,
            "bottomwear": bottomwear,
            "createdAt": datetime.utcnow()
        })

        return jsonify({
            "success": True,
            "message": "Favourite saved successfully."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# GET FAVOURITES
# =======================================

@app.route("/getFavourites", methods=["GET"])
def get_favourites():
    try:
        user_id = request.args.get("userId", "").strip()
        if not user_id:
            return jsonify({
                "success": False,
                "error": "userId is required."
            }), 400

        favourites = list(
            favourites_collection.find({
                "userId": user_id
            }).sort("createdAt", -1)
        )

        results = []

        for item in favourites:
            results.append({
                "id": str(item["_id"]),
                "topwear": item.get("topwear"),
                "bottomwear": item.get("bottomwear"),
                "createdAt": item.get("createdAt")
            })

        return jsonify({
            "success": True,
            "favourites": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# REMOVE FAVOURITE
# =======================================

from bson import ObjectId

@app.route("/removeFavourite", methods=["POST"])
def remove_favourite():
    try:
        data = request.get_json()

        favourite_id = data.get("id")

        if not favourite_id:
            return jsonify({
                "success": False,
                "error": "Favourite id is required."
            }), 400

        result = favourites_collection.delete_one({
            "_id": ObjectId(favourite_id)
        })

        if result.deleted_count == 0:
            return jsonify({
                "success": False,
                "error": "Favourite not found."
            }), 404

        return jsonify({
            "success": True,
            "message": "Favourite removed successfully."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================================
# RUN SERVER
# =======================================
if __name__ == "__main__":
    app.run(port=5000, debug=True)
