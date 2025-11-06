# BAE — Bringing Aesthetics to Emotions

**BAE** is a smart wardrobe web application that merges **emotion recognition** and **fashion intelligence** to help users manage their wardrobe and generate outfit recommendations tailored to their mood and preferences.

This repository represents the **MVP (Minimum Viable Product)** version — focused on **authentication**, **mood detection**, and the initial dashboard UI integration.

---

## Current Features (MVP)

### User Authentication
- Users can **sign up** and **log in** using their credentials.
- Authentication data is securely stored in **MongoDB Atlas**.

### Mood Detection System
- After login, users lands Dashboard where they se **Mood Detection** option.
- Two options available:
  1. **Detect via Webcam:** Capture a real-time image to detect mood (Happy, Neutral, or Sad).
  2. **Select Manually:** Choose mood manually if webcam access is denied.
- Based on the detected/selected mood, the **UI theme dynamically changes**.
- The captured image is **not stored** to maintain privacy.

### Interactive Dashboard
- Displays current mood and quick actions.
- Serves as the central hub for navigation.
- (Upcoming features like wardrobe view, upload, and outfit generator are placeholder sections for now.)

---

## Upcoming Features (Final Version Plan)

- **Upload Clothes:**  
  Users will upload images of their clothing items.  
  → Backend **MobileNetV2** model will classify them as **Topwear** or **Bottomwear** or **Footwear** or **Accessories**.

- **Add Details:**  
  After classification, following tagging will be done:
  - Gender: *Men / Women*  
  - Season: *Summer / Fall / Spring / Winter*  
  - Usage: *Casual / Sports / Formal / Ethnic*

- **Outfit Generator:**  
  The system will generate topwear–bottomwear combinations based on compatibility and user preferences.  
  Users can save favorite outfits or discard others.

- **My Wardrobe:**  
  Displays all uploaded items.  
  (Currently under development.)

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | React.js (UI built using Figma design) |
| **Backend** | Flask (Python) |
| **Machine Learning** | TensorFlow / Keras |
| **Database** | MongoDB Atlas (Cloud) |

---

## Current Flow

1. **User Signup / Login** → Credentials stored in MongoDB.  
2. **Dashboard Loads** → Option for mood detection appears.  
3. **User Chooses Mood Detection Option** →  
   - Webcam captures mood image (processed via Flask backend),  
   - or mood is selected manually.  
4. **UI Updates Dynamically** → Based on mood (Happy, Neutral, Sad).  
5. **Next Steps (Coming Soon)** → Upload clothes → classify → generate outfits.


---

## Setup Instructions (Local Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/roopakjeetkaur/BAE--Bringing-Aesthetics-to-Emotions.git
   cd BAE--Bringing-Aesthetics-to-Emotions
