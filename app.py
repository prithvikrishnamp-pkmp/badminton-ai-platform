import streamlit as st
import pandas as pd
import os
import random
import tempfile
import pose_analysis

st.set_page_config(page_title="AI Badminton Coach", layout="wide")

st.title("🏸 AI Badminton Coaching System")

# =====================================================
# 🧑 STEP 1 — PLAYER PROFILE SYSTEM
# =====================================================

st.header("🧑 Player Information")

name = st.text_input("Player Name")
age = st.number_input("Age", 8, 50, 18)
height = st.number_input("Height (cm)", 120, 220, 170)
weight = st.number_input("Weight (kg)", 30, 120, 65)

fitness = st.slider("Fitness Level", 1, 10, 5)
stamina = st.slider("Stamina", 1, 10, 5)
speed = st.slider("Speed / Agility", 1, 10, 5)
footwork = st.slider("Footwork Skill", 1, 10, 5)
smash = st.slider("Smash Power", 1, 10, 5)
reaction = st.slider("Reaction Speed", 1, 10, 5)

# ---------------- REALISTIC AI SKILL MODEL ----------------

movement_score = (speed + footwork) / 2
endurance_score = (stamina + fitness) / 2
attack_score = smash

overall_score = movement_score*0.45 + endurance_score*0.35 + attack_score*0.20

if movement_score < 5:
    level = "Beginner (movement foundation weak)"
    months = 8
elif overall_score < 4:
    level = "Beginner"
    months = 8
elif overall_score < 7:
    level = "Intermediate"
    months = 4
else:
    level = "Advanced"
    months = 1

# ---------------- PLAY STYLE AI ----------------

singles_score = speed*0.35 + stamina*0.30 + footwork*0.20 + fitness*0.15
doubles_score = smash*0.35 + reaction*0.30 + speed*0.20 + fitness*0.15

if singles_score > doubles_score:
    play_style = "Better suited for Singles"
else:
    play_style = "Better suited for Doubles"

# ---------------- EQUIPMENT DATABASE ----------------

racquets = {
    "Beginner": ["Yonex Nanoray Light 18i", "Li-Ning Windstorm 72", "Victor Ultramate 8", "Apacs Finapi 232"],
    "Intermediate": ["Yonex Astrox 77 Play", "Victor Brave Sword 12", "Li-Ning Turbo Charging 75", "Apacs Ziggler LHI Pro"],
    "Advanced": ["Yonex Astrox 100 ZZ", "Yonex Arcsaber 11 Pro", "Victor Thruster F Enhanced", "Li-Ning Aeronaut 9000C"]
}

shoes = {
    "Beginner": ["Yonex SHB 65X", "Li-Ning Ranger Lite", "Victor A170", "Asics Gel Rocket 10"],
    "Intermediate": ["Yonex SHB 65Z3", "Victor P9200", "Li-Ning Ranger IV", "Asics Gel Blade 8"],
    "Advanced": ["Yonex SHB Eclipsion Z3", "Victor P9200II", "Li-Ning Ranger VI", "Asics Court FF 3"]
}

base_level = "Beginner" if "Beginner" in level else level
suggested_racquet = random.choice(racquets[base_level])
suggested_shoes = random.choice(shoes[base_level])

# ---------------- DISPLAY STEP 1 RESULTS ----------------

st.subheader("📊 AI Player Analysis")

st.success(f"Current Level: {level}")
st.write(f"⏳ Estimated Time to Next Level: **{months} months**")
st.write(f"🎯 Play Style: **{play_style}**")

st.subheader("🏸 Equipment Recommendation")
st.write(f"Racquet: **{suggested_racquet}**")
st.write(f"Shoes: **{suggested_shoes}**")

# ---------------- SAVE PLAYER PROFILE ----------------

if st.button("💾 Save Player Profile"):
    data = {
        "Name": name,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Fitness": fitness,
        "Stamina": stamina,
        "Speed": speed,
        "Footwork": footwork,
        "Smash": smash,
        "Reaction": reaction,
        "Level": level,
        "Months_to_Improve": months,
        "Play_Style": play_style,
        "Racquet": suggested_racquet,
        "Shoes": suggested_shoes
    }

    df = pd.DataFrame([data])

    if os.path.exists("player_progress.csv"):
        df.to_csv("player_progress.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("player_progress.csv", index=False)

    st.success("Player profile saved successfully!")

# =====================================================
# 🎥 STEP 2 — MOVEMENT & INJURY ANALYSIS
# =====================================================

st.header("🎥 Movement Analysis")
balance = 0
posture = 0
footwork = 0

video = st.file_uploader("Upload playing video", type=["mp4", "mov"])

if video is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    stframe = st.empty()
    st.info("Analyzing movement...")

    balance, posture, footwork = pose_analysis.analyze_video(tfile.name, stframe)

    st.subheader("📊 Movement Results")

    st.write("Balance:", round(balance,3))
    st.write("Posture:", round(posture,3))
    st.write("Footwork Speed:", round(footwork,2))

    # -------- WEAKNESS DETECTION --------
    st.subheader("⚠ Weakness Analysis")

    if balance > 0.1:
        st.warning("Poor body balance")

    if posture > 0.15:
        st.warning("Bad spine posture")

    if footwork < 15:
        st.warning("Slow footwork")

    # -------- INJURY RISK --------
    st.subheader("🩹 Injury Risk")

    if posture > 0.2:
        st.error("Risk of lower back injury")

    if balance > 0.15:
        st.error("Risk of ankle/knee strain")

    # -------- CORRECTION PLAN --------
    st.subheader("🏋 Correction Training Plan")

    if footwork < 15:
        st.write("• Ladder drills (10 mins daily)")
        st.write("• Multi-shuttle footwork practice")

    if posture > 0.15:
        st.write("• Core workouts (plank, superman hold)")
        st.write("• Back strengthening exercises")

    if balance > 0.1:
        st.write("• Single leg balance drills")
        st.write("• Shadow badminton movement")

    # =====================================================
# 📈 STEP 3 — PLAYER PROGRESS TRACKING
# =====================================================

st.header("📈 Player Progress Dashboard")

if os.path.exists("player_progress.csv"):
    data = pd.read_csv("player_progress.csv")

    st.subheader("📋 All Player Records")
    st.dataframe(data)

    # Select player
    players = data["Name"].unique()
    selected_player = st.selectbox("Select Player to View Progress", players)

    player_data = data[data["Name"] == selected_player]

    st.subheader(f"📊 Progress of {selected_player}")

    st.line_chart(player_data[["Fitness", "Stamina", "Speed", "Footwork", "Smash"]])

    st.subheader("🎯 Skill Level Over Time")
    st.write(player_data[["Level", "Months_to_Improve"]])

else:
    st.info("No player data saved yet.")


   
    # =====================================================
    # 🧠 AI WEEKLY TRAINING PLAN GENERATOR
    # =====================================================

    st.subheader("📅 AI Weekly Training Plan")

    plan = []

    # Movement focus
    if footwork < 15:
        plan.append("Footwork ladder drills – 15 mins daily")
        plan.append("Multi-shuttle court movement drills")

    # Posture focus
    if posture > 0.15:
        plan.append("Core strengthening (planks, superman holds)")
        plan.append("Back posture correction exercises")

    # Balance focus
    if balance > 0.1:
        plan.append("Single-leg balance drills")
        plan.append("Shadow badminton with split-step focus")

    # Level-based conditioning
    if "Beginner" in level:
        plan.append("Basic stamina building – light jogging 20 mins")
    elif "Intermediate" in level:
        plan.append("Interval running – 30 mins")
    else:
        plan.append("High-intensity match simulation drills")

    # Injury prevention
    if posture > 0.2:
        plan.append("Lower back stretching routine")
    if balance > 0.15:
        plan.append("Ankle strengthening & stability exercises")

    # Display plan
    if plan:
        for item in plan:
            st.write("•", item)
    else:
        st.success("Player movement is balanced. Maintain current training.")

